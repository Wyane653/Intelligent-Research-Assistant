# src/agent.py
from typing import List, Dict, Any, Optional, ClassVar
from langchain_core.prompts import PromptTemplate
from langchain.tools import BaseTool
import logging
import re
import time
import sys
import json
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path
import sqlite3

# 动态导入以兼容不同环境
# 由于SimplifiedResearchAssistant不需要内存，我们定义一个简单的空类占位
class ConversationBufferMemory:
    def __init__(self, memory_key="chat_history", return_messages=True, output_key="output"):
        self.memory_key = memory_key
        # 简化实现，无需实际功能
        pass
    def clear(self):
        pass
    def load_memory_variables(self, inputs):
        return {}

from config.settings import settings

logger = logging.getLogger(__name__)

# ==================== ReAct解析器 ====================

class ReActParser:
    """
    强化版ReAct格式解析器。
    核心策略：提取第一个'最终答案：'标记后的内容，并暴力截断其后所有可能标志新循环开始的文本。
    """
    # 定义所有可能表示"新一轮开始"的触发器
    BREAK_TRIGGERS: ClassVar[List[str]] = [
        "\n思考：", "\nThought：", "\n行动：", "\nAction:",
        "\n观察：", "\nObservation:", "Invalid Format:",
        "思考：", "Thought："  # 也检查行首的情况
    ]
    FINAL_MARKERS: ClassVar[List[str]] = ["最终答案：", "Final Answer:", "Answer:"]

    @classmethod
    def parse_react_output(cls, output: str) -> str:
        """解析原始输出，返回清理后的最终答案。"""
        if not output or output == "没有获取到回答":
            return output

        logging.debug(f"[解析器] 输入长度: {len(output)} 字符")

        # 1. 寻找第一个出现的最终答案标记
        first_marker_pos = -1
        used_marker = None
        for marker in cls.FINAL_MARKERS:
            pos = output.find(marker)
            if pos != -1:
                first_marker_pos = pos
                used_marker = marker
                break

        if first_marker_pos != -1 and used_marker:
            # 提取第一个标记之后的所有内容
            content_after_marker = output[first_marker_pos + len(used_marker):].strip()
            logging.debug(f"[解析器] 找到标记 '{used_marker}', 其后内容长度: {len(content_after_marker)}")

            # 2. 暴力截断：在剩余内容中查找任何表示新循环开始的标记
            earliest_break = len(content_after_marker)  # 初始化为末尾
            for trigger in cls.BREAK_TRIGGERS:
                pos = content_after_marker.find(trigger)
                if pos != -1 and pos < earliest_break:
                    earliest_break = pos
                    logging.debug(f"[解析器] 发现截断点 '{trigger}' 于位置 {pos}")

            # 截取第一个截断点之前的内容（如果没找到，则earliest_break为末尾）
            final_answer = content_after_marker[:earliest_break].strip()
            logging.debug(f"[解析器] 截断后答案长度: {len(final_answer)}")
            return final_answer

        # 3. 如果没有找到任何最终标记，则清理掉所有明显的ReAct内部行
        logging.debug("[解析器] 未找到最终标记，执行行级清理。")
        lines = output.split('\n')
        cleaned_lines = []
        skip_keywords = ["思考：", "Thought:", "行动：", "Action:", "观察：", "Observation:", "行动输入："]
        for line in lines:
            if not any(line.strip().startswith(kw) for kw in skip_keywords):
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines).strip()

# ==================== 记忆模块核心组件 ====================

@dataclass
class ConversationMemory:
    """单次对话记忆"""
    query: str
    response: str
    timestamp: str
    topic: str = "general"
    tools_used: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)

class ResearchMemory:
    """研究记忆管理器 - 完全基于SQLite，无需额外依赖"""
    
    def __init__(self, user_id: str = "default", db_path: str = "./research_memory.db"):
        self.user_id = user_id
        self.db_path = db_path
        self.current_topic = "general"
        
        # 初始化数据库
        self._init_database()
        logger.info(f"[记忆] 记忆管理器初始化完成，用户: {user_id}")
    
    def _init_database(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建对话记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            topic TEXT DEFAULT 'general',
            tools_used TEXT DEFAULT '',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 创建研究主题表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS research_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            topic_name TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, topic_name)
        )
        ''')
        
        # 创建关键发现表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS key_findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            finding TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 创建用户偏好表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT DEFAULT '',
            UNIQUE(user_id, key)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def remember_conversation(self, query: str, response: str, 
                            tools_used: List[str] = None, topic: str = None):
        """记住一次对话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        topic_to_use = topic or self.current_topic
        tools_str = ",".join(tools_used) if tools_used else ""
        
        cursor.execute('''
        INSERT INTO conversations (user_id, query, response, topic, tools_used)
        VALUES (?, ?, ?, ?, ?)
        ''', (self.user_id, query, response, topic_to_use, tools_str))
        
        # 更新主题的最后访问时间
        if topic_to_use != "general":
            cursor.execute('''
            INSERT OR REPLACE INTO research_topics 
            (user_id, topic_name, last_accessed)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (self.user_id, topic_to_use))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"[记忆] 已保存对话: {query[:50]}...")
        return True
    
    def set_research_topic(self, topic: str, description: str = ""):
        """设置或创建研究主题"""
        self.current_topic = topic
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO research_topics 
        (user_id, topic_name, description, last_accessed)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (self.user_id, topic, description))
        
        conn.commit()
        conn.close()
        
        logger.info(f"[记忆] 研究主题已设置: {topic}")
        return topic
    
    def get_conversation_history(self, limit: int = 5, topic: str = None) -> List[Dict]:
        """获取对话历史"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if topic:
            cursor.execute('''
            SELECT query, response, topic, timestamp 
            FROM conversations 
            WHERE user_id = ? AND topic = ?
            ORDER BY timestamp DESC 
            LIMIT ?
            ''', (self.user_id, topic, limit))
        else:
            cursor.execute('''
            SELECT query, response, topic, timestamp 
            FROM conversations 
            WHERE user_id = ?
            ORDER BY timestamp DESC 
            LIMIT ?
            ''', (self.user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def add_key_finding(self, finding: str, topic: str = None):
        """添加关键发现"""
        topic_to_use = topic or self.current_topic
        
        if not finding or len(finding) < 10:
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO key_findings (user_id, topic, finding)
        VALUES (?, ?, ?)
        ''', (self.user_id, topic_to_use, finding[:500]))  # 限制长度
        
        conn.commit()
        conn.close()
        
        logger.debug(f"[记忆] 已添加关键发现: {finding[:100]}...")
        return True
    
    def get_topic_summary(self, topic: str = None) -> Dict:
        """获取主题摘要"""
        topic_to_use = topic or self.current_topic
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 获取主题信息
        cursor.execute('''
        SELECT topic_name, description, created_at, last_accessed
        FROM research_topics
        WHERE user_id = ? AND topic_name = ?
        ''', (self.user_id, topic_to_use))
        
        topic_info = cursor.fetchone()
        
        # 获取对话数量
        cursor.execute('''
        SELECT COUNT(*) as count FROM conversations
        WHERE user_id = ? AND topic = ?
        ''', (self.user_id, topic_to_use))
        
        conv_count = cursor.fetchone()["count"]
        
        # 获取关键发现
        cursor.execute('''
        SELECT finding FROM key_findings
        WHERE user_id = ? AND topic = ?
        ORDER BY created_at DESC
        LIMIT 5
        ''', (self.user_id, topic_to_use))
        
        findings = [row["finding"] for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "topic": topic_to_use,
            "info": dict(topic_info) if topic_info else {"topic_name": topic_to_use, "description": ""},
            "conversation_count": conv_count,
            "key_findings": findings
        }
    
    def search_conversations(self, keyword: str, limit: int = 5) -> List[Dict]:
        """简单关键词搜索"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        search_term = f"%{keyword}%"
        cursor.execute('''
        SELECT query, response, topic, timestamp 
        FROM conversations 
        WHERE user_id = ? AND (query LIKE ? OR response LIKE ?)
        ORDER BY timestamp DESC 
        LIMIT ?
        ''', (self.user_id, search_term, search_term, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def clear_topic(self, topic: str):
        """清除特定主题的记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        DELETE FROM conversations 
        WHERE user_id = ? AND topic = ?
        ''', (self.user_id, topic))
        
        cursor.execute('''
        DELETE FROM research_topics 
        WHERE user_id = ? AND topic_name = ?
        ''', (self.user_id, topic))
        
        cursor.execute('''
        DELETE FROM key_findings 
        WHERE user_id = ? AND topic = ?
        ''', (self.user_id, topic))
        
        conn.commit()
        conn.close()
        
        logger.info(f"[记忆] 已清除主题: {topic}")
        return True
    
    def get_stats(self) -> Dict:
        """获取记忆统计"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 对话总数
        cursor.execute('''
        SELECT COUNT(*) as total FROM conversations WHERE user_id = ?
        ''', (self.user_id,))
        total_convs = cursor.fetchone()["total"]
        
        # 主题列表
        cursor.execute('''
        SELECT topic_name, description FROM research_topics 
        WHERE user_id = ? AND topic_name != 'general'
        ORDER BY last_accessed DESC
        ''', (self.user_id,))
        topics = [dict(row) for row in cursor.fetchall()]
        
        # 关键发现总数
        cursor.execute('''
        SELECT COUNT(*) as findings_count FROM key_findings WHERE user_id = ?
        ''', (self.user_id,))
        findings_count = cursor.fetchone()["findings_count"]
        
        conn.close()
        
        return {
            "total_conversations": total_convs,
            "total_findings": findings_count,
            "research_topics": topics,
            "current_topic": self.current_topic,
            "db_path": self.db_path,
            "user_id": self.user_id
        }
    
    def get_recent_context(self, limit: int = 3, topic: str = None) -> str:
        """获取最近的对话上下文"""
        conversations = self.get_conversation_history(limit=limit, topic=topic)
        if not conversations:
            return ""
        
        context = ""
        for i, conv in enumerate(conversations, 1):
            context += f"{i}. Q: {conv['query'][:80]}{'...' if len(conv['query']) > 80 else ''}\n"
            context += f"   A: {conv['response'][:120]}{'...' if len(conv['response']) > 120 else ''}\n"
        
        return context
    
    def set_preference(self, key: str, value: str):
        """设置用户偏好"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO user_preferences (user_id, key, value)
        VALUES (?, ?, ?)
        ''', (self.user_id, key, value))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"[记忆] 已设置偏好: {key}={value}")
        return True
    
    def get_preference(self, key: str, default: str = "") -> str:
        """获取用户偏好"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT value FROM user_preferences 
        WHERE user_id = ? AND key = ?
        ''', (self.user_id, key))
        
        row = cursor.fetchone()
        conn.close()
        
        return row["value"] if row else default

# ==================== 研究助手类（带记忆功能） ====================

class ResearchAssistant:
    """研究助手代理 - 集成记忆模块"""
    
    def __init__(self, tools: List[BaseTool], model_name: str = None, 
                 enable_memory: bool = True, user_id: str = "default"):
        """
        初始化研究助手。
        :param tools: 可用的工具列表
        :param model_name: 模型名称，默认为配置中的DEEPSEEK_MODEL
        :param enable_memory: 是否启用记忆功能
        :param user_id: 用户ID，用于记忆隔离
        """
        self.tools = tools
        self.model_name = model_name or settings.DEEPSEEK_MODEL
        self.enable_memory = enable_memory
        self.user_id = user_id
        
        # 诊断：实例创建记录
        self._instance_id = int(time.time() * 1000) % 10000
        logging.info(f"[助手初始化] 实例ID: {self._instance_id}, 工具数: {len(tools)}")
        
        # 初始化记忆系统
        if enable_memory:
            self.memory = ResearchMemory(user_id=user_id)
            logger.info(f"[记忆] 记忆功能已启用，用户: {user_id}")
        else:
            self.memory = None
            logger.info("[记忆] 记忆功能已禁用")

        # 动态导入ChatOpenAI
        self.llm = self._create_llm()

        # 保持原有的简单内存（用于LangChain代理）
        self.langchain_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        self.prompt = self._create_prompt()

        # 创建代理（注意：这里需要根据您的实际环境调整导入）
        try:
            from langchain.agents import create_react_agent, AgentExecutor
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.prompt
            )
            
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=self.langchain_memory,
                verbose=settings.DEBUG_MODE,
                handle_parsing_errors=False,
                max_iterations=3,
                early_stopping_method="generate",
                return_intermediate_steps=True,
                max_execution_time=45,
            )
            logging.info(f"[助手初始化] AgentExecutor 配置完成")
        except ImportError as e:
            logger.warning(f"无法导入LangChain Agent相关模块: {e}")
            logger.warning("将使用简化模式，AgentExecutor功能不可用")
            self.agent_executor = None
    
    def _create_llm(self):
        """创建LLM实例 - 动态导入解决兼容性问题"""
        # 尝试从多个可能的路径导入
        ChatOpenAIClass = None
        potential_imports = [
            ('langchain_community.chat_models', 'ChatOpenAI'),
            ('langchain.chat_models', 'ChatOpenAI'),
            ('langchain_openai', 'ChatOpenAI')
        ]
        
        for module_path, class_name in potential_imports:
            try:
                module = __import__(module_path, fromlist=[class_name])
                ChatOpenAIClass = getattr(module, class_name)
                logger.info(f"✅ 成功从 '{module_path}' 导入 '{class_name}'")
                break
            except (ImportError, AttributeError):
                continue
        
        if ChatOpenAIClass is None:
            error_msg = "无法导入 ChatOpenAI。请安装必要的包：pip install langchain-community langchain-openai"
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        # 创建LLM实例
        return ChatOpenAIClass(
            model=self.model_name,
            temperature=settings.TEMPERATURE,
            openai_api_key=settings.DEEPSEEK_API_KEY,
            openai_api_base=settings.DEEPSEEK_API_BASE,
            streaming=False,
            max_tokens=800,
        )
    
    def _create_prompt(self) -> PromptTemplate:
        """创建提示词模板 - 集成记忆上下文"""
        # 如果有记忆系统，构建上下文
        memory_context = ""
        if self.memory:
            memory_context = self.memory.get_recent_context(limit=3)
            if memory_context:
                memory_context = "【相关历史对话】\n" + memory_context + "\n"
            
            # 获取当前主题摘要
            if self.memory.current_topic != "general":
                summary = self.memory.get_topic_summary()
                if summary.get('key_findings'):
                    memory_context += f"【研究主题：{self.memory.current_topic}】\n"
                    memory_context += "关键发现：\n"
                    for finding in summary['key_findings'][:3]:
                        memory_context += f"- {finding[:80]}{'...' if len(finding) > 80 else ''}\n"
                    memory_context += "\n"
        
        template = f"""你是一个专业的研究助手。请严格按照以下格式思考和回答问题：

思考：[分析问题，决定是否需要使用工具。如果需要，说明使用哪个工具。]
行动：[工具名称，必须是 {{tool_names}} 中的一个] 或 [直接回答]
行动输入：[查询输入] 或 [无]
观察：[工具返回的结果] 或 [无]
（此思考-行动-观察循环最多重复3次）
最终答案：[基于所有信息的完整、简洁回答]

**重要规则**：
1.  '最终答案：' 必须是你的最后一行输出，之后不要生成任何文本。
2.  如果不需要工具，流程为：思考 -> 行动：直接回答 -> 行动输入：无 -> 观察：无 -> 最终答案：[你的回答]

**历史记忆**：
{memory_context if memory_context else "无相关历史记忆"}

可用工具：
{{tools}}

之前的对话记录：
{{chat_history}}

现在，请回答以下问题：
问题：{{input}}

开始：{{agent_scratchpad}}"""
        return PromptTemplate.from_template(template)
    
    def set_research_topic(self, topic: str, description: str = ""):
        """设置或切换研究主题"""
        if self.memory:
            self.memory.set_research_topic(topic, description)
            return f"研究主题已设置为: {topic}"
        return "记忆功能未启用"
    
    def query(self, question: str, remember: bool = True, 
             research_topic: str = None) -> Dict[str, Any]:
        """
        向助手提问。
        :param question: 问题
        :param remember: 是否记忆本次对话
        :param research_topic: 研究主题（可选）
        """
        query_start_time = time.time()
        query_tracker = f"{self._instance_id}_{int(query_start_time * 1000) % 10000}"
        
        logging.info(f"[查询开始] 跟踪号: {query_tracker}, 问题: '{question[:60]}...'")
        
        # 设置研究主题
        if research_topic and self.memory:
            self.memory.set_research_topic(research_topic, "用户指定的研究主题")
        
        # 如果没有AgentExecutor，使用简化模式
        if not self.agent_executor:
            return self._simple_query(question, remember)
        
        try:
            # 执行查询
            result = self.agent_executor.invoke({"input": question})
            
            raw_answer = result.get("output", "")
            parsed_answer = ReActParser.parse_react_output(raw_answer)
            final_answer = self._post_clean_answer(parsed_answer)
            
            # 记忆本次对话
            if self.memory and remember:
                # 提取使用的工具
                intermediate_steps = result.get("intermediate_steps", [])
                tools_used = []
                for step in intermediate_steps:
                    if isinstance(step, tuple) and len(step) > 0:
                        action = step[0]
                        if hasattr(action, 'tool'):
                            tools_used.append(action.tool)
                
                # 保存到记忆
                self.memory.remember_conversation(
                    query=question,
                    response=final_answer,
                    tools_used=tools_used
                )
                
                # 如果回答简短且有价值，作为关键发现
                if len(final_answer) < 200 and len(final_answer) > 20:
                    # 简单启发式：判断是否是重要发现
                    important_keywords = ["发现", "结论", "表明", "证明", "应该", "建议", "因为", "因此"]
                    if any(keyword in final_answer for keyword in important_keywords):
                        self.memory.add_key_finding(final_answer)
            
            total_duration = time.time() - query_start_time
            logging.info(f"[查询{query_tracker}] 处理完成，总耗时: {total_duration:.2f}s")
            
            return {
                "answer": final_answer,
                "intermediate_steps": result.get("intermediate_steps", []),
                "success": True,
                "memory_used": self.memory is not None,
                "research_topic": self.memory.current_topic if self.memory else None
            }
            
        except Exception as e:
            logging.error(f"[查询{query_tracker}] 执行失败: {str(e)}", exc_info=True)
            return self._simple_query(question, remember)
    
    def _simple_query(self, question: str, remember: bool = True) -> Dict[str, Any]:
        """简化查询模式（当AgentExecutor不可用时）"""
        try:
            # 如果有记忆，添加历史上下文
            context = ""
            if self.memory and remember:
                context = self.memory.get_recent_context(limit=2)
                if context:
                    context = "相关历史对话：\n" + context + "\n"
            
            # 构建提示词
            prompt = f"{context}请回答以下问题：{question}\n\n请提供专业、准确的回答。"
            
            # 调用LLM
            response = self.llm.invoke(prompt)
            final_answer = response.content if hasattr(response, 'content') else str(response)
            
            # 记忆本次对话
            if self.memory and remember:
                self.memory.remember_conversation(question, final_answer)
                
                # 如果回答简短且有价值，作为关键发现
                if len(final_answer) < 200 and len(final_answer) > 20:
                    important_keywords = ["发现", "结论", "表明", "证明", "应该", "建议", "因为", "因此"]
                    if any(keyword in final_answer for keyword in important_keywords):
                        self.memory.add_key_finding(final_answer)
            
            return {
                "answer": final_answer,
                "intermediate_steps": [],
                "success": True,
                "memory_used": self.memory is not None,
                "research_topic": self.memory.current_topic if self.memory else None
            }
        except Exception as e:
            logging.error(f"[简化查询] 执行失败: {str(e)}")
            return {
                "answer": f"处理请求时出错: {str(e)}",
                "success": False
            }
    
    def _post_clean_answer(self, answer: str) -> str:
        """对解析后的答案进行后处理（去重、截断等）。"""
        if not answer:
            return answer

        # 1. 简单句子去重（基于前60个字符的简化哈希）
        sentences = re.split(r'[。！？；\n]', answer)
        seen = set()
        unique_sentences = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            key = s[:60]
            if key not in seen:
                seen.add(key)
                unique_sentences.append(s)

        cleaned = '。'.join(unique_sentences)
        if cleaned and not cleaned.endswith('。'):
            cleaned += '。'

        # 2. 长度限制
        max_len = getattr(settings, 'MAX_ANSWER_LENGTH', 600)
        if len(cleaned) > max_len:
            # 在句子边界处截断
            sentences_again = re.split(r'[。！？]', cleaned)
            result = []
            current_len = 0
            for s in sentences_again:
                s = s.strip()
                if not s:
                    continue
                s_with_punct = s + '。'
                if current_len + len(s_with_punct) <= max_len:
                    result.append(s_with_punct)
                    current_len += len(s_with_punct)
                else:
                    break
            if result:
                cleaned = ''.join(result)
                if len(result) < len(sentences_again):
                    cleaned += " (内容已精简)"

        return cleaned
    
    def get_conversation_history(self, limit: int = 5, topic: str = None):
        """获取对话历史"""
        if self.memory:
            return self.memory.get_conversation_history(limit, topic)
        return []
    
    def search_memory(self, keyword: str, limit: int = 5):
        """搜索记忆"""
        if self.memory:
            return self.memory.search_conversations(keyword, limit)
        return []
    
    def get_topic_summary(self, topic: str = None):
        """获取主题摘要"""
        if self.memory:
            return self.memory.get_topic_summary(topic)
        return {}
    
    def add_key_finding(self, finding: str):
        """添加关键发现"""
        if self.memory:
            self.memory.add_key_finding(finding)
            return True
        return False
    
    def get_memory_stats(self):
        """获取记忆统计"""
        if self.memory:
            return self.memory.get_stats()
        return {"message": "记忆功能未启用"}
    
    def clear_topic_memory(self, topic: str):
        """清除特定主题的记忆"""
        if self.memory:
            self.memory.clear_topic(topic)
            return True
        return False
    
    def export_memory(self, output_file: str = "memory_export.json"):
        """导出记忆为JSON文件"""
        if not self.memory:
            return False
        
        try:
            # 获取所有数据
            stats = self.memory.get_stats()
            topics = stats.get("research_topics", [])
            
            data = {
                "user_id": self.user_id,
                "export_time": datetime.now().isoformat(),
                "stats": stats,
                "topics": []
            }
            
            # 为每个主题获取详细信息
            for topic_info in topics:
                topic_name = topic_info["topic_name"]
                topic_data = self.memory.get_topic_summary(topic_name)
                
                # 获取该主题的对话
                conversations = self.memory.get_conversation_history(
                    limit=50,
                    topic=topic_name
                )
                
                topic_data["conversations"] = conversations
                data["topics"].append(topic_data)
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[记忆] 记忆已导出到: {output_file}")
            return True
        except Exception as e:
            logger.error(f"[记忆] 导出失败: {e}")
            return False
    
    def set_preference(self, key: str, value: str):
        """设置用户偏好"""
        if self.memory:
            return self.memory.set_preference(key, value)
        return False
    
    def get_preference(self, key: str, default: str = ""):
        """获取用户偏好"""
        if self.memory:
            return self.memory.get_preference(key, default)
        return default

# ==================== 简化版研究助手（带记忆） ====================

class SimplifiedResearchAssistant:
    """
    极简版研究助手。不依赖复杂的AgentExecutor，直接使用LLM和工具。
    可作为ResearchAssistant出问题时的备选方案。
    """
    def __init__(self, tools: List[BaseTool], llm=None, 
                 enable_memory: bool = True, user_id: str = "default"):
        self.tools = {tool.name: tool for tool in tools}
        self.user_id = user_id
        
        # 初始化记忆
        if enable_memory:
            self.memory = ResearchMemory(user_id=user_id)
        else:
            self.memory = None
        
        # 动态解析并导入 ChatOpenAI，确保稳定性
        ChatOpenAIClass = None
        if llm is not None:
            # 如果外部传入了llm，直接使用
            self.llm = llm
            return
            
        # 尝试从多个可能的路径导入
        potential_imports = [
            ('langchain_community.chat_models', 'ChatOpenAI'),
            ('langchain.chat_models', 'ChatOpenAI'),
            ('langchain_openai', 'ChatOpenAI')
        ]
        
        for module_path, class_name in potential_imports:
            try:
                module = __import__(module_path, fromlist=[class_name])
                ChatOpenAIClass = getattr(module, class_name)
                print(f"✅ [SimplifiedAssistant] 成功从 '{module_path}' 导入 '{class_name}'")
                break
            except (ImportError, AttributeError):
                continue
        
        if ChatOpenAIClass is None:
            error_msg = """
                ❌ 无法导入 ChatOpenAI。这通常是由于 LangChain 安装不完整。
                请在你的虚拟环境中运行以下命令来安装必要的包：
                    pip install langchain-community langchain-openai
                或者，如果你希望安装较旧的稳定版本：
                    pip install langchain==0.0.348 langchain-community==0.0.10

                安装后，请重新启动程序。
                        """
            print(error_msg)
            raise ImportError("ChatOpenAI 模块未找到，请检查 LangChain 安装。")
        
        # 成功导入后，创建实例
        self.llm = ChatOpenAIClass(
            model=settings.DEEPSEEK_MODEL,
            temperature=0.2,
            openai_api_key=settings.DEEPSEEK_API_KEY,
            openai_api_base=settings.DEEPSEEK_API_BASE,
            max_tokens=600,
        )
        print(f"✅ [SimplifiedAssistant] 语言模型实例创建成功，使用模型: {settings.DEEPSEEK_MODEL}")

    def query(self, question: str, remember: bool = True) -> Dict[str, Any]:
        """简化的查询流程：让LLM决定是否使用工具，然后生成答案。"""
        logging.info(f"[简化助手] 处理问题: {question[:50]}...")
        
        # 如果有记忆，添加上下文
        context = ""
        if self.memory and remember:
            context = self.memory.get_recent_context(limit=2)
            if context:
                context = "相关历史对话：\n" + context + "\n"
        
        try:
            # 步骤1: 决定是否需要工具
            decision_prompt = f"""{context}用户问题：{question}
可用工具：{list(self.tools.keys())}

请判断是否需要使用工具来回答这个问题？
如果**需要**，请严格按以下格式回复：
NEED_TOOL: YES
TOOL_NAME: [工具名称]
TOOL_INPUT: [输入内容]

如果**不需要**，请直接回答问题。"""
            decision = self.llm.invoke(decision_prompt).content

            final_answer = ""
            if "NEED_TOOL: YES" in decision:
                # 解析工具信息
                tool_name, tool_input = None, None
                for line in decision.split('\n'):
                    if line.startswith('TOOL_NAME:'):
                        tool_name = line.replace('TOOL_NAME:', '').strip()
                    elif line.startswith('TOOL_INPUT:'):
                        tool_input = line.replace('TOOL_INPUT:', '').strip()
                if tool_name and tool_input and tool_name in self.tools:
                    tool_result = self.tools[tool_name].run(tool_input)
                    # 基于结果生成答案
                    answer_prompt = f"问题：{question}\n相关背景信息：{tool_result[:1000]}\n请基于以上信息给出简洁回答："
                    final_answer = self.llm.invoke(answer_prompt).content
                else:
                    final_answer = "抱歉，工具调用失败。"
            else:
                # 不需要工具，直接使用决策文本作为答案（或清理一下）
                final_answer = decision.replace("NEED_TOOL: NO", "").strip()

            # 记忆本次对话
            if self.memory and remember:
                tools_used = [tool_name] if "NEED_TOOL: YES" in decision else []
                self.memory.remember_conversation(question, final_answer, tools_used)
                
                # 如果回答简短且有价值，作为关键发现
                if len(final_answer) < 200 and len(final_answer) > 20:
                    important_keywords = ["发现", "结论", "表明", "证明", "应该", "建议", "因为", "因此"]
                    if any(keyword in final_answer for keyword in important_keywords):
                        self.memory.add_key_finding(final_answer)

            return {"answer": final_answer, "success": True}

        except Exception as e:
            logging.error(f"[简化助手] 错误: {e}")
            return {"answer": f"简化流程出错: {e}", "success": False}
    
    def set_research_topic(self, topic: str, description: str = ""):
        """设置研究主题"""
        if self.memory:
            self.memory.set_research_topic(topic, description)
            return f"研究主题已设置为: {topic}"
        return "记忆功能未启用"
    
    def get_conversation_history(self, limit: int = 5):
        """获取对话历史"""
        if self.memory:
            return self.memory.get_conversation_history(limit)
        return []

# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("研究助手记忆模块测试")
    print("=" * 60)
    
    # 创建一个简单的工具示例（实际使用时替换为您的工具）
    class TestTool(BaseTool):
        name = "test_tool"
        description = "测试工具"
        def _run(self, query: str) -> str:
            return f"测试工具返回: {query}"
    
    # 创建带记忆的研究助手
    assistant = ResearchAssistant(
        tools=[TestTool()],
        enable_memory=True,
        user_id="test_user"
    )
    
    # 设置研究主题
    print("\n1. 设置研究主题...")
    result = assistant.set_research_topic("人工智能伦理", "研究AI的公平性、透明性和责任")
    print(f"   {result}")
    
    # 进行几次查询
    print("\n2. 进行查询...")
    questions = [
        "什么是算法偏见？",
        "如何检测算法偏见？",
        "有哪些解决算法偏见的方法？"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n   Q{i}: {question}")
        result = assistant.query(question)
        if result["success"]:
            answer = result["answer"]
            print(f"   A{i}: {answer[:80]}...")
        else:
            print(f"   错误: {result['answer']}")
    
    # 获取对话历史
    print("\n3. 获取对话历史...")
    history = assistant.get_conversation_history(limit=3)
    print(f"   最近 {len(history)} 条对话:")
    for i, conv in enumerate(history, 1):
        print(f"   {i}. [{conv['topic']}] {conv['query'][:50]}...")
    
    # 获取主题摘要
    print("\n4. 获取主题摘要...")
    summary = assistant.get_topic_summary()
    print(f"   主题: {summary.get('topic', 'N/A')}")
    print(f"   对话数: {summary.get('conversation_count', 0)}")
    print(f"   关键发现: {len(summary.get('key_findings', []))} 条")
    
    # 搜索记忆
    print("\n5. 搜索记忆...")
    search_results = assistant.search_memory("偏见", limit=2)
    print(f"   搜索到 {len(search_results)} 条相关对话")
    
    # 获取统计信息
    print("\n6. 获取记忆统计...")
    stats = assistant.get_memory_stats()
    print(f"   总对话数: {stats.get('total_conversations', 0)}")
    print(f"   研究主题数: {len(stats.get('research_topics', []))}")
    print(f"   当前主题: {stats.get('current_topic', 'N/A')}")
    
    # 导出记忆
    print("\n7. 导出记忆...")
    if assistant.export_memory("test_memory_export.json"):
        print("   记忆已导出到 test_memory_export.json")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)