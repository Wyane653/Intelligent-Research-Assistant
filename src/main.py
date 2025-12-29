import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Optional, Dict, Any
import logging
from colorama import init, Fore, Style

from config.settings import settings
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.tools import create_tools

# 修改后的导入语句
try:
    from src.agent import ResearchAssistant, SimplifiedResearchAssistant
except ImportError:
    try:
        from src.agent_final import ResearchAssistant, SimplifiedResearchAssistant
    except ImportError:
        print("无法导入ResearchAssistant，请检查依赖包安装")
        exit(1)

from src.research_writer import ResearchWriter
from src.utils import setup_logging
init(autoreset=True)

class IntelligentResearchAssistant:
    """智能研究助手主类"""
    
    def __init__(self, use_memory: bool = True, user_id: str = "default"):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        if not settings.DEEPSEEK_API_KEY:
            print(Fore.YELLOW + "警告: 未设置DeepSeek API密钥。请设置DEEPSEEK_API_KEY环境变量。")
        
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.vector_store = None
        
        self.tools = create_tools(
            vector_store_manager=self.vector_store_manager,
            search_api_key=os.getenv("SERPAPI_API_KEY")
        )
        
        # 创建研究助手实例（带记忆功能）
        try:
            print(Fore.CYAN + "正在初始化研究助手...")
            self.assistant = ResearchAssistant(
                tools=self.tools,
                enable_memory=use_memory,
                user_id=user_id
            )
            if use_memory:
                print(Fore.GREEN + "✅ [系统] 已启用记忆功能")
                print(Fore.CYAN + f"   用户ID: {user_id}")
                print(Fore.CYAN + f"   记忆数据库: research_memory.db")
            else:
                print(Fore.YELLOW + "⚠️  [系统] 记忆功能已禁用")
        except Exception as e:
            print(Fore.RED + f"❌ [系统] ResearchAssistant初始化失败: {e}")
            print(Fore.YELLOW + "正在回退到SimplifiedResearchAssistant...")
            self.assistant = SimplifiedResearchAssistant(
                tools=self.tools,
                enable_memory=use_memory,
                user_id=user_id
            )
            print(Fore.GREEN + "✅ [系统] 已切换至SimplifiedResearchAssistant")
        
        self.writer = ResearchWriter(self.assistant.llm)
        
        self.logger.info("智能研究助手初始化完成")
        
        # 显示记忆状态
        if use_memory and hasattr(self.assistant, 'get_memory_stats'):
            stats = self.assistant.get_memory_stats()
            if isinstance(stats, dict) and "message" not in stats:
                print(Fore.CYAN + f"📊 记忆状态: {stats.get('total_conversations', 0)} 条历史对话")
    
    def load_documents(self, path: str):
        """加载文档到知识库"""
        print(Fore.CYAN + f"正在加载文档: {path}")
        
        if os.path.isfile(path):
            documents = self.document_processor.load_document(path)
            if documents:
                documents = self.document_processor.chunk_documents(documents)
        elif os.path.isdir(path):
            documents = self.document_processor.process_directory(path)
        else:
            print(Fore.RED + f"路径不存在: {path}")
            return False
        
        if not documents:
            print(Fore.RED + "未找到可处理的文档")
            return False
        
        print(Fore.GREEN + f"成功加载 {len(documents)} 个文档块")
        
        self.vector_store = self.vector_store_manager.create_vector_store(documents)
        return True
    
    def ask_question(self, question: str, research_topic: str = None) -> Dict[str, Any]:
        """提问并获取回答"""
        print(Fore.YELLOW + f"\n问题: {question}")
        print(Fore.CYAN + "思考中...\n")
        
        # 如果有研究主题，先设置主题
        if research_topic and hasattr(self.assistant, 'set_research_topic'):
            self.assistant.set_research_topic(research_topic, f"关于{research_topic}的研究")
            print(Fore.CYAN + f"📌 研究主题已设置为: {research_topic}")
        
        result = self.assistant.query(question)
        
        if result["success"]:
            print(Fore.GREEN + "\n" + "="*50)
            print(Fore.GREEN + "回答:")
            print(Fore.GREEN + "="*50)
            print(Fore.WHITE + result["answer"])
            print(Fore.GREEN + "="*50)
            
            # 显示记忆相关信息
            if result.get("memory_used", False):
                print(Fore.CYAN + f"💾 本次对话已保存到记忆")
            if result.get("research_topic") and result["research_topic"] != "general":
                print(Fore.CYAN + f"📚 研究主题: {result['research_topic']}")
        else:
            print(Fore.RED + f"错误: {result['answer']}")
        
        return result
    
    def interactive_chat(self):
        """交互式聊天模式 - 增强版，支持记忆操作命令"""
        print(Fore.CYAN + "="*60)
        print(Fore.CYAN + "智能研究助手已启动 (输入 '退出' 或 'quit' 结束)")
        print(Fore.CYAN + "="*60)
        print(Fore.YELLOW + "可用命令:")
        print(Fore.WHITE + "  /topic [主题] - 设置研究主题")
        print(Fore.WHITE + "  /history [数量] - 查看历史对话")
        print(Fore.WHITE + "  /stats - 查看记忆统计")
        print(Fore.WHITE + "  /search [关键词] - 搜索记忆")
        print(Fore.WHITE + "  /clear - 清除当前主题记忆")
        print(Fore.WHITE + "  /export - 导出记忆")
        print(Fore.CYAN + "="*60)
        
        current_topic = "general"
        
        while True:
            try:
                user_input = input(Fore.YELLOW + "\n您的问题/命令: " + Style.RESET_ALL)
                
                if user_input.lower() in ['退出', 'quit', 'exit', 'q']:
                    print(Fore.CYAN + "再见！")
                    break
                
                # 处理命令
                if user_input.startswith('/'):
                    self._handle_command(user_input, current_topic)
                    continue
                
                if user_input.strip():
                    result = self.assistant.query(user_input)
                    
                    if result["success"]:
                        print(Fore.GREEN + "\n" + "="*50)
                        print(Fore.GREEN + "回答:")
                        print(Fore.GREEN + "="*50)
                        print(Fore.WHITE + result["answer"])
                        print(Fore.GREEN + "="*50)
                        
                        # 更新当前主题
                        if result.get("research_topic"):
                            current_topic = result["research_topic"]
                    else:
                        print(Fore.RED + f"错误: {result['answer']}")
                        
            except KeyboardInterrupt:
                print(Fore.CYAN + "\n\n再见！")
                break
            except Exception as e:
                print(Fore.RED + f"发生错误: {str(e)}")
    
    def _handle_command(self, command: str, current_topic: str):
        """处理记忆相关命令"""
        parts = command.strip().split()
        cmd = parts[0].lower()
        
        if cmd == "/topic" and len(parts) > 1:
            topic = " ".join(parts[1:])
            if hasattr(self.assistant, 'set_research_topic'):
                result = self.assistant.set_research_topic(topic, f"关于{topic}的研究")
                print(Fore.GREEN + result)
                current_topic = topic
            else:
                print(Fore.RED + "当前助手不支持设置研究主题")
        
        elif cmd == "/history":
            limit = 5
            if len(parts) > 1:
                try:
                    limit = int(parts[1])
                except:
                    pass
            
            if hasattr(self.assistant, 'get_conversation_history'):
                history = self.assistant.get_conversation_history(limit=limit, topic=current_topic)
                if history:
                    print(Fore.CYAN + f"\n📜 最近 {len(history)} 条对话记录:")
                    for i, conv in enumerate(history, 1):
                        print(Fore.YELLOW + f"{i}. [{conv.get('topic', 'general')}]")
                        print(Fore.WHITE + f"   问: {conv.get('query', '')[:80]}...")
                        print(Fore.WHITE + f"   答: {conv.get('response', '')[:100]}...")
                        if conv.get('timestamp'):
                            print(Fore.CYAN + f"   时间: {conv['timestamp'][:19]}")
                        print()
                else:
                    print(Fore.YELLOW + "暂无历史对话记录")
            else:
                print(Fore.RED + "当前助手不支持查看历史对话")
        
        elif cmd == "/stats":
            if hasattr(self.assistant, 'get_memory_stats'):
                stats = self.assistant.get_memory_stats()
                if isinstance(stats, dict):
                    if "message" in stats:
                        print(Fore.YELLOW + stats["message"])
                    else:
                        print(Fore.CYAN + "\n📊 记忆统计:")
                        print(Fore.WHITE + f"   总对话数: {stats.get('total_conversations', 0)}")
                        print(Fore.WHITE + f"   关键发现数: {stats.get('total_findings', 0)}")
                        print(Fore.WHITE + f"   研究主题数: {len(stats.get('research_topics', []))}")
                        print(Fore.WHITE + f"   当前主题: {stats.get('current_topic', 'general')}")
                        print(Fore.WHITE + f"   用户ID: {stats.get('user_id', 'default')}")
                        
                        topics = stats.get('research_topics', [])
                        if topics:
                            print(Fore.CYAN + "\n   研究主题列表:")
                            for topic in topics[:5]:  # 最多显示5个
                                print(Fore.WHITE + f"     - {topic.get('topic_name', '')}: {topic.get('description', '')[:50]}...")
                else:
                    print(Fore.YELLOW + "无法获取记忆统计")
            else:
                print(Fore.RED + "当前助手不支持记忆统计")
        
        elif cmd == "/search" and len(parts) > 1:
            keyword = " ".join(parts[1:])
            if hasattr(self.assistant, 'search_memory'):
                results = self.assistant.search_memory(keyword, limit=3)
                if results:
                    print(Fore.CYAN + f"\n🔍 找到 {len(results)} 条相关记忆:")
                    for i, result in enumerate(results, 1):
                        print(Fore.YELLOW + f"{i}. [{result.get('topic', 'general')}]")
                        print(Fore.WHITE + f"   问: {result.get('query', '')[:80]}...")
                        print(Fore.WHITE + f"   答: {result.get('response', '')[:100]}...")
                        if result.get('timestamp'):
                            print(Fore.CYAN + f"   时间: {result['timestamp'][:19]}")
                        print()
                else:
                    print(Fore.YELLOW + f"未找到包含 '{keyword}' 的相关记忆")
            else:
                print(Fore.RED + "当前助手不支持记忆搜索")
        
        elif cmd == "/clear":
            if hasattr(self.assistant, 'clear_topic_memory'):
                if current_topic and current_topic != "general":
                    confirm = input(Fore.RED + f"确认清除主题 '{current_topic}' 的所有记忆吗？(y/N): ")
                    if confirm.lower() == 'y':
                        success = self.assistant.clear_topic_memory(current_topic)
                        if success:
                            print(Fore.GREEN + f"已清除主题 '{current_topic}' 的记忆")
                        else:
                            print(Fore.RED + "清除失败")
                else:
                    print(Fore.YELLOW + "当前没有设置研究主题")
            else:
                print(Fore.RED + "当前助手不支持清除记忆")
        
        elif cmd == "/export":
            if hasattr(self.assistant, 'export_memory'):
                filename = f"memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                success = self.assistant.export_memory(filename)
                if success:
                    print(Fore.GREEN + f"记忆已导出到: {filename}")
                else:
                    print(Fore.RED + "导出失败")
            else:
                print(Fore.RED + "当前助手不支持导出记忆")
        
        else:
            print(Fore.RED + f"未知命令: {cmd}")
            print(Fore.YELLOW + "可用命令: /topic, /history, /stats, /search, /clear, /export")
    
    def generate_research_report(self, topic: str, questions: list):
        """生成研究报告"""
        print(Fore.CYAN + f"开始生成研究报告: {topic}")
        
        # 设置研究主题
        if hasattr(self.assistant, 'set_research_topic'):
            self.assistant.set_research_topic(topic, f"关于{topic}的研究报告")
            print(Fore.CYAN + f"📌 研究主题已设置为: {topic}")
        
        research_data = {
            "summary": "",
            "background": "",
            "findings": [],
            "analysis": "",
            "conclusion": "",
            "references": [],
            "process": ""
        }
        
        all_answers = []
        for i, question in enumerate(questions, 1):
            print(Fore.YELLOW + f"\n[{i}/{len(questions)}] 研究问题: {question}")
            result = self.ask_question(question)
            
            if result["success"]:
                all_answers.append({
                    "question": question,
                    "answer": result["answer"],
                    "steps": result.get("intermediate_steps", [])
                })
        
        research_data["process"] = self._format_research_process(all_answers)
        
        summary_chain = self.writer.generate_summary_chain()
        combined_content = "\n\n".join([a["answer"] for a in all_answers])
        research_data["summary"] = summary_chain.run(content=combined_content)
        
        report_file = self.writer.generate_report(topic, research_data, "markdown")
        
        print(Fore.GREEN + f"\n研究报告已生成: {report_file}")
        
        # 将报告作为关键发现添加到记忆
        if hasattr(self.assistant, 'add_key_finding'):
            finding = f"完成研究报告《{topic}》，包含{len(questions)}个研究问题"
            self.assistant.add_key_finding(finding)
            print(Fore.CYAN + f"💾 研究进展已保存到记忆")
        
        return report_file
    
    def _format_research_process(self, answers: list) -> str:
        """格式化研究过程"""
        process = "## 研究过程记录\n\n"
        
        for i, answer in enumerate(answers, 1):
            process += f"### 问题 {i}: {answer['question']}\n\n"
            process += f"**回答**:\n{answer['answer']}\n\n"
            
            if answer.get('steps'):
                process += "**推理步骤**:\n"
                for step in answer['steps']:
                    if isinstance(step, tuple) and len(step) >= 2:
                        action, observation = step[0], step[1]
                        process += f"- 行动: {action}\n"
                        process += f"  结果: {observation[:200]}...\n\n"
        
        return process
    
    def show_capabilities(self):
        """显示助手功能"""
        print(Fore.CYAN + "="*60)
        print(Fore.CYAN + "智能研究助手功能")
        print(Fore.CYAN + "="*60)
        print(Fore.YELLOW + "1. 文档处理")
        print(Fore.WHITE + "   - 支持PDF、DOCX、TXT、MD格式")
        print(Fore.WHITE + "   - 自动分块和嵌入")
        print(Fore.YELLOW + "2. 知识库检索")
        print(Fore.WHITE + "   - 基于向量的语义搜索")
        print(Fore.WHITE + "   - 相关文档推荐")
        print(Fore.YELLOW + "3. 研究能力")
        print(Fore.WHITE + "   - 多步骤推理")
        print(Fore.WHITE + "   - 工具使用（搜索、计算等）")
        print(Fore.YELLOW + "4. 记忆功能")
        print(Fore.WHITE + "   - 对话历史记录")
        print(Fore.WHITE + "   - 研究主题管理")
        print(Fore.WHITE + "   - 关键发现提取")
        print(Fore.WHITE + "   - 记忆搜索和导出")
        print(Fore.YELLOW + "5. 报告生成")
        print(Fore.WHITE + "   - 自动生成研究报告")
        print(Fore.WHITE + "   - 支持多种格式输出")
        print(Fore.CYAN + "="*60)

def main():
    """主函数"""
    print(Fore.CYAN + """
    ╔══════════════════════════════════════════════════════════╗
    ║           基于LangChain与DeepSeek的智能研究助手           ║
    ║                 （集成记忆功能版）                         ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # 询问是否启用记忆功能
    use_memory = input(Fore.GREEN + "是否启用记忆功能？(Y/n): " + Style.RESET_ALL).strip().lower()
    use_memory = use_memory != 'n'
    
    # 询问用户ID（用于记忆隔离）
    user_id = "default"
    if use_memory:
        user_input = input(Fore.GREEN + "请输入用户ID（回车使用默认）: " + Style.RESET_ALL).strip()
        if user_input:
            user_id = user_input
    
    assistant = IntelligentResearchAssistant(use_memory=use_memory, user_id=user_id)
    
    while True:
        print(Fore.CYAN + "="*60)
        print(Fore.YELLOW + "1. 交互式聊天（带记忆功能）")
        print(Fore.YELLOW + "2. 加载文档到知识库")
        print(Fore.YELLOW + "3. 生成研究报告")
        print(Fore.YELLOW + "4. 显示功能")
        print(Fore.YELLOW + "5. 记忆管理")
        print(Fore.YELLOW + "6. 退出")
        print(Fore.CYAN + "="*60)
        
        choice = input(Fore.GREEN + "请选择 (1-6): " + Style.RESET_ALL)
        
        if choice == "1":
            assistant.interactive_chat()
        elif choice == "2":
            path = input(Fore.GREEN + "请输入文档路径或目录: " + Style.RESET_ALL)
            assistant.load_documents(path)
        elif choice == "3":
            topic = input(Fore.GREEN + "请输入研究主题: " + Style.RESET_ALL)
            print(Fore.GREEN + "请输入研究问题（每行一个问题，空行结束）:")
            questions = []
            while True:
                q = input(Fore.WHITE + "> " + Style.RESET_ALL)
                if not q.strip():
                    break
                questions.append(q)
            
            if questions:
                assistant.generate_research_report(topic, questions)
            else:
                print(Fore.RED + "未输入任何问题")
        elif choice == "4":
            assistant.show_capabilities()
        elif choice == "5":
            # 记忆管理子菜单
            if hasattr(assistant.assistant, 'get_memory_stats'):
                print(Fore.CYAN + "\n" + "="*60)
                print(Fore.CYAN + "记忆管理")
                print(Fore.CYAN + "="*60)
                print(Fore.YELLOW + "1. 查看记忆统计")
                print(Fore.YELLOW + "2. 导出记忆")
                print(Fore.YELLOW + "3. 搜索记忆")
                print(Fore.YELLOW + "4. 返回主菜单")
                print(Fore.CYAN + "="*60)
                
                mem_choice = input(Fore.GREEN + "请选择 (1-4): " + Style.RESET_ALL)
                
                if mem_choice == "1":
                    stats = assistant.assistant.get_memory_stats()
                    if isinstance(stats, dict):
                        print(Fore.CYAN + "\n📊 记忆统计:")
                        for key, value in stats.items():
                            if key != "research_topics":
                                print(Fore.WHITE + f"  {key}: {value}")
                        
                        topics = stats.get('research_topics', [])
                        if topics:
                            print(Fore.CYAN + "\n  研究主题:")
                            for topic in topics:
                                print(Fore.WHITE + f"    - {topic.get('topic_name', '')}: {topic.get('description', '')[:50]}")
                
                elif mem_choice == "2":
                    filename = input(Fore.GREEN + "导出文件名（回车使用默认）: " + Style.RESET_ALL).strip()
                    if not filename:
                        from datetime import datetime
                        filename = f"memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    success = assistant.assistant.export_memory(filename)
                    if success:
                        print(Fore.GREEN + f"✅ 记忆已导出到: {filename}")
                    else:
                        print(Fore.RED + "❌ 导出失败")
                
                elif mem_choice == "3":
                    keyword = input(Fore.GREEN + "请输入搜索关键词: " + Style.RESET_ALL)
                    results = assistant.assistant.search_memory(keyword, limit=5)
                    if results:
                        print(Fore.CYAN + f"\n🔍 找到 {len(results)} 条相关记忆:")
                        for i, result in enumerate(results, 1):
                            print(Fore.YELLOW + f"{i}. [{result.get('topic', 'general')}]")
                            print(Fore.WHITE + f"   问: {result.get('query', '')}")
                            print(Fore.WHITE + f"   答: {result.get('response', '')[:100]}...")
                            print()
                    else:
                        print(Fore.YELLOW + "未找到相关记忆")
            else:
                print(Fore.RED + "当前助手不支持记忆管理")
        elif choice == "6":
            print(Fore.CYAN + "再见！")
            break
        else:
            print(Fore.RED + "无效选择，请重新输入")

if __name__ == "__main__":
    main()