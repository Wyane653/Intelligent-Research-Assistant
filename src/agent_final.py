from typing import List, Dict, Any, Optional
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

# ==================== 兼容性导入开始 ====================

# 导入 PromptTemplate
try:
    from langchain_core.prompts import PromptTemplate
    print("使用 langchain_core.prompts.PromptTemplate")
except ImportError:
    try:
        from langchain.prompts import PromptTemplate
        print("使用 langchain.prompts.PromptTemplate")
    except ImportError:
        raise ImportError("需要安装 langchain-core 或 langchain")

# 导入 ConversationBufferMemory
try:
    from langchain.memory import ConversationBufferMemory
    print("使用 langchain.memory.ConversationBufferMemory")
except ImportError:
    try:
        from langchain_community.memory import ConversationBufferMemory
        print("使用 langchain_community.memory.ConversationBufferMemory")
    except ImportError:
        try:
            from langchain_core.memory import ConversationBufferMemory
            print("使用 langchain_core.memory.ConversationBufferMemory")
        except ImportError:
            # 新增：从 langchain-classic 兼容包中导入
            try:
                from langchain_classic.memory import ConversationBufferMemory
                print("使用 langchain_classic.memory.ConversationBufferMemory (兼容模式)")
            except ImportError:
                # 如果以上全部失败，再报错
                raise ImportError("无法导入 ConversationBufferMemory。请确保已安装 'langchain-classic' 包。")

# 导入 StreamingStdOutCallbackHandler
try:
    from langchain.callbacks import StreamingStdOutCallbackHandler
    print("使用 langchain.callbacks.StreamingStdOutCallbackHandler")
except ImportError:
    try:
        from langchain_community.callbacks import StreamingStdOutCallbackHandler
        print("使用 langchain_community.callbacks.StreamingStdOutCallbackHandler")
    except ImportError:
        try:
            from langchain_core.callbacks import StreamingStdOutCallbackHandler
            print("使用 langchain_core.callbacks.StreamingStdOutCallbackHandler")
        except ImportError:
            # 如果没有回调处理器，创建一个虚拟的
            class StreamingStdOutCallbackHandler:
                pass
            print("使用虚拟的 StreamingStdOutCallbackHandler")

# 导入 BaseTool
try:
    from langchain.tools import BaseTool
    print("使用 langchain.tools.BaseTool")
except ImportError:
    try:
        from langchain_community.tools import BaseTool
        print("使用 langchain_community.tools.BaseTool")
    except ImportError:
        raise ImportError("无法导入 BaseTool")

# 导入 AgentExecutor 和 create_react_agent
try:
    from langchain.agents import create_react_agent
    from langchain.agents.agent import AgentExecutor
    print("使用 langchain.agents 新版导入")
except ImportError:
    try:
        from langchain.agents import AgentExecutor, create_react_agent
        print("使用 langchain.agents 旧版导入")
    except ImportError:
        try:
            from langchain.agents.agent_executor import AgentExecutor
            from langchain.agents.react.agent import create_react_agent
            print("使用 langchain.agents 备选导入")
        except ImportError:
            # 新增：从 langchain-classic 兼容包中导入
            try:
                from langchain_classic.agents import AgentExecutor, create_react_agent
                print("使用 langchain_classic.agents (兼容模式)")
            except ImportError as e:
                logger.error(f"无法导入AgentExecutor或create_react_agent，请确保已安装 'langchain-classic' 包。错误: {e}")
                raise

# 导入 ChatOpenAI
try:
    from langchain_community.chat_models import ChatOpenAI
    print("使用 langchain_community.chat_models.ChatOpenAI")
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
        print("使用 langchain.chat_models.ChatOpenAI")
    except ImportError:
        raise ImportError("需要安装 langchain-community 或旧版 langchain")

# ==================== 兼容性导入结束 ====================

class ResearchAssistant:
    """研究助手代理"""
    
    def __init__(self, tools: List[BaseTool], model_name: str = None):
        self.tools = tools
        self.model_name = model_name or settings.DEEPSEEK_MODEL
        
        # 初始化LLM
        self.llm = self._create_llm()
        
        # 创建记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # 创建提示模板
        self.prompt = self._create_prompt()
        
        # 创建代理
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # 创建代理执行器
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=settings.MAX_ITERATIONS,
            return_intermediate_steps=True
        )
    
    def _create_llm(self):
        """创建DeepSeek LLM实例"""
        return ChatOpenAI(
            model=self.model_name,
            temperature=settings.TEMPERATURE,
            openai_api_key=settings.DEEPSEEK_API_KEY,
            openai_api_base=settings.DEEPSEEK_API_BASE,
            streaming=False,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    def _create_prompt(self) -> PromptTemplate:
        """创建代理提示模板"""
        template = """你是一个专业的研究助手，拥有广泛的知识和强大的研究能力。
        
        你可以使用以下工具：
        {tools}
        
        使用以下格式：
        问题：用户提出的问题
        思考：你需要思考如何解决问题
        行动：要采取的行动，应该是[{tool_names}]中的一个
        行动输入：行动的输入
        观察：行动的结果
        ...（这个思考/行动/行动输入/观察可以重复多次）
        思考：我现在知道了最终答案
        最终答案：对原始问题的最终回答
        
        请用中文回答，并且回答应该详细、准确、有引用来源。
        
        如果你需要引用来源，请明确指出信息来源。
        
        之前的对话：
        {chat_history}
        
        开始！
        
        问题：{input}
        思考：{agent_scratchpad}"""
        
        return PromptTemplate.from_template(template)
    
    def query(self, question: str) -> Dict[str, Any]:
        """向助手提问"""
        try:
            result = self.agent_executor.invoke({
                "input": question
            })
            
            return {
                "answer": result.get("output", "没有获取到回答"),
                "intermediate_steps": result.get("intermediate_steps", []),
                "success": True
            }
        except Exception as e:
            logger.error(f"执行查询时出错: {str(e)}")
            return {
                "answer": f"抱歉，处理请求时出现错误: {str(e)}",
                "success": False
            }
    
    def reset_memory(self):
        """重置对话记忆"""
        self.memory.clear()
        logger.info("对话记忆已重置")
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        memory_variables = self.memory.load_memory_variables({})
        chat_history = memory_variables.get("chat_history", [])
        
        formatted_history = []
        for i in range(0, len(chat_history), 2):
            if i + 1 < len(chat_history):
                formatted_history.append({
                    "human": chat_history[i].content,
                    "ai": chat_history[i + 1].content
                })
        
        return formatted_history