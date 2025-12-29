import os

# 更新 agent.py
agent_content = '''from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

# 动态导入以兼容不同版本
try:
    # 新版导入方式
    from langchain.agents import create_react_agent
    from langchain.agents.agent import AgentExecutor
except ImportError:
    try:
        # 旧版导入方式
        from langchain.agents import AgentExecutor, create_react_agent
    except ImportError:
        try:
            # 备选导入方式
            from langchain.agents.agent_executor import AgentExecutor
            from langchain.agents.react.agent import create_react_agent
        except ImportError as e:
            logger.error(f"无法导入AgentExecutor或create_react_agent: {e}")
            raise

try:
    from langchain_community.chat_models import ChatOpenAI
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        raise ImportError("需要安装langchain-community或旧版langchain")

class ResearchAssistant:
    """研究助手代理"""
    
    def __init__(self, tools: List[BaseTool], model_name: str = None):
        self.tools = tools
        self.model_name = model_name or settings.DEEPSEEK_MODEL
        
        self.llm = self._create_llm()
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        self.prompt = self._create_prompt()
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
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
'''

# 写入文件
with open('src/agent.py', 'w', encoding='utf-8') as f:
    f.write(agent_content)

print("✓ agent.py 已更新为兼容版本")

# 更新 main.py 中的导入
main_file = 'src/main.py'
if os.path.exists(main_file):
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 确保导入正确的模块
    if 'from src.agent_fixed import ResearchAssistant' in content:
        content = content.replace(
            'from src.agent_fixed import ResearchAssistant',
            'from src.agent import ResearchAssistant'
        )
        print("✓ main.py 中的导入已更新")
    
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)

print("\n更新完成！现在可以运行: python main.py")