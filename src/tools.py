from typing import ClassVar, Optional, List, Dict, Any, Callable
from langchain_core.tools import BaseTool, Tool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.utilities import SerpAPIWrapper
import requests
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import json
import logging

logger = logging.getLogger(__name__)

class SearchTool(BaseTool):
    """
    一个修复了Pydantic v2继承问题的搜索工具。
    通过 model_config 和 __init_subclass__ 来安全地设置默认的 name 和 description。
    """
    # 1. 声明自定义字段，并提供安全的默认值
    has_api: bool = False
    search_wrapper: Any = None
    
    # 2. 使用 Pydantic 的配置来禁用可能引起问题的“字段覆盖”行为（可选，但更安全）
    #    并明确指定从父类继承哪些字段。
    model_config = {
        'ignored_types': (BaseTool,), # 更清晰地处理父类
    }

    # 3. 类级别的元数据，但不作为“字段”直接覆盖父类
    #    我们稍后会在 __init__ 中通过 kwargs 传递它们
    _tool_name: ClassVar[str] = "search"
    _tool_description: ClassVar[str] = """
    用于搜索最新信息。当需要最新数据、新闻或特定事实时使用此工具。
    输入应为搜索查询字符串。
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        # 准备自定义字段的值
        has_api_val = api_key is not None
        search_wrapper_val = None
        
        if has_api_val:
            try:
                from langchain_community.utilities import SerpAPIWrapper
                search_wrapper_val = SerpAPIWrapper(serpapi_api_key=api_key)
            except ImportError:
                logger.warning("未找到 SerpAPIWrapper，搜索功能将受限。")
                has_api_val = False

        # 关键：在调用父类初始化时，显式传入 name 和 description。
        #       这里使用我们类中定义的 _tool_name 和 _tool_description。
        #       同时传入我们计算好的自定义字段。
        super().__init__(
            **{
                **kwargs,  # 先展开用户可能传入的其他参数
                'name': self._tool_name,          # 显式设置 name
                'description': self._tool_description, # 显式设置 description
                'has_api': has_api_val,
                'search_wrapper': search_wrapper_val,
            }
        )

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            if self.has_api and self.search_wrapper:
                return self.search_wrapper.run(query)
            else:
                return f"搜索查询: {query}\n\n注意：搜索功能需要配置API密钥。请使用本地知识库或提供具体问题。"
        except Exception as e:
            logger.error(f"搜索时出错: {str(e)}")
            return f"搜索失败: {str(e)}"

class CalculatorTool(BaseTool):
    """计算器工具"""
    name: str = "calculator"
    description: str = """
    用于执行数学计算。当需要计算数字、百分比、统计量等时使用此工具。
    输入应为数学表达式字符串。
    """
    
    def _run(self, expression: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "错误：表达式中包含非法字符"
            
            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

class CurrentDateTimeTool(BaseTool):
    """获取当前日期时间工具"""
    name: str = "get_current_datetime"
    description: str = """
    获取当前的日期和时间。当需要知道当前时间或计算时间相关问题时使用此工具。
    输入可以是任何字符串，通常为空。
    """
    
    def _run(self, query: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        now = datetime.now()
        return f"当前日期和时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"

class KnowledgeBaseQueryTool(BaseTool):
    """知识库查询工具"""
    
    # 定义字段
    vector_store_manager: Any = Field(default=None, description="向量存储管理器")
    vector_store: Any = Field(default=None, description="加载的向量存储")
    
    # 工具名称和描述
    name: str = "query_knowledge_base"
    description: str = """
    从本地知识库中检索相关信息。当需要参考已上传的文档时使用此工具。
    输入应为要查询的问题或关键词。
    """
    
    # 不再需要自定义 __init__ 方法，因为 Pydantic 会自动处理字段初始化
    # 如果需要，可以完全移除 __init__ 方法
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """执行查询，返回最相关的信息"""
        try:
            if not self.vector_store_manager:
                return "向量存储管理器未初始化"
        
            # 如果 vector_store 为 None，则加载
            if self.vector_store is None:
                self.vector_store = self.vector_store_manager.load_vector_store()
        
            if self.vector_store is None:
                return "知识库未加载或为空。请先上传文档。"
        
            # 根据查询复杂度调整返回结果数量
            # 简单查询：返回1-2个结果；复杂查询：返回3-4个结果
            if len(query.split()) <= 3:  # 短查询
                k = 2
            elif "详细" in query or "所有" in query or "全面" in query:
                k = 4
            else:
                k = 3
        
            results = self.vector_store_manager.search_similar(query, self.vector_store, k=k)
        
            if not results:
                return "未找到相关信息。"
        
            # 过滤低质量结果（相关性分数太低）
            filtered_results = [r for r in results if r.get('score', 0) > 0.5]
            if not filtered_results:
                # 如果没有高质量结果，返回最相关的一个
                filtered_results = [results[0]]
        
            # 去重：根据内容相似性去重
            unique_results = []
            seen_contents = set()
            for result in filtered_results:
                content_preview = result['content'][:50]  # 取前50个字符作为标识
                if content_preview not in seen_contents:
                    seen_contents.add(content_preview)
                    unique_results.append(result)
        
            # 构建简洁的响应
            if len(unique_results) == 1:
                # 只有一个结果时，直接返回最相关的信息
                result = unique_results[0]
                response = f"根据知识库信息：{result['content'][:300]}"
                if len(result['content']) > 300:
                    response += "..."
            else:
                # 多个结果时，总结关键信息
                response = f"根据知识库中的{len(unique_results)}条相关信息：\n"
            
                # 提取每个结果的核心信息
                for i, result in enumerate(unique_results, 1):
                    # 智能截断：在句子边界处截断
                    content = result['content']
                    if len(content) > 150:
                        # 在150字符附近找句子结束点
                        trunc_point = content[:200].rfind('。')
                        if trunc_point > 80:  # 至少保留80个字符
                            content = content[:trunc_point + 1] + "..."
                        else:
                            content = content[:150] + "..."
                
                    source_info = result['metadata'].get('source', '未知')
                    page_info = f"，第{result['metadata'].get('page')}页" if 'page' in result['metadata'] else ""
                
                    response += f"\n{i}. {content}"
                    response += f"\n   来源：{source_info}{page_info}"
        
            return response
        
        except Exception as e:
            logger.error(f"查询知识库时出错: {str(e)}")
            return f"查询知识库时出错: {str(e)}"

def create_tools(vector_store_manager=None, search_api_key=None):
    """创建所有工具"""
    tools = []
    
    tools.append(SearchTool(api_key=search_api_key))
    tools.append(CalculatorTool())
    tools.append(CurrentDateTimeTool())
    
    if vector_store_manager:
        tools.append(KnowledgeBaseQueryTool(
            vector_store_manager=vector_store_manager,
            name="query_knowledge_base",
            description="从已上传的文档中检索相关信息"
        ))
    
    return tools
