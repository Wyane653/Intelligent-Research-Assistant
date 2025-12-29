import os
from typing import List, Optional, Dict, Any
try:
    from langchain_community.embeddings import OpenAIEmbeddings
except ImportError:
    from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LangchainDocument
import chromadb
from chromadb.config import Settings as ChromaSettings
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """管理向量数据库"""
    
    def __init__(self):
        self.embeddings = self._create_embeddings()
        self.client_settings = ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=settings.CHROMA_PERSIST_DIR,
            anonymized_telemetry=False
        )
        
    def _create_embeddings(self):
        """创建嵌入模型"""
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.DEEPSEEK_API_KEY,
            openai_api_base=settings.DEEPSEEK_API_BASE
        )
    
    def create_vector_store(self, 
                           documents: List[LangchainDocument],
                           collection_name: str = "research_docs") -> Chroma:
        """创建新的向量存储"""
        if os.path.exists(settings.CHROMA_PERSIST_DIR):
            import shutil
            shutil.rmtree(settings.CHROMA_PERSIST_DIR)
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
            collection_name=collection_name,
            client_settings=self.client_settings
        )
        
        vector_store.persist()
        logger.info(f"向量存储已创建，包含 {len(documents)} 个文档块")
        return vector_store
    
    def load_vector_store(self, collection_name: str = "research_docs") -> Optional[Chroma]:
        """加载现有的向量存储"""
        try:
            vector_store = Chroma(
                persist_directory=settings.CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_name=collection_name,
                client_settings=self.client_settings
            )
            
            collection = vector_store._collection
            if collection.count() == 0:
                logger.warning("向量存储为空")
                return None
            
            logger.info(f"向量存储已加载，包含 {collection.count()} 个文档")
            return vector_store
        except Exception as e:
            logger.error(f"加载向量存储时出错: {str(e)}")
            return None
    
    def search_similar(self, 
                      query: str, 
                      vector_store: Chroma, 
                      k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        results = vector_store.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        
        return formatted_results
    
    def delete_collection(self, collection_name: str = "research_docs"):
        """删除向量存储"""
        try:
            client = chromadb.Client(self.client_settings)
            client.delete_collection(collection_name)
            logger.info(f"集合 '{collection_name}' 已删除")
        except Exception as e:
            logger.error(f"删除集合时出错: {str(e)}")
