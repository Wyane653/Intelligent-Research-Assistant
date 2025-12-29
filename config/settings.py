import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # DeepSeek API配置
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # 向量数据库配置
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    CHROMA_PERSIST_DIR: str = "./data/embeddings"
    
    # 文件处理配置
    SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".txt", ".md", ".docx", ".pptx"]
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # 代理配置
    MAX_ITERATIONS: int = 4
    TEMPERATURE: float = 0.3
    
    # 输出配置
    OUTPUT_DIR: str = "./data/outputs"

    # 研究助手配置
    # MAX_ITERATIONS = 4  # 最大迭代次数
    DEBUG_MODE:bool = True  # 调试模式
    MAX_ANSWER_LENGTH: int = 600   # 最大答案长度
    # TEMPERATURE = 0.3  # LLM温度
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = Settings()