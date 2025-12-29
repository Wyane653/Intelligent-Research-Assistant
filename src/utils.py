import logging
import sys
from pathlib import Path
from typing import Optional
import json

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """设置日志配置"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )

def save_conversation(conversation: list, filepath: str):
    """保存对话记录"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)

def load_conversation(filepath: str) -> list:
    """加载对话记录"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def format_document_for_display(document: dict, max_length: int = 300) -> str:
    """格式化文档用于显示"""
    content = document.get('content', '')
    if len(content) > max_length:
        content = content[:max_length] + '...'
    
    metadata = document.get('metadata', {})
    source = metadata.get('source', '未知来源')
    page = metadata.get('page', '')
    
    result = f"来源: {source}"
    if page:
        result += f" (页 {page})"
    result += f"\n内容: {content}\n"
    
    return result
