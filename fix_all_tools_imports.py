import os
import re

def fix_imports_in_file(file_path):
    """修复单个文件中的导入语句"""
    
    if not os.path.exists(file_path):
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义替换规则
    replacements = [
        # (旧导入, 新导入)
        ("from langchain.tools import Tool", "from langchain.tools.base import Tool"),
        ("from langchain.tools import BaseTool", "from langchain_core.tools import BaseTool"),
        ("from langchain.tools.base import BaseTool", "from langchain_core.tools import BaseTool"),
        ("from langchain.utilities import SerpAPIWrapper", "from langchain_community.utilities import SerpAPIWrapper"),
        ("from langchain.callbacks.manager import CallbackManagerForToolRun", "from langchain_core.callbacks import CallbackManagerForToolRun"),
    ]
    
    modified = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"在 {file_path} 中将 '{old}' 替换为 '{new}'")
            modified = True
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

def main():
    # 修复 tools.py
    fix_imports_in_file("src/tools.py")
    
    # 修复 agent.py 中可能存在的类似导入
    fix_imports_in_file("src/agent.py")
    
    print("\n导入修复完成！")

if __name__ == "__main__":
    main()