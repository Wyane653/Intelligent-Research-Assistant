import os
import re

def fix_file_imports(file_path):
    """修复单个文件中的导入语句"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 记录原始内容长度
    original_length = len(content)
    
    # 替换导入语句
    replacements = {
        # 旧的导入方式
        "from langchain.schema import Document": "from langchain_core.documents import Document",
        "from langchain.schema import Document as LangchainDocument": "from langchain_core.documents import Document as LangchainDocument",
        "from langchain.schema import BaseTool": "from langchain_core.tools import BaseTool",
        # 添加其他可能的导入替换
    }
    
    for old_import, new_import in replacements.items():
        if old_import in content:
            content = content.replace(old_import, new_import)
            print(f"在 {file_path} 中将 '{old_import}' 替换为 '{new_import}'")
    
    # 另外，检查是否有其他需要替换的模式
    # 使用正则表达式匹配更灵活的导入模式
    patterns = [
        (r"from langchain\.schema import (\w+)", r"from langchain_core.\1 import \1"),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if len(content) != original_length:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    else:
        print(f"{file_path} 中未找到需要替换的导入语句")
        return False

def main():
    """主修复函数"""
    files_to_fix = [
        "src/document_processor.py",
        "src/vector_store.py",
        "src/tools.py",
        "src/agent.py",
    ]
    
    print("开始修复LangChain导入问题...")
    print("=" * 60)
    
    fixed_files = []
    for file_path in files_to_fix:
        print(f"\n处理: {file_path}")
        if fix_file_imports(file_path):
            fixed_files.append(file_path)
    
    print("\n" + "=" * 60)
    print(f"修复完成！共修复了 {len(fixed_files)} 个文件")
    
    if fixed_files:
        print("已修复的文件:")
        for file in fixed_files:
            print(f"  - {file}")
    
    print("\n现在可以尝试运行: python main.py")

if __name__ == "__main__":
    main()