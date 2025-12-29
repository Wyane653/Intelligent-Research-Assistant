import os
import re

def update_all_imports():
    """更新所有文件中的旧导入语句"""
    
    print("批量更新所有文件中的导入语句...")
    print("=" * 60)
    
    # 需要更新的模式
    patterns = [
        # 格式: (正则表达式模式, 替换字符串)
        (r'from langchain\.vectorstores import (\w+)', r'from langchain_community.vectorstores import \1'),
        (r'from langchain\.embeddings import (\w+)', r'from langchain_community.embeddings import \1'),
        (r'from langchain\.llms import (\w+)', r'from langchain_community.llms import \1'),
        (r'from langchain\.chat_models import (\w+)', r'from langchain_community.chat_models import \1'),
    ]
    
    updated_files = []
    
    # 遍历src目录
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = content
                changes_made = False
                
                for pattern, replacement in patterns:
                    # 使用正则表达式进行替换
                    new_content, count = re.subn(pattern, replacement, new_content)
                    if count > 0:
                        changes_made = True
                        print(f"在 {file_path} 中替换了 {count} 处")
                
                if changes_made:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    updated_files.append(file_path)
    
    print("\n" + "=" * 60)
    print(f"更新完成！共更新了 {len(updated_files)} 个文件")
    
    if updated_files:
        print("更新的文件:")
        for file in updated_files:
            print(f"  - {file}")

if __name__ == "__main__":
    update_all_imports()