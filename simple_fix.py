import os

print("修复vectorstores导入问题...")

# 修复 vector_store.py
file_path = "src/vector_store.py"

if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换导入语句
    if "from langchain.vectorstores import Chroma" in content:
        content = content.replace(
            "from langchain.vectorstores import Chroma", 
            "from langchain_community.vectorstores import Chroma"
        )
        print("✓ 已修复 vector_store.py")
    elif "from langchain_community.vectorstores import Chroma" in content:
        print("✓ vector_store.py 已经是正确的导入")
    else:
        print("⚠ vector_store.py 中没有找到 vectorstores 导入语句")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
else:
    print("✗ vector_store.py 文件不存在")

print("\n修复完成！")