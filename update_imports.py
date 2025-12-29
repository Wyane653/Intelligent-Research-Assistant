import os

def update_imports():
    """更新导入语句以适应新版本"""
    
    # 更新document_processor.py
    doc_processor_file = "src/document_processor.py"
    
    if os.path.exists(doc_processor_file):
        with open(doc_processor_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换导入语句
        new_content = content.replace(
            "from langchain.text_splitter import RecursiveCharacterTextSplitter",
            "try:\n    from langchain.text_splitter import RecursiveCharacterTextSplitter\nexcept ImportError:\n    from langchain_text_splitters import RecursiveCharacterTextSplitter"
        )
        
        with open(doc_processor_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✓ 已更新 {doc_processor_file}")
    
    # 检查其他可能的导入问题
    files_to_check = ["src/vector_store.py", "src/agent.py", "src/tools.py"]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否有旧的导入方式
            if "from langchain.embeddings import OpenAIEmbeddings" in content:
                new_content = content.replace(
                    "from langchain.embeddings import OpenAIEmbeddings",
                    "try:\n    from langchain.embeddings import OpenAIEmbeddings\nexcept ImportError:\n    from langchain_openai import OpenAIEmbeddings"
                )
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"✓ 已更新 {file_path} 中的embeddings导入")
    
    print("\n所有导入已更新完成！")

if __name__ == "__main__":
    update_imports()