import os

def fix_vectorstores_import():
    """修复vectorstores导入问题"""
    
    # 需要修复的文件
    files_to_fix = [
        "src/vector_store.py",
        # 可能还有其他文件使用了vectorstores
    ]
    
    print("修复vectorstores导入...")
    print("=" * 60)
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换导入语句
        replacements = {
            "from langchain.vectorstores import Chroma": "from langchain_community.vectorstores import Chroma",
            "from langchain.vectorstores.chroma import Chroma": "from langchain_community.vectorstores.chroma import Chroma",
            "from langchain_community.vectorstores import Chroma": "from langchain_community.vectorstores import Chroma",  # 已经是正确的
            "import langchain.vectorstores": "import langchain_community.vectorstores",
        }
        
        modified = False
        for old, new in replacements.items():
            if old in content:
                content = content.replace(old, new)
                print(f"✓ 在 {file_path} 中将 '{old}' 替换为 '{new}'")
                modified = True
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            # 检查是否有vectorstores相关导入
            if "vectorstores" in content and "langchain_community" not in content:
                print(f"⚠ {file_path} 中有vectorstores相关导入但未修复")
            else:
                print(f"✓ {file_path} 导入正常")
    
    print("\n" + "=" * 60)
    print("修复完成！")

def check_and_install_missing():
    """检查并安装缺少的包"""
    print("\n检查依赖包...")
    
    try:
        import langchain_community
        print("✓ langchain-community 已安装")
    except ImportError:
        print("✗ langchain-community 未安装，正在安装...")
        import subprocess
        subprocess.check_call(["pip", "install", "langchain-community"])
        print("✓ langchain-community 安装完成")
    
    try:
        from langchain_community.vectorstores import Chroma
        print("✓ 可以从 langchain_community.vectorstores 导入 Chroma")
    except ImportError as e:
        print(f"✗ 导入 Chroma 失败: {e}")

if __name__ == "__main__":
    fix_vectorstores_import()
    check_and_install_missing()
    
    print("\n现在可以尝试运行: python main.py")