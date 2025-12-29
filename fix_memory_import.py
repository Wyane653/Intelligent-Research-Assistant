import os

def fix_memory_imports():
    """修复memory导入问题"""
    
    print("修复memory导入问题...")
    print("=" * 60)
    
    # 需要修复的文件
    files_to_fix = [
        "src/agent.py",
    ]
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换导入语句
        old_import = "from langchain.memory import ConversationBufferMemory"
        new_import = "from langchain.memory import ConversationBufferMemory"
        
        # 尝试从langchain导入，如果失败则尝试从其他位置导入
        compatibility_import = '''try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    try:
        from langchain_community.memory import ConversationBufferMemory
    except ImportError:
        from langchain_core.memory import ConversationBufferMemory'''
        
        if old_import in content:
            content = content.replace(old_import, compatibility_import)
            print(f"✓ 在 {file_path} 中修复了ConversationBufferMemory导入")
        else:
            # 检查是否已经是正确的导入
            if "ConversationBufferMemory" in content and ("import" in content or "from" in content):
                print(f"⚠ {file_path} 中已有ConversationBufferMemory导入，但可能不是兼容性导入")
            else:
                print(f"✗ {file_path} 中没有找到ConversationBufferMemory导入语句")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print("\n" + "=" * 60)
    print("修复完成！")

def test_imports():
    """测试导入"""
    print("\n测试导入...")
    
    # 测试ConversationBufferMemory
    import_tests = [
        ("langchain.memory.ConversationBufferMemory", "from langchain.memory import ConversationBufferMemory"),
        ("langchain_community.memory.ConversationBufferMemory", "from langchain_community.memory import ConversationBufferMemory"),
        ("langchain_core.memory.ConversationBufferMemory", "from langchain_core.memory import ConversationBufferMemory"),
    ]
    
    for module_path, import_stmt in import_tests:
        try:
            exec(import_stmt)
            print(f"✓ {import_stmt}")
            # 如果成功，测试实际功能
            try:
                memory = ConversationBufferMemory(memory_key="test", return_messages=True)
                print(f"  ✓ 可以创建ConversationBufferMemory实例")
                break  # 找到一个可用的就停止
            except Exception as e:
                print(f"  ✗ 创建实例失败: {e}")
        except ImportError as e:
            print(f"✗ {import_stmt} - 错误: {e}")

def check_packages():
    """检查相关包"""
    print("\n检查相关包...")
    
    packages = [
        "langchain",
        "langchain-community",
        "langchain-core",
    ]
    
    import pkg_resources
    for pkg in packages:
        try:
            version = pkg_resources.get_distribution(pkg).version
            print(f"✓ {pkg}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"✗ {pkg}: 未安装")

if __name__ == "__main__":
    fix_memory_imports()
    check_packages()
    test_imports()
    
    print("\n现在可以尝试运行: python main.py")