import os

def update_main_import():
    """更新main.py中的导入语句"""
    
    file_path = "src/main.py"
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    print(f"正在修改 {file_path} 中的导入语句...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找原来的导入语句
    old_import = "from src.agent import ResearchAssistant"
    
    # 新的导入语句
    new_import = '''try:
    from src.agent import ResearchAssistant
except ImportError:
    try:
        from src.agent_final import ResearchAssistant
    except ImportError:
        print("无法导入ResearchAssistant，请检查依赖包安装")
        exit(1)'''
    
    if old_import in content:
        # 替换导入语句
        content = content.replace(old_import, new_import)
        print("✓ 已替换导入语句")
    else:
        # 检查是否已经有try-except导入
        if "try:" in content and "from src.agent import" in content:
            print("✓ main.py 中已有try-except导入，无需修改")
            return True
        
        # 如果没有找到旧的导入，可能在不同的位置，尝试找到所有导入agent的地方
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "ResearchAssistant" in line and ("import" in line or "from" in line):
                print(f"在第 {i+1} 行找到导入语句: {line}")
                lines[i] = new_import
                content = '\n'.join(lines)
                print("✓ 已更新导入语句")
                break
    
    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ main.py 修改完成")
    return True

def check_agent_files():
    """检查agent相关文件是否存在"""
    print("\n检查agent相关文件...")
    
    files_to_check = [
        ("src/agent.py", "主agent文件"),
        ("src/agent_final.py", "兼容性agent文件"),
    ]
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} 存在 ({description})")
        else:
            print(f"✗ {file_path} 不存在 ({description})")
            if file_path == "src/agent_final.py":
                print("  请先创建 agent_final.py 文件")

if __name__ == "__main__":
    print("更新main.py导入语句")
    print("=" * 60)
    
    check_agent_files()
    update_main_import()
    
    print("\n" + "=" * 60)
    print("修改完成！现在可以尝试运行: python main.py")