import os
import re

def fix_agent_imports():
    """修复agent.py中的导入问题"""
    
    file_path = "src/agent.py"
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    print("修复 agent.py 导入...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换导入语句
    new_imports = '''try:
    # 新版本的导入方式
    from langchain.agents import create_react_agent
    from langchain.agents.agent import AgentExecutor
except ImportError:
    try:
        # 旧版本的导入方式
        from langchain.agents import AgentExecutor, create_react_agent
    except ImportError:
        # 如果都不行，尝试其他可能的路径
        from langchain.agents.agent_executor import AgentExecutor
        from langchain.agents.react.agent import create_react_agent
'''
    
    # 查找并替换原来的导入
    if 'from langchain.agents import AgentExecutor, create_react_agent' in content:
        content = content.replace(
            'from langchain.agents import AgentExecutor, create_react_agent',
            new_imports
        )
        print("✓ 替换了旧的导入语句")
    elif 'from langchain.agents import AgentExecutor' in content:
        # 使用正则表达式替换
        pattern = r'from langchain\.agents import (.*?)(AgentExecutor.*?)create_react_agent'
        content = re.sub(pattern, new_imports, content)
        print("✓ 使用正则表达式替换了导入语句")
    else:
        # 在文件开头添加
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('from langchain') or line.startswith('import langchain'):
                lines.insert(i, new_imports)
                break
        content = '\n'.join(lines)
        print("✓ 添加了新的导入语句")
    
    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ agent.py 修复完成")
    return True

def check_agent_versions():
    """检查agent相关包的版本"""
    print("\n检查相关包版本...")
    
    import pkg_resources
    
    packages = [
        'langchain',
        'langchain-community',
        'langchain-core',
    ]
    
    for pkg in packages:
        try:
            version = pkg_resources.get_distribution(pkg).version
            print(f"✓ {pkg}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"✗ {pkg}: 未安装")

def test_import():
    """测试导入"""
    print("\n测试导入...")
    
    test_code = '''
try:
    from langchain.agents import create_react_agent
    print("✓ create_react_agent 导入成功")
except ImportError as e:
    print(f"✗ create_react_agent 导入失败: {e}")

try:
    from langchain.agents.agent import AgentExecutor
    print("✓ AgentExecutor (from agent) 导入成功")
except ImportError as e:
    print(f"✗ AgentExecutor (from agent) 导入失败: {e}")

try:
    from langchain.agents.agent_executor import AgentExecutor
    print("✓ AgentExecutor (from agent_executor) 导入成功")
except ImportError as e:
    print(f"✗ AgentExecutor (from agent_executor) 导入失败: {e}")

try:
    from langchain.agents import AgentExecutor
    print("✓ AgentExecutor (from agents) 导入成功")
except ImportError as e:
    print(f"✗ AgentExecutor (from agents) 导入失败: {e}")
'''
    
    exec(test_code)

if __name__ == "__main__":
    print("修复AgentExecutor导入问题")
    print("=" * 60)
    
    fix_agent_imports()
    check_agent_versions()
    test_import()
    
    print("\n" + "=" * 60)
    print("修复完成！现在可以尝试运行: python main.py")