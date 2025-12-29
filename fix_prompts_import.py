import os

def fix_prompts_imports():
    """修复prompts导入问题"""
    
    print("修复prompts导入问题...")
    print("=" * 60)
    
    # 需要修复的文件
    files_to_fix = [
        "src/agent.py",
        "src/research_writer.py",
    ]
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换导入语句
        old_import = "from langchain.prompts import PromptTemplate"
        new_import = "from langchain_core.prompts import PromptTemplate"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            print(f"✓ 在 {file_path} 中修复了PromptTemplate导入")
        else:
            # 检查是否已经是正确的导入
            if "from langchain_core.prompts import PromptTemplate" in content:
                print(f"✓ {file_path} 已经有正确的导入")
            elif "PromptTemplate" in content and "import" in content:
                # 检查是否有其他形式的导入
                print(f"⚠ {file_path} 中有PromptTemplate但导入方式可能不同")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print("\n" + "=" * 60)
    print("修复完成！")

def test_imports():
    """测试导入"""
    print("\n测试导入...")
    
    try:
        from langchain_core.prompts import PromptTemplate
        print("✓ langchain_core.prompts.PromptTemplate 导入成功")
    except ImportError as e:
        print(f"✗ langchain_core.prompts.PromptTemplate 导入失败: {e}")
        
        try:
            from langchain.prompts import PromptTemplate
            print("✓ langchain.prompts.PromptTemplate 导入成功（旧版）")
        except ImportError as e2:
            print(f"✗ langchain.prompts.PromptTemplate 也导入失败: {e2}")

if __name__ == "__main__":
    fix_prompts_imports()
    test_imports()
    
    print("\n现在可以尝试运行: python main.py")