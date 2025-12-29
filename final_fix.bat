@echo off
echo 一键修复所有LangChain导入问题
echo.

echo 1. 安装必要的包...
pip install langchain-community langchain-core langchain-openai

echo.
echo 2. 修复vectorstores导入...
python -c "
import os
file_path = 'src/vector_store.py'
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('from langchain.vectorstores import Chroma', 'from langchain_community.vectorstores import Chroma')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('vector_store.py 修复完成')
else:
    print('vector_store.py 不存在')
"

echo.
echo 3. 修复agent.py中的ChatOpenAI导入...
python -c "
import os
file_path = 'src/agent.py'
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 确保使用正确的导入
    if 'from langchain_community.chat_models import ChatOpenAI' not in content:
        if 'from langchain.chat_models import ChatOpenAI' in content:
            content = content.replace('from langchain.chat_models import ChatOpenAI', 'from langchain_community.chat_models import ChatOpenAI')
            print('替换了旧的ChatOpenAI导入')
        else:
            # 添加导入语句
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'import' in line and 'langchain' in line:
                    lines.insert(i+1, 'from langchain_community.chat_models import ChatOpenAI')
                    break
            content = '\n'.join(lines)
            print('添加了ChatOpenAI导入')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    print('agent.py 修复完成')
else:
    print('agent.py 不存在')
"

echo.
echo 4. 清理Python缓存...
del /q __pycache__ 2>nul
del /q src\__pycache__ 2>nul
del /q config\__pycache__ 2>nul
rmdir /s /q __pycache__ 2>nul

echo.
echo 修复完成！现在运行: python main.py
pause