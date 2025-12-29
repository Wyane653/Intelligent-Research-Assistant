@echo off
echo 安装智能研究助手最小依赖...
echo.

echo 1. 升级pip...
python -m pip install --upgrade pip

echo.
echo 2. 安装核心包（不指定版本以避免冲突）...
pip install langchain
pip install langchain-community
pip install langchain-openai
pip install langchain-text-splitters

echo.
echo 3. 安装AI SDK...
pip install openai

echo.
echo 4. 安装向量数据库...
pip install chromadb

echo.
echo 5. 安装文档处理包...
pip install pypdf2
pip install python-docx
pip install markdown

echo.
echo 6. 安装配置和工具包...
pip install pydantic-settings
pip install python-dotenv
pip install colorama
pip install requests

echo.
echo 安装完成！测试导入...
python -c "import langchain; import langchain_text_splitters; import openai; print('✓ 核心包安装成功')"

echo.
pause