import sys

print("测试 tools 模块的导入...")

# 测试从 langchain.tools.base 导入 Tool
try:
    from langchain.tools.base import Tool
    print("✓ 从 langchain.tools.base 导入 Tool 成功")
except ImportError as e:
    print(f"✗ 从 langchain.tools.base 导入 Tool 失败: {e}")

# 测试从 langchain_core.tools 导入 BaseTool
try:
    from langchain_core.tools import BaseTool
    print("✓ 从 langchain_core.tools 导入 BaseTool 成功")
except ImportError as e:
    print(f"✗ 从 langchain_core.tools 导入 BaseTool 失败: {e}")

# 测试从 langchain_community.utilities 导入 SerpAPIWrapper
try:
    from langchain_community.utilities import SerpAPIWrapper
    print("✓ 从 langchain_community.utilities 导入 SerpAPIWrapper 成功")
except ImportError as e:
    print(f"✗ 从 langchain_community.utilities 导入 SerpAPIWrapper 失败: {e}")

# 测试从 langchain_core.callbacks 导入 CallbackManagerForToolRun
try:
    from langchain_core.callbacks import CallbackManagerForToolRun
    print("✓ 从 langchain_core.callbacks 导入 CallbackManagerForToolRun 成功")
except ImportError as e:
    print(f"✗ 从 langchain_core.callbacks 导入 CallbackManagerForToolRun 失败: {e}")

print("\n测试完成。")