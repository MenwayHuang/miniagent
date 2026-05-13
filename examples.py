"""
MiniAgent 使用示例
"""
import asyncio
from mini_agent import MiniAgent
from mini_agent.llm import create_llm_from_env


def create_example_agent(name: str) -> MiniAgent:
    """创建示例Agent，默认连接本地Ollama"""
    llm = create_llm_from_env()
    print(f"使用模型: {llm.provider}/{llm.model}")
    return MiniAgent(llm=llm, name=name)


async def example_file_operations():
    """文件操作示例"""
    print("=== 文件操作示例 ===")
    
    agent = create_example_agent("FileAgent")
    
    # 任务: 创建文件并写入内容
    task = "在当前目录创建一个名为demo.txt的文件，内容是'这是一个测试文件，由MiniAgent创建'"
    result = await agent.run(task)
    print(f"结果: {result}")


async def example_python_execution():
    """Python代码执行示例"""
    print("\n=== Python代码执行示例 ===")
    
    agent = create_example_agent("PythonAgent")
    
    # 任务: 数学计算
    task = "用Python计算斐波那契数列的前10项，并将结果保存到fib.txt文件中"
    result = await agent.run(task)
    print(f"结果: {result}")


async def example_data_processing():
    """数据处理示例"""
    print("\n=== 数据处理示例 ===")
    
    agent = create_example_agent("DataAgent")
    
    # 任务: 清洗文本文件
    task = """
    请执行以下任务:
    1. 创建一个包含一些杂乱文本的文件data.txt
    2. 写一个Python程序清洗这个文件（去除空行，去除首尾空格）
    3. 将清洗后的结果保存到cleaned_data.txt
    """
    result = await agent.run(task)
    print(f"结果: {result}")


async def run_all_examples():
    """运行所有示例"""
    print("🚀 MiniAgent 示例程序")
    print("默认使用本地Ollama，可用 OLLAMA_MODEL 切换模型")
    
    try:
        await example_file_operations()
        await example_python_execution()
        await example_data_processing()
        print("\n✅ 所有示例执行完成!")
    except Exception as e:
        print(f"❌ 示例执行出错: {e}")


if __name__ == "__main__":
    asyncio.run(run_all_examples())
