"""
MiniAgent 主运行文件
"""
import asyncio
from mini_agent import MiniAgent
from mini_agent.llm import create_llm_from_env


async def main():
    """主函数"""
    try:
        llm = create_llm_from_env()
    except ValueError as e:
        print(f"❌ LLM配置错误: {e}")
        return
    
    # 创建代理
    agent = MiniAgent(
        llm=llm,
        name="MiniAgent",
        max_steps=10
    )
    
    print("🤖 MiniAgent 已启动!")
    print(f"🧠 LLM提供方: {llm.provider}")
    print(f"📦 当前模型: {llm.model}")
    print(f"🔗 接口地址: {llm.base_url}")
    print("💡 该Agent支持以下功能:")
    print("   - Python代码执行")
    print("   - 文件读写操作")
    print("   - 命令行执行")
    print("\n本地Ollama默认配置:")
    print("   MINI_AGENT_PROVIDER=ollama")
    print("   OLLAMA_MODEL=qwen2.5:7b")
    print("   OLLAMA_BASE_URL=http://127.0.0.1:11434/v1")
    print("\n切换模型示例:")
    print("   OLLAMA_MODEL=qwen3.5:9b python main_mini.py")
    print("\n例如，你可以输入:")
    print("   '在当前目录创建一个hello.txt文件，内容是Hello World'")
    print("   '列出当前目录的所有文件'")
    print("   '写一个Python程序计算1到100的和'")
    
    # 交互循环
    while True:
        try:
            user_input = input("\n请输入你的任务 (输入 'quit' 退出): ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见!")
                break
            
            if not user_input.strip():
                continue
            
            # 执行任务
            result = await agent.run(user_input)
            print(f"\n📋 执行结果:\n{result}")
            
        except KeyboardInterrupt:
            print("\n👋 程序被中断，再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())
