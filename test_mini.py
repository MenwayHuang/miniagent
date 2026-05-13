"""
MiniAgent 简单测试
"""
import asyncio
import os
from mini_agent import MiniAgent
from mini_agent.schema import Memory, Message
from mini_agent.llm import (
    create_llm_from_env,
    OllamaLLM,
    SimpleLLM,
)
from mini_agent.tools import PythonExecutor, FileEditor, BashExecutor


async def test_tools():
    """测试工具功能"""
    print("=== 测试工具功能 ===")
    
    # 测试Python执行工具
    python_tool = PythonExecutor()
    result = await python_tool.execute(code="print('Hello from Python tool!')")
    print(f"Python工具测试: {result}")

    result = await python_tool.execute(code="sum_of_numbers = sum(range(1, 201))\nsum_of_numbers")
    assert result.success
    assert "没有stdout输出" in result.output
    assert "print" in result.output
    print(f"Python无stdout提示测试: {result}")
    
    # 测试文件编辑工具
    file_tool = FileEditor()
    result = await file_tool.execute(action="write", path="test.txt", content="测试内容")
    print(f"文件工具测试: {result}")
    
    result = await file_tool.execute(action="read", path="test.txt")
    print(f"文件读取测试: {result}")
    
    # 测试Bash工具
    bash_tool = BashExecutor()
    result = await bash_tool.execute(command="echo 'Hello from bash!'")
    print(f"Bash工具测试: {result}")


async def test_agent_without_llm():
    """测试代理功能（不依赖真实LLM）"""
    print("\n=== 测试代理结构 ===")
    
    # 创建一个模拟的LLM（不实际调用API）
    class MockLLM:
        async def chat(self, messages, system_prompt=None, tools=None):
            from mini_agent.llm import LLMResponse
            return LLMResponse(content="这是模拟响应，实际使用需要真实模型配置")
    
    # 创建代理
    mock_llm = MockLLM()
    agent = MiniAgent(mock_llm, name="TestAgent")
    
    print(f"代理名称: {agent.name}")
    print(f"工具数量: {len(agent.tools.tools)}")
    print(f"可用工具: {list(agent.tools.tools.keys())}")
    print(f"初始状态: {agent.state}")


def test_llm_configuration():
    """测试LLM配置工厂（不发起网络请求）"""
    print("\n=== 测试LLM配置 ===")

    ollama_llm = OllamaLLM()
    assert ollama_llm.model == "qwen2.5:7b"
    assert ollama_llm.base_url == "http://127.0.0.1:11434/v1"
    print(f"Ollama默认配置: model={ollama_llm.model}, base_url={ollama_llm.base_url}")

    old_env = {
        key: os.environ.get(key)
        for key in [
            "MINI_AGENT_PROVIDER",
            "OLLAMA_MODEL",
            "OLLAMA_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
            "OPENAI_BASE_URL",
        ]
    }

    try:
        os.environ["MINI_AGENT_PROVIDER"] = "ollama"
        os.environ["OLLAMA_MODEL"] = "qwen3.5:9b"
        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"
        env_llm = create_llm_from_env()
        assert isinstance(env_llm, OllamaLLM)
        assert env_llm.model == "qwen3.5:9b"
        assert env_llm.base_url == "http://localhost:11434/v1"
        print(f"Ollama环境变量配置: model={env_llm.model}, base_url={env_llm.base_url}")

        os.environ["MINI_AGENT_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
        os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
        openai_llm = create_llm_from_env()
        assert isinstance(openai_llm, SimpleLLM)
        assert not isinstance(openai_llm, OllamaLLM)
        assert openai_llm.model == "gpt-4o-mini"
        assert openai_llm.base_url == "https://api.openai.com/v1"
        print(f"OpenAI环境变量配置: model={openai_llm.model}, base_url={openai_llm.base_url}")

    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_memory_preserves_empty_message_content():
    """测试空字符串content不会被序列化成缺失字段"""
    print("\n=== 测试空消息内容序列化 ===")

    memory = Memory()
    memory.add_message(Message.assistant_message(content="", tool_calls=[{
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "python_execute",
            "arguments": '{"code": "x = 1"}',
        },
    }]))
    memory.add_message(Message.tool_message(content="", tool_call_id="call_123"))

    messages = memory.get_messages()
    assert messages[0]["content"] == ""
    assert messages[1]["content"] == ""
    print(f"空content序列化结果: {messages}")


async def main():
    """主测试函数"""
    print("🧪 MiniAgent 功能测试")
    print("注意: 这个测试不需要连接真实模型，只测试基础功能")
    
    await test_tools()
    await test_agent_without_llm()
    test_llm_configuration()
    test_memory_preserves_empty_message_content()
    
    print("\n✅ 基础功能测试完成!")
    print("💡 要测试完整功能，请先启动Ollama，然后运行 main_mini.py")


if __name__ == "__main__":
    asyncio.run(main())
