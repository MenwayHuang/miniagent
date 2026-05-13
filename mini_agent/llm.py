"""
LLM接口实现
"""
import os
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """LLM响应结果"""
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class SimpleLLM:
    """简化的LLM接口"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        provider: str = "openai",
    ):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.base_url = base_url
        self.provider = provider
    
    async def chat(
        self, 
        messages: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """发送聊天请求"""
        
        # 构建消息列表
        chat_messages = []
        
        # 添加系统消息
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        
        # 添加对话历史
        chat_messages.extend(messages)
        
        # 构建请求参数
        request_params = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": 0.7,
        }
        
        # 如果有工具，添加工具调用参数
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"
        
        try:
            # 调用OpenAI兼容API
            response = await self.client.chat.completions.create(**request_params)
            
            message = response.choices[0].message
            
            # 解析响应
            result = LLMResponse()
            result.content = message.content
            
            # 解析工具调用
            if message.tool_calls:
                result.tool_calls = []
                for tool_call in message.tool_calls:
                    result.tool_calls.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            return result
            
        except Exception as e:
            print(f"LLM调用错误: {e}")
            return LLMResponse(content=f"LLM调用失败: {str(e)}")


class OllamaLLM(SimpleLLM):
    """本地Ollama LLM适配器，复用OpenAI兼容接口"""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://127.0.0.1:11434/v1",
        api_key: str = "ollama",
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider="ollama",
        )


def create_llm_from_env() -> SimpleLLM:
    """根据环境变量创建LLM实例，默认使用本地Ollama"""
    provider = os.getenv("MINI_AGENT_PROVIDER", "ollama").strip().lower()

    if provider == "ollama":
        return OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("使用OpenAI时请设置环境变量 OPENAI_API_KEY")

        return SimpleLLM(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    raise ValueError(
        "不支持的 MINI_AGENT_PROVIDER: "
        f"{provider}，可选值为 ollama 或 openai"
    )
