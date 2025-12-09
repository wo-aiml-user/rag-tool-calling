"""
DeepSeek client wrapper using OpenAI SDK.
Provides chat completions with tool calling support.
"""
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class DeepSeekClient:
    """
    DeepSeek client wrapper using OpenAI SDK.
    Supports chat completions and tool calling.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            model: Model name to use
        """
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model
    
    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4)
    )
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: Optional[int] = None
    ) -> Any:
        """
        Create a chat completion with optional tool calling.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            tool_choice: Optional tool choice strategy ("auto", "none", or specific tool)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Chat completion response
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            
            if tools:
                kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice
            
            response = self.client.chat.completions.create(**kwargs)
            return response
            
        except Exception as e:
            logger.error(f"Error in DeepSeek chat completion: {e}")
            raise
    
    def get_usage(self, response: Any) -> Dict[str, int]:
        """
        Extract token usage from response.
        
        Args:
            response: Chat completion response
            
        Returns:
            Dictionary with token usage metrics
        """
        if hasattr(response, 'usage') and response.usage:
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
