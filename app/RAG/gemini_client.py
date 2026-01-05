"""
Gemini client wrapper using google-genai SDK.
Provides chat completions with tool calling support.
Maintains compatible interface with previous DeepSeekClient.
"""
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class ToolCall:
    """Represents a tool call from the model."""
    id: str
    function: 'FunctionCall'


@dataclass  
class FunctionCall:
    """Represents a function call."""
    name: str
    arguments: str  # JSON string of arguments


@dataclass
class Message:
    """Represents a message in the response."""
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]


@dataclass
class Choice:
    """Represents a choice in the response."""
    message: Message


@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletionResponse:
    """OpenAI-compatible response wrapper for Gemini responses."""
    choices: List[Choice]
    usage: Optional[Usage]


class GeminiClient:
    """
    Gemini client wrapper using google-genai SDK.
    Provides OpenAI-compatible interface for easy migration.
    Supports chat completions and tool calling.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Model name to use (default: gemini-2.5-flash)
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model
    
    def _convert_openai_tools_to_gemini(self, tools: List[Dict[str, Any]]) -> List[types.Tool]:
        """
        Convert OpenAI-format tool definitions to Gemini format.
        
        Args:
            tools: List of OpenAI-format tool definitions
            
        Returns:
            List of Gemini Tool objects
        """
        function_declarations = []
        
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                
                # Convert OpenAI parameters to Gemini schema
                params = func.get("parameters", {})
                
                function_declarations.append(types.FunctionDeclaration(
                    name=func.get("name", ""),
                    description=func.get("description", ""),
                    parameters=params if params else None
                ))
        
        if function_declarations:
            return [types.Tool(function_declarations=function_declarations)]
        return []
    
    def _convert_messages_to_gemini(self, messages: List[Dict[str, Any]]) -> tuple:
        """
        Convert OpenAI-format messages to Gemini format.
        
        Args:
            messages: List of OpenAI-format messages
            
        Returns:
            Tuple of (system_instruction, contents)
        """
        system_instruction = None
        contents = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)]
                ))
            elif role == "assistant":
                # Handle assistant messages with potential tool calls
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    # Create function call parts
                    parts = []
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        try:
                            args = json.loads(func.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {}
                        parts.append(types.Part.from_function_call(
                            name=func.get("name", ""),
                            args=args
                        ))
                    contents.append(types.Content(role="model", parts=parts))
                elif content:
                    contents.append(types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=content)]
                    ))
            elif role == "tool":
                # Tool result message
                tool_call_id = msg.get("tool_call_id", "")
                tool_content = msg.get("content", "")
                
                # Find the function name from the tool_call_id in previous messages
                func_name = ""
                for prev_msg in messages:
                    if prev_msg.get("role") == "assistant":
                        for tc in prev_msg.get("tool_calls", []):
                            if tc.get("id") == tool_call_id:
                                func_name = tc.get("function", {}).get("name", "")
                                break
                
                # Try to parse content as JSON, otherwise wrap in dict
                try:
                    if isinstance(tool_content, str):
                        result = json.loads(tool_content)
                    else:
                        result = tool_content
                except json.JSONDecodeError:
                    result = {"result": tool_content}
                
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=func_name,
                        response=result
                    )]
                ))
        
        return system_instruction, contents
    
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
    ) -> ChatCompletionResponse:
        """
        Create a chat completion with optional tool calling.
        Returns OpenAI-compatible response format.
        
        Args:
            messages: List of message dictionaries (OpenAI format)
            tools: Optional list of tool definitions (OpenAI format)
            tool_choice: Optional tool choice strategy ("auto", "none")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            OpenAI-compatible ChatCompletionResponse
        """
        try:
            # Convert messages
            system_instruction, contents = self._convert_messages_to_gemini(messages)
            
            # Build config
            config_kwargs = {
                "temperature": temperature,
            }
            
            if max_tokens:
                config_kwargs["max_output_tokens"] = max_tokens
            
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction
            
            # Convert and add tools if provided
            gemini_tools = None
            if tools and tool_choice != "none":
                gemini_tools = self._convert_openai_tools_to_gemini(tools)
                if gemini_tools:
                    config_kwargs["tools"] = gemini_tools
                    
                    # Set tool config based on tool_choice
                    if tool_choice == "auto":
                        config_kwargs["tool_config"] = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="AUTO"
                            )
                        )
            
            generate_config = types.GenerateContentConfig(**config_kwargs)
            
            # Make the API call
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_config
            )
            
            # Convert response to OpenAI-compatible format
            return self._convert_response_to_openai_format(response)
            
        except Exception as e:
            logger.error(f"Error in Gemini chat completion: {e}")
            raise
    
    def _convert_response_to_openai_format(self, response) -> ChatCompletionResponse:
        """
        Convert Gemini response to OpenAI-compatible format.
        
        Args:
            response: Gemini GenerateContentResponse
            
        Returns:
            OpenAI-compatible ChatCompletionResponse
        """
        content = None
        tool_calls = None
        
        # Extract content and tool calls from response
        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            text_parts = []
            function_calls = []
            
            for i, part in enumerate(parts):
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
                elif hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    function_calls.append(ToolCall(
                        id=f"call_{i}",
                        function=FunctionCall(
                            name=fc.name,
                            arguments=json.dumps(dict(fc.args) if fc.args else {})
                        )
                    ))
            
            if text_parts:
                content = "".join(text_parts)
            if function_calls:
                tool_calls = function_calls
        
        # Extract usage if available
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = Usage(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
                total_tokens=response.usage_metadata.total_token_count or 0
            )
        
        return ChatCompletionResponse(
            choices=[Choice(message=Message(content=content, tool_calls=tool_calls))],
            usage=usage
        )
    
    def get_usage(self, response: ChatCompletionResponse) -> Dict[str, int]:
        """
        Extract token usage from response.
        
        Args:
            response: Chat completion response
            
        Returns:
            Dictionary with token usage metrics
        """
        if response.usage:
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# Alias for backward compatibility
DeepSeekClient = GeminiClient
