"""
RAG chain implementation using DeepSeek with optimized single LLM call.
Handles tool calling intelligently with proper context formatting and token tracking.
"""
from typing import Dict, Any
from pymilvus import utility, connections
from app.RAG.deepseek_client import DeepSeekClient
from app.RAG.prompt import get_system_prompt
from app.config import Settings
from app.api.chat.models.chat_model import ChatRequest
from tools.tools_schema import RETRIEVAL_TOOL_SCHEMA, SEARCH_ARTICLES_SCHEMA, WEATHER_TOOL_SCHEMA
from tools.functions import execute_tool
from loguru import logger
import json


async def execute_rag_chain(request: ChatRequest, collection_name: str, settings: Settings) -> Dict:
    """
    Execute RAG chain using DeepSeek with single optimized LLM call.
    
    Args:
        request: Chat request with user query
        collection_name: Milvus collection to search
        settings: Application settings
        
    Returns:
        Response with answer and token usage
    """
    logger.info(f"[RAG_CHAIN] Starting execution for collection: {collection_name}")
    logger.info(f"[RAG_CHAIN] User query: {request.user_query}")
    
    # Connect to Milvus
    connections.connect(
        uri=settings.MILVUS_URI,
        db_name=settings.MILVUS_DB_NAME,
        token=settings.MILVUS_TOKEN
    )

    # Check if collection exists
    collection_exists = utility.has_collection(collection_name)
    if not collection_exists:
        logger.warning(f"[RAG_CHAIN] Collection {collection_name} does not exist")

    # Initialize DeepSeek client
    deepseek = DeepSeekClient(
        api_key=settings.DEEPSEEK_API_KEY,
        model="deepseek-chat"
    )
    logger.info("[RAG_CHAIN] DeepSeek client initialized")
    
    # Prepare messages with system prompt
    messages = [
        {
            "role": "system",
            "content": get_system_prompt()
        },
        {
            "role": "user",
            "content": request.user_query
        }
    ]
    
    # Track token usage
    total_token_usage = {
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "embedding_tokens": 0,
        "rerank_tokens": 0
    }
    
    # Single LLM call with tools - let DeepSeek decide
    logger.info("[RAG_CHAIN] Making DeepSeek call with tools")
    response = deepseek.chat_completion(
        messages=messages,
        tools=[RETRIEVAL_TOOL_SCHEMA, SEARCH_ARTICLES_SCHEMA, WEATHER_TOOL_SCHEMA],
        tool_choice="auto",
        temperature=0.6
    )
    
    # Update token usage
    usage = deepseek.get_usage(response)
    total_token_usage["llm_input_tokens"] += usage.get("prompt_tokens", 0)
    total_token_usage["llm_output_tokens"] += usage.get("completion_tokens", 0)
    logger.info(f"[RAG_CHAIN] Initial call tokens: {usage}")
    
    # Check if tool calls were made
    tool_calls = response.choices[0].message.tool_calls if response.choices[0].message.tool_calls else []
    
    if tool_calls:
        logger.info(f"[RAG_CHAIN] Processing {len(tool_calls)} tool call(s)")
        
        # Execute tool calls and collect results
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            logger.info(f"[RAG_CHAIN] Executing tool: {function_name}")
            logger.info(f"[RAG_CHAIN] Tool arguments: {function_args}")
            
            # Execute the tool and get token usage
            tool_result, tool_tokens = execute_tool(
                function_name=function_name,
                function_args=function_args,
                collection_name=collection_name
            )
            
            # Aggregate token usage from tool execution
            total_token_usage["embedding_tokens"] += tool_tokens.get("embedding_tokens", 0)
            total_token_usage["rerank_tokens"] += tool_tokens.get("rerank_tokens", 0)
            
            logger.info(f"[RAG_CHAIN] Tool result length: {len(tool_result)} characters")
            logger.info(f"[RAG_CHAIN] Tool tokens: {tool_tokens}")
            
            # Add tool call to messages
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": tool_call.function.arguments
                    }
                }]
            })
            
            # Format context properly for retrieval tool
            if function_name == "retrieve_documents":
                formatted_context = f"\n**Retrieved Context:**\n{tool_result}\n"
                logger.info(f"[RAG_CHAIN] context:{formatted_context}")
                tool_result = formatted_context
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
        
        # Second call: Generate final answer with tool results
        logger.info("[RAG_CHAIN] Making final DeepSeek call with tool results")
        final_response = deepseek.chat_completion(
            messages=messages,
            temperature=0.6
        )
        
        final_usage = deepseek.get_usage(final_response)
        total_token_usage["llm_input_tokens"] += final_usage.get("prompt_tokens", 0)
        total_token_usage["llm_output_tokens"] += final_usage.get("completion_tokens", 0)
        logger.info(f"[RAG_CHAIN] Final call tokens: {final_usage}")
        
        answer = final_response.choices[0].message.content
    else:
        # No tool calls, use direct response
        logger.info("[RAG_CHAIN] No tool calls made, using direct response")
        answer = response.choices[0].message.content
    
    logger.info(f"[RAG_CHAIN] Generated answer length: {len(answer)} characters")
    logger.info(f"[RAG_CHAIN] Total token usage: {total_token_usage}")
    
    # Return response
    return {
        "response": answer,
        "token_usage": total_token_usage
    }