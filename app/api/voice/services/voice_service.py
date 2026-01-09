"""
Voice Agent service for building Deepgram Voice Agent configuration.
Uses existing prompt and function schemas from the application.
"""
from typing import Dict, List
from loguru import logger
from app.config import Settings
from tools.tools_schema import retrieval_tool, search_articles, weather_tool
from app.RAG.prompt import get_voice_prompt

def get_function_definitions_openai() -> List[Dict]:
    """
    Get the function definitions in OpenAI format.
    Returns the same tools used in the chat RAG pipeline.
    """
    return [retrieval_tool, search_articles, weather_tool]


def convert_to_deepgram_format(openai_tools: List[Dict]) -> List[Dict]:
    """
    Convert OpenAI function calling format to Deepgram Voice Agent format.
    
    OpenAI format:
    {
        "type": "function",
        "function": {
            "name": "...",
            "description": "...",
            "parameters": {...}
        }
    }
    
    Deepgram format:
    {
        "name": "...",
        "description": "...",
        "parameters": {...}
    }
    """
    deepgram_tools = []
    for tool in openai_tools:
        if tool.get("type") == "function" and "function" in tool:
            # Extract the inner function definition
            deepgram_tools.append(tool["function"])
        else:
            deepgram_tools.append(tool)
    return deepgram_tools


def get_function_definitions_deepgram() -> List[Dict]:
    """
    Get function definitions in Deepgram Voice Agent format.
    """
    openai_tools = get_function_definitions_openai()
    return convert_to_deepgram_format(openai_tools)


async def get_voice_agent_settings(settings: Settings, context: dict = None) -> Dict:
    """
    Configure Deepgram Voice Agent with Gemini as the LLM provider.
    Uses the existing prompt and function schema from the application.
    
    Args:
        settings: Application settings containing API keys
        context: Optional dict with pre-provided user info (name, role, years_of_experience)
        
    Returns:
        Voice Agent settings dictionary for Deepgram API
    """
    context = context or {}
    logger.info(f"[VOICE_SERVICE] Building voice agent settings with Gemini (context: {context})")
    
    # Get function definitions in Deepgram format
    function_definitions = get_function_definitions_deepgram()
    logger.info(f"[VOICE_SERVICE] Loaded {len(function_definitions)} function definitions (Deepgram format)")
    
    # Get voice-optimized prompt with user context
    voice_prompt = get_voice_prompt(context=context)
    logger.info("[VOICE_SERVICE] Loaded voice-optimized prompt")
    
    # Customize greeting based on provided context
    if context.get('name'):
        greeting = f"Hey {context['name']}, this is Jane. How is it going?"
    else:
        greeting = "Hello! I'm Jane, how is it going?"
    
    return {
        "type": "Settings",
        "audio": {
            "input": {
                "encoding": "linear16",
                "sample_rate": 16000
            },
            "output": {
                "encoding": "linear16",
                "sample_rate": 24000,
                "container": "none"
            }
        },
        "agent": {
            "language": "en",
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "model": "nova-3"
                }
            },
            "think": {
                "provider": {
                    "type": "google",
                    "temperature": 0.7
                },
                "endpoint": {
                    "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:streamGenerateContent?alt=sse",
                    "headers": {
                        "x-goog-api-key": settings.GEMINI_API_KEY
                    }
                },
                "prompt": voice_prompt,
                "functions": function_definitions
            },
            "speak": {
                "provider": {
                    "type": "deepgram",
                    "model": "aura-2-janus-en"
                }
            },
            "greeting": greeting
        }
    }

