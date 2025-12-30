"""
Voice Agent service for building Grok Voice Agent configuration.
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


def convert_to_grok_format(openai_tools: List[Dict]) -> List[Dict]:
    """
    Convert OpenAI/DeepSeek function calling format to Grok Voice Agent format.
    
    OpenAI format:
    {
        "type": "function",
        "function": {
            "name": "...",
            "description": "...",
            "parameters": {...}
        }
    }
    
    Grok format (same as OpenAI for tools):
    {
        "type": "function",
        "name": "...",
        "description": "...",
        "parameters": {...}
    }
    """
    grok_tools = []
    for tool in openai_tools:
        if tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            grok_tools.append({
                "type": "function",
                "name": func.get("name"),
                "description": func.get("description"),
                "parameters": func.get("parameters", {})
            })
        else:
            grok_tools.append(tool)
    return grok_tools


def get_function_definitions_grok() -> List[Dict]:
    """
    Get function definitions in Grok Voice Agent format.
    """
    openai_tools = get_function_definitions_openai()
    return convert_to_grok_format(openai_tools)


async def get_voice_agent_settings(settings: Settings) -> Dict:
    """
    Configure Grok Voice Agent session.
    Uses the existing prompt and function schema from the application.
    
    Args:
        settings: Application settings containing API keys
        
    Returns:
        Voice Agent session.update message for Grok API
    """
    logger.info("[VOICE_SERVICE] Building Grok voice agent settings")
    
    # Get function definitions in Grok format
    function_definitions = get_function_definitions_grok()
    logger.info(f"[VOICE_SERVICE] Loaded {len(function_definitions)} function definitions (Grok format)")
    for func in function_definitions:
        logger.debug(f"[VOICE_SERVICE] Tool: {func.get('name')}")
    
    # Get voice-optimized prompt
    voice_prompt = get_voice_prompt()
    logger.info("[VOICE_SERVICE] Loaded voice-optimized prompt")
    
    # Build session.update message for Grok Voice Agent
    session_config = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": voice_prompt,
            "voice": "aura",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 16000
                    }
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    }
                }
            },
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "turn_detection": {
                "type": "server_vad",
                "interrupt_response": True,  # Allow user speech to interrupt agent response
                "silence_duration_ms": 300,  # Pause duration to detect end of speech
                "threshold": 0.5,  # Voice activity detection sensitivity
            },
            "tools": function_definitions,
            "tool_choice": "auto",
            "temperature": 0.4
        }
    }
    
    logger.info("[VOICE_SERVICE] Session config built successfully")
    logger.debug(f"[VOICE_SERVICE] Config: modalities={session_config['session']['modalities']}, voice={session_config['session']['voice']}")
    
    return session_config


# Legacy function for backward compatibility
def convert_to_deepgram_format(openai_tools: List[Dict]) -> List[Dict]:
    """
    Deprecated: Use convert_to_grok_format instead.
    Kept for backward compatibility.
    """
    return convert_to_grok_format(openai_tools)


def get_function_definitions_deepgram() -> List[Dict]:
    """
    Deprecated: Use get_function_definitions_grok instead.
    Kept for backward compatibility.
    """
    return get_function_definitions_grok()
