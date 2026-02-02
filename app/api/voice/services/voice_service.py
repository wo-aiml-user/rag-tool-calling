"""
Voice Agent service for building Deepgram Voice Agent configuration.
Uses existing prompt and function schemas from the application.
"""
from typing import Dict, List
from loguru import logger
from app.config import Settings
from app.api.voice.services.prompt import get_voice_prompt


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
                    "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:streamGenerateContent?alt=sse",
                    "headers": {
                        "x-goog-api-key": settings.GEMINI_API_KEY
                    }
                },
                "prompt": voice_prompt
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

