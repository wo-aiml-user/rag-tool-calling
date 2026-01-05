"""
Voice Agent service for building Gemini Live API configuration.
Uses Google's Gemini native audio model for voice-to-voice conversations.
"""
from typing import Dict
from loguru import logger
from google.genai import types
from app.RAG.prompt import get_voice_prompt


# Audio configuration constants
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000

# Gemini model for native audio
GEMINI_VOICE_MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"


def get_gemini_live_config(
    name: str = "", 
    role: str = "", 
    years_of_experience: str = ""
) -> types.LiveConnectConfig:
    """
    Get the Gemini Live API configuration for voice-to-voice conversations.
    
    Args:
        name: User's name (collected before call)
        role: User's role/position (collected before call)
        years_of_experience: Years of frontend experience (collected before call)
    
    Returns:
        LiveConnectConfig for Gemini Live API connection
    """
    logger.info(f"[VOICE_SERVICE] Building Gemini Live config for user: {name}, role: {role}, exp: {years_of_experience}")
    
    # Get voice-optimized prompt with user context
    voice_prompt = get_voice_prompt(name=name, role=role, years_of_experience=years_of_experience)
    logger.info("[VOICE_SERVICE] Loaded personalized voice system instruction")
    
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=voice_prompt,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
            ),
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=25600,
            sliding_window=types.SlidingWindow(target_tokens=12800),
        ), 
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
    
    logger.info("[VOICE_SERVICE] Gemini Live config created with voice='Alnilam'")
    return config


def get_audio_config() -> Dict:
    """
    Get audio configuration for client communication.
    
    Returns:
        Dictionary with input/output audio settings
    """
    return {
        "input": {
            "encoding": "linear16",
            "sample_rate": SEND_SAMPLE_RATE,
            "channels": 1
        },
        "output": {
            "encoding": "linear16",
            "sample_rate": RECEIVE_SAMPLE_RATE,
            "channels": 1
        }
    }
