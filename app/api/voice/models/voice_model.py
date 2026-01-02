"""
Pydantic models for Gemini Voice Agent configuration.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class AudioInputConfig(BaseModel):
    """Audio input configuration for voice capture."""
    encoding: str = Field(default="linear16", description="Audio encoding format")
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")


class AudioOutputConfig(BaseModel):
    """Audio output configuration for playback."""
    encoding: str = Field(default="linear16", description="Audio encoding format")
    sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")


class AudioConfig(BaseModel):
    """Combined audio configuration."""
    input: AudioInputConfig = Field(default_factory=AudioInputConfig)
    output: AudioOutputConfig = Field(default_factory=AudioOutputConfig)


class VoiceAgentConfig(BaseModel):
    """Configuration for the Gemini voice agent session."""
    language: str = Field(default="en", description="Language for the agent")
    model: str = Field(
        default="models/gemini-2.5-flash-native-audio-preview-09-2025",
        description="Gemini model for voice"
    )
    voice_name: str = Field(default="Alnilam", description="Voice preset name")
    temperature: float = Field(default=0.7, description="Response temperature")
    max_turns: int = Field(default=20, description="Maximum conversation turns")
    greeting: Optional[str] = Field(
        default="Hello! I'm Maya, a Business Consultant. How can I help you today?",
        description="Initial greeting message"
    )


class TokenUsage(BaseModel):
    """Token usage tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total: int = 0


class VoiceSessionStatus(BaseModel):
    """Status information for a voice session."""
    session_id: str
    is_active: bool = True
    turn_count: int = 0
    max_turns: int = 20
    message_count: int = 0
    audio_chunk_count: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
