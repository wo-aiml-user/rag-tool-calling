"""
Voice Agent WebSocket Controller.
Handles WebSocket connections for real-time voice interactions using Gemini Live API.
"""
import json
import base64
import asyncio
from typing import Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
from app.config import settings
from app.api.voice.services.voice_session import GeminiVoiceSession
from app.api.voice.services.voice_service import get_audio_config


router = APIRouter()

# Store active sessions
active_sessions: Dict[str, GeminiVoiceSession] = {}


@router.websocket("/ws/voice/{session_id}")
async def websocket_voice_endpoint(
    websocket: WebSocket,
    session_id: str
):

    """
    Handle WebSocket connection for Gemini voice agent.
    
    Message Types (Client -> Server):
    - start_session: Initialize Gemini Live connection
    - audio_chunk: Forward audio data (base64 encoded in audio_data field)
    - end_session: Close the session
    - get_stats: Get current session statistics
    
    Message Types (Server -> Client):
    - session_started: Session initialized successfully
    - audio_config: Audio settings for client
    - speech_started: User started speaking
    - playback_started: Audio playback beginning
    - playback_finished: Audio playback complete
    - audio_chunk: Audio data from Gemini TTS (base64 encoded)
    - transcript: User speech transcript
    - response: Agent text response
    - token_usage: Token usage for the turn
    - session_stats: Session statistics
    - transcript_analysis: Post-session analysis
    - error: Error message
    """
    await websocket.accept()
    logger.info(f"[{session_id}] Client connected")
    
    # Session will be created when start_session is received with user context
    session = None
    
    try:
        while True:
            message = await websocket.receive()
            
            if message.get("type") == "websocket.disconnect":
                break
                
            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")
                
                if msg_type == "start_session":
                    # Extract user context from the message
                    user_name = data.get("name", "")
                    user_role = data.get("role", "") 
                    user_experience = data.get("years_of_experience", "")
                    
                    logger.info(f"[{session_id}] Starting session for {user_name} ({user_role}, {user_experience} years exp)")
                    
                    # Create session with user context
                    session = GeminiVoiceSession(
                        session_id=session_id,
                        client_ws=websocket,
                        settings=settings,
                        name=user_name,
                        role=user_role,
                        years_of_experience=user_experience
                    )
                    active_sessions[session_id] = session
                    
                    success = await session.connect_and_run()
                    
                    if success:
                        # Send session started and audio config
                        await session.client_ws.send_text(json.dumps({
                            "type": "session_started",
                            "session_id": session_id,
                            "user_context": {
                                "name": user_name,
                                "role": user_role,
                                "years_of_experience": user_experience
                            }
                        }))
                        await session.client_ws.send_text(json.dumps({
                            "type": "audio_config",
                            "config": get_audio_config()
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Failed to connect to Gemini Live API"
                        }))
                
                elif msg_type == "audio_chunk":
                    # Decode and forward audio to Gemini
                    if session and "audio_data" in data:
                        audio_bytes = base64.b64decode(data["audio_data"])
                        await session.forward_audio_to_agent(audio_bytes)
                
                elif msg_type == "get_stats":
                    # Send current session stats
                    if session:
                        stats = session.get_session_stats()
                        await session.client_ws.send_text(json.dumps({
                            "type": "session_stats",
                            "stats": stats
                        }))
                
                elif msg_type == "end_session":
                    logger.info(f"[{session_id}] Ending session...")
                    break
            
            elif "bytes" in message:
                # Raw binary audio - forward directly
                if session:
                    await session.forward_audio_to_agent(message["bytes"])
    
    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected")
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if session:
            await session.close()
        if session_id in active_sessions:
            del active_sessions[session_id]


async def get_active_session(session_id: str) -> GeminiVoiceSession | None:
    """Get an active voice session by ID."""
    return active_sessions.get(session_id)


async def get_active_session_count() -> int:
    """Get the number of active voice sessions."""
    return len(active_sessions)
