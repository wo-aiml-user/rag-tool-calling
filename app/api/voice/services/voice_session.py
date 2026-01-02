"""
Voice Agent Session management using Gemini Live API.
Handles real-time voice-to-voice conversations with Google's Gemini native audio model.
"""
import os
import json
import base64
import asyncio
import time
from typing import Optional, List, Dict
from fastapi import WebSocket
from loguru import logger
from google import genai
from google.genai import types
from app.config import Settings
from app.api.voice.services.voice_service import (
    get_gemini_live_config,
    get_audio_config,
    GEMINI_VOICE_MODEL,
    SEND_SAMPLE_RATE,
    RECEIVE_SAMPLE_RATE
)
from app.RAG.prompt import get_transcript_analysis_prompt
import traceback


class GeminiVoiceSession:
    """Manages a real-time voice session using Gemini Live API."""
    
    def __init__(self, session_id: str, client_ws: WebSocket, settings: Settings, max_turns: int = 20):
        self.session_id = session_id
        self.client_ws = client_ws
        self.settings = settings
        self.is_active = True
        self.start_time: Optional[float] = None
        
        # Gemini session
        self.gemini_session = None
        self.gemini_client = None
        
        # Audio queue for playback
        self.audio_out_queue: asyncio.Queue = asyncio.Queue()
        
        # State tracking
        self.last_speaker: Optional[str] = None
        self.ai_speaking = False
        self.audio_chunk_count = 0
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_thinking_tokens = 0
        self.turn_count = 0
        self.max_turns = max_turns
        
        # Conversation history for analysis
        self.conversation_history: List[Dict[str, str]] = []
        
        # Main session task
        self.session_task: Optional[asyncio.Task] = None
        
        # Event to signal when session should stop
        self._stop_event = asyncio.Event()
    
    async def connect_and_run(self) -> bool:
        """Connect to Gemini Live API and start the audio streaming session."""
        try:
            gemini_api_key = self.settings.GEMINI_API_KEY
            if not gemini_api_key:
                logger.error(f"[{self.session_id}] GEMINI_API_KEY not configured")
                return False
            
            logger.info(f"[{self.session_id}] Connecting to Gemini Live API...")
            
            # Initialize Gemini client
            self.gemini_client = genai.Client(
                http_options={"api_version": "v1beta"},
                api_key=gemini_api_key,
            )
            
            # Get the live config
            config = get_gemini_live_config()
            
            # Start the session task that manages the context manager
            self.session_task = asyncio.create_task(self._run_session(config))
            
            # Wait a moment for connection to establish
            await asyncio.sleep(0.5)
            
            if self.gemini_session:
                logger.info(f"[{self.session_id}] Connected to Gemini Live API")
                return True
            else:
                logger.error(f"[{self.session_id}] Failed to establish Gemini session")
                return False
                
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to connect to Gemini Live: {e}")
            traceback.print_exc()
            return False
    
    async def _run_session(self, config):
        """Run the Gemini Live session within async context manager."""
        try:
            async with self.gemini_client.aio.live.connect(
                model=GEMINI_VOICE_MODEL,
                config=config
            ) as session:
                self.gemini_session = session
                logger.info(f"[{self.session_id}] Gemini session established")
                
                # Create tasks for receiving and sending audio
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._receive_from_gemini())
                    tg.create_task(self._send_audio_to_client())
                    
                    # Wait until stop event is set
                    await self._stop_event.wait()
                    
        except asyncio.CancelledError:
            logger.info(f"[{self.session_id}] Session task cancelled")
        except Exception as e:
            logger.error(f"[{self.session_id}] Session error: {e}")
            traceback.print_exc()
        finally:
            self.gemini_session = None
    
    async def forward_audio_to_agent(self, audio_data: bytes):
        """Forward audio from client to Gemini Live API."""
        if self.gemini_session and not self.ai_speaking:
            try:
                await self.gemini_session.send(
                    input={"data": audio_data, "mime_type": "audio/pcm"}
                )
            except Exception as e:
                logger.error(f"[{self.session_id}] Error sending audio to Gemini: {e}")
    
    async def _receive_from_gemini(self):
        """Background task to receive responses from Gemini Live API."""
        try:
            while self.is_active and self.gemini_session:
                try:
                    turn = self.gemini_session.receive()
                    
                    turn_input_tokens = 0
                    turn_output_tokens = 0
                    turn_thinking_tokens = 0
                    user_text = ""
                    model_text = ""
                    
                    async for response in turn:
                        if not self.is_active:
                            break
                            
                        # Handle audio data
                        if data := response.data:
                            self.audio_out_queue.put_nowait(data)
                        
                        # Handle input transcription (user speech)
                        if (server_content := response.server_content) and server_content.input_transcription:
                            if self.last_speaker != "User":
                                self.last_speaker = "User"
                                self.start_time = time.perf_counter()
                            
                            text_chunk = server_content.input_transcription.text
                            user_text += text_chunk
                            
                            # Send transcript to client
                            await self._send_to_client({
                                "type": "transcript",
                                "text": text_chunk,
                                "role": "user"
                            })
                        
                        # Handle output transcription (model speech)
                        if (server_content := response.server_content) and server_content.output_transcription:
                            if self.last_speaker != "Model":
                                self.last_speaker = "Model"
                                if self.start_time:
                                    latency_ms = int((time.perf_counter() - self.start_time) * 1000)
                                    logger.info(f"[{self.session_id}] ⚡ First response (latency: {latency_ms}ms)")
                                
                                await self._send_to_client({"type": "playback_started"})
                            
                            text_chunk = server_content.output_transcription.text
                            model_text += text_chunk
                            
                            # Send response text to client
                            await self._send_to_client({
                                "type": "response",
                                "text": text_chunk,
                                "role": "assistant"
                            })
                        
                        # Track token usage
                        if hasattr(response, 'usage_metadata') and response.usage_metadata:
                            usage = response.usage_metadata
                            if hasattr(usage, 'prompt_token_count') and usage.prompt_token_count:
                                turn_input_tokens = usage.prompt_token_count
                            if hasattr(usage, 'response_token_count') and usage.response_token_count:
                                turn_output_tokens = usage.response_token_count
                            if hasattr(usage, 'thoughts_token_count') and usage.thoughts_token_count:
                                turn_thinking_tokens = usage.thoughts_token_count
                    
                    # Save to conversation history
                    if user_text:
                        self.conversation_history.append({
                            "role": "user",
                            "content": user_text.strip(),
                            "turn": self.turn_count + 1
                        })
                        logger.info(f"[{self.session_id}] User: {user_text.strip()[:100]}...")
                    
                    if model_text:
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": model_text.strip(),
                            "turn": self.turn_count + 1
                        })
                        logger.info(f"[{self.session_id}] Model: {model_text.strip()[:100]}...")
                    
                    # Update token counters
                    if turn_input_tokens > 0 or turn_output_tokens > 0:
                        self.total_input_tokens += turn_input_tokens
                        self.total_output_tokens += turn_output_tokens
                        self.total_thinking_tokens += turn_thinking_tokens
                        self.turn_count += 1
                        
                        logger.info(f"[{self.session_id}] Turn #{self.turn_count}: "
                                  f"in={turn_input_tokens}, out={turn_output_tokens}, "
                                  f"total={self.total_input_tokens + self.total_output_tokens}")
                        
                        # Send token usage to client
                        await self._send_to_client({
                            "type": "token_usage",
                            "turn": self.turn_count,
                            "input_tokens": turn_input_tokens,
                            "output_tokens": turn_output_tokens,
                            "total_input": self.total_input_tokens,
                            "total_output": self.total_output_tokens
                        })
                    
                    # Notify turn complete
                    await self._send_to_client({"type": "playback_finished"})
                    
                    # Clear audio queue on turn complete (handles interruptions)
                    while not self.audio_out_queue.empty():
                        try:
                            self.audio_out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if self.is_active:
                        logger.error(f"[{self.session_id}] Error in receive loop: {e}")
                        traceback.print_exc()
                    break
                    
        except Exception as e:
            logger.error(f"[{self.session_id}] Receive task error: {e}")
            traceback.print_exc()
    
    async def _send_audio_to_client(self):
        """Background task to forward audio from Gemini to client."""
        try:
            while self.is_active:
                try:
                    audio_data = await asyncio.wait_for(
                        self.audio_out_queue.get(),
                        timeout=0.1
                    )
                    
                    self.ai_speaking = True
                    self.audio_chunk_count += 1
                    
                    # Log first chunk
                    if self.audio_chunk_count == 1:
                        logger.info(f"[{self.session_id}] Sending audio chunks to client...")
                    
                    # Send audio to client as base64
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    await self._send_to_client({
                        "type": "audio_chunk",
                        "audio": audio_base64,
                        "encoding": "linear16",
                        "sample_rate": RECEIVE_SAMPLE_RATE
                    })
                    
                except asyncio.TimeoutError:
                    self.ai_speaking = False
                    continue
                except asyncio.CancelledError:
                    break
                    
        except Exception as e:
            logger.error(f"[{self.session_id}] Audio send task error: {e}")
    
    async def _send_to_client(self, message: dict):
        """Send a JSON message to the client WebSocket."""
        try:
            await self.client_ws.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"[{self.session_id}] Error sending to client: {e}")
    
    async def post_transcript(self) -> Dict:
        """
        Analyze the conversation transcript using Gemini.
        Extracts stakeholder insights based on the Maya Business Consultant discovery conversation.
        
        Returns:
            Analysis report with extracted verticals and sentiment
        """
        if not self.conversation_history:
            logger.info(f"[{self.session_id}] No conversation to analyze")
            return {"error": "No conversation history"}
        
        logger.info(f"[{self.session_id}] Analyzing conversation ({len(self.conversation_history)} messages)...")
        
        # Format conversation for analysis
        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in self.conversation_history
        ])
        
        # Get analysis prompt from centralized prompt module
        analysis_prompt = get_transcript_analysis_prompt(conversation_text)

        try:
            # Use Gemini for analysis
            client = genai.Client(api_key=self.settings.GEMINI_API_KEY)
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    {"role": "user", "parts": [{"text": analysis_prompt}]}
                ],
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert business analyst. Analyze conversations and extract structured insights. Always respond with valid JSON.",
                    temperature=0.4,
                )
            )
            
            analysis_text = response.text
            logger.info(f"[{self.session_id}] Post-transcript analysis complete")
            
            # Try to parse as JSON
            try:
                import re
                # Extract JSON from response (handle markdown code blocks)
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', analysis_text)
                if json_match:
                    analysis_text = json_match.group(1)
                analysis_result = json.loads(analysis_text)
            except json.JSONDecodeError:
                analysis_result = {"raw_analysis": analysis_text}
            
            # Add metadata
            analysis_result["session_id"] = self.session_id
            analysis_result["message_count"] = len(self.conversation_history)
            analysis_result["total_tokens"] = {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "thinking": self.total_thinking_tokens
            }
            
            logger.info(f"[{self.session_id}] Analysis result: {json.dumps(analysis_result, indent=2)}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"[{self.session_id}] Error in post_transcript analysis: {e}")
            traceback.print_exc()
            return {"error": str(e), "conversation_history": self.conversation_history}
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "max_turns": self.max_turns,
            "message_count": len(self.conversation_history),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_thinking_tokens": self.total_thinking_tokens,
            "grand_total_tokens": self.total_input_tokens + self.total_output_tokens + self.total_thinking_tokens,
            "audio_chunks_sent": self.audio_chunk_count
        }
    
    async def close(self):
        """Close the Gemini Live session and analyze transcript."""
        self.is_active = False
        
        # Signal stop to the session task
        self._stop_event.set()
        
        # Analyze conversation before closing
        if self.conversation_history:
            try:
                analysis = await self.post_transcript()
                # Send analysis to client
                await self._send_to_client({
                    "type": "transcript_analysis",
                    "analysis": analysis
                })
            except Exception as e:
                logger.error(f"[{self.session_id}] Error sending analysis: {e}")
        
        # Send final stats
        try:
            await self._send_to_client({
                "type": "session_stats",
                "stats": self.get_session_stats()
            })
        except Exception:
            pass
        
        # Cancel the session task
        if self.session_task:
            self.session_task.cancel()
            try:
                await self.session_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"[{self.session_id}] Session closed - Turns: {self.turn_count}, "
                   f"Tokens: {self.total_input_tokens + self.total_output_tokens}")


# Alias for backward compatibility
VoiceAgentSession = GeminiVoiceSession
