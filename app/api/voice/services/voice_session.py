"""
Voice Agent Session management.
Handles WebSocket connection to Deepgram Voice Agent API.
"""
import json
import base64
import asyncio
import time
from typing import Optional
import websockets
from fastapi import WebSocket
from loguru import logger
from app.config import Settings
from app.api.voice.services.voice_service import get_voice_agent_settings
from app.api.voice.services.prompt import get_transcript_analysis_prompt
from app.api.voice.models.gemini_client import GeminiClient
import traceback
from typing import List, Dict


# Deepgram Voice Agent V1 endpoint
VOICE_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"


class VoiceAgentSession:
    """Manages a session with Deepgram Voice Agent API."""
    
    def __init__(self, session_id: str, client_ws: WebSocket, settings: Settings):
        self.session_id = session_id
        self.client_ws = client_ws
        self.settings = settings
        self.agent_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_active = True
        self.start_time: Optional[float] = None
        self.audio_chunk_count = 0
        self.playback_started_sent = False
        self.conversation_history: List[Dict[str, str]] = []  # Store conversation for analysis
    
    async def connect_to_agent(self, context: dict = None) -> bool:
        """Connect to Deepgram Voice Agent API.
        
        Args:
            context: Optional dict with pre-provided user info (name, role, years_of_experience)
        """
        try:
            deepgram_api_key = self.settings.DEEPGRAM_API_KEY
            
            # Log API key status (masked for security)
            if not deepgram_api_key:
                logger.error(f"[{self.session_id}] DEEPGRAM_API_KEY is empty or not configured")
                return False
            else:
                masked_key = f"{deepgram_api_key[:8]}...{deepgram_api_key[-4:]}" if len(deepgram_api_key) > 12 else "***"
                logger.info(f"[{self.session_id}] DEEPGRAM_API_KEY found: {masked_key}")
            
            logger.info(f"[{self.session_id}] Connecting to Deepgram Voice Agent at {VOICE_AGENT_URL}...")
            
            # Connect with longer timeout for slow networks
            self.agent_ws = await asyncio.wait_for(
                websockets.connect(
                    VOICE_AGENT_URL,
                    additional_headers={"Authorization": f"Token {deepgram_api_key}"},
                    ping_interval=20,
                    ping_timeout=20,
                ),
                timeout=30.0  # 30 second timeout for connection
            )
            logger.info(f"[{self.session_id}] Connected to Deepgram Voice Agent")
            
            # Send Settings message to configure the agent (with user context for personalized prompt)
            logger.info(f"[{self.session_id}] Sending settings to Voice Agent...")
            settings_dict = await get_voice_agent_settings(self.settings, context=context)
            logger.debug(f"[{self.session_id}] Settings payload: {json.dumps(settings_dict, indent=2)[:500]}")
            await self.agent_ws.send(json.dumps(settings_dict))
            logger.info(f"[{self.session_id}] Settings sent to Voice Agent")
            
            return True
        except asyncio.TimeoutError:
            logger.error(f"[{self.session_id}] Connection to Deepgram timed out after 30s. Check network/firewall.")
            return False
        except websockets.exceptions.InvalidStatusCode as e:
            logger.error(f"[{self.session_id}] Deepgram rejected connection with status {e.status_code}")
            if e.status_code == 401:
                logger.error(f"[{self.session_id}] Invalid API key - check your DEEPGRAM_API_KEY")
            elif e.status_code == 403:
                logger.error(f"[{self.session_id}] API key doesn't have voice agent permissions")
            return False
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to connect to Voice Agent: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False

    
    async def forward_audio_to_agent(self, audio_data: bytes):
        """Forward audio from client to Deepgram Voice Agent."""
        if self.agent_ws:
            try:
                await self.agent_ws.send(audio_data)
            except Exception as e:
                logger.error(f"[{self.session_id}] Error sending audio to agent: {e}")
    
    async def receive_from_agent(self):
        """Receive messages/audio from Deepgram Voice Agent and forward to client."""
        try:
            while self.is_active and self.agent_ws:
                try:
                    msg = await asyncio.wait_for(self.agent_ws.recv(), timeout=0.1)
                    
                    if isinstance(msg, bytes):
                        # Audio data from TTS - forward to client
                        self.audio_chunk_count += 1
                        
                        # Send playback_started on first audio chunk
                        if not self.playback_started_sent:
                            self.playback_started_sent = True
                            if self.start_time:
                                latency_ms = int((time.perf_counter() - self.start_time) * 1000)
                                logger.info(f"[{self.session_id}] Agent | ⚡ First audio (latency: {latency_ms}ms)")
                            await self.client_ws.send_text(json.dumps({
                                "type": "playback_started"
                            }))
                        
                        # Log only first audio chunk
                        if self.audio_chunk_count == 1:
                            logger.info(f"[{self.session_id}] Agent | Receiving audio chunks...")
                        
                        audio_base64 = base64.b64encode(msg).decode('utf-8')
                        await self.client_ws.send_text(json.dumps({
                            "type": "audio_chunk",
                            "audio": audio_base64,
                            "encoding": "linear16",
                            "sample_rate": 24000
                        }))
                        
                    elif isinstance(msg, str):
                        # JSON message from agent
                        await self._handle_agent_message(msg)
                            
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"[{self.session_id}] Agent connection closed")
                    break
                    
        except Exception as e:
            logger.error(f"[{self.session_id}] Error receiving from agent: {e}")
    
    async def _handle_agent_message(self, msg: str):
        """Handle JSON message from Deepgram Voice Agent."""
        data = json.loads(msg)
        msg_type = data.get("type")
        
        if msg_type == "Welcome":
            logger.info(f"[{self.session_id}] Agent | Welcome received")
            await self.client_ws.send_text(json.dumps({
                "type": "agent_ready"
            }))
            
        elif msg_type == "SettingsApplied":
            logger.info(f"[{self.session_id}] Agent | Settings applied")
            await self.client_ws.send_text(json.dumps({
                "type": "settings_applied"
            }))
            
        elif msg_type == "UserStartedSpeaking":
            self.start_time = time.perf_counter()
            logger.info(f"[{self.session_id}] Agent | User started speaking")
            await self.client_ws.send_text(json.dumps({
                "type": "speech_started"
            }))
            
        elif msg_type == "AgentThinking":
            logger.info(f"[{self.session_id}] Agent | Thinking...")
            await self.client_ws.send_text(json.dumps({
                "type": "thinking"
            }))
            
        elif msg_type == "AgentStartedSpeaking":
            if self.start_time:
                latency_ms = int((time.perf_counter() - self.start_time) * 1000)
                logger.info(f"[{self.session_id}] Agent | ⚡ Started speaking (latency: {latency_ms}ms)")
            await self.client_ws.send_text(json.dumps({
                "type": "playback_started"
            }))
            
        elif msg_type == "AgentAudioDone":
            logger.info(f"[{self.session_id}] Agent | Audio done (total chunks: {self.audio_chunk_count})")
            # Reset for next response
            self.audio_chunk_count = 0
            self.playback_started_sent = False
            await self.client_ws.send_text(json.dumps({
                "type": "playback_finished"
            }))
            
        elif msg_type == "ConversationText":
            # Transcript or response text
            role = data.get("role")
            content = data.get("content", "")
            
            if role == "user":
                logger.info(f"[{self.session_id}] Agent | User: {content}")
                # Store user message in conversation history
                self.conversation_history.append({"role": "user", "content": content})
                await self.client_ws.send_text(json.dumps({
                    "type": "transcript",
                    "text": content
                }))
            elif role == "assistant":
                logger.info(f"[{self.session_id}] Agent | Assistant: {content}")
                # Store assistant message in conversation history
                self.conversation_history.append({"role": "assistant", "content": content})
                await self.client_ws.send_text(json.dumps({
                    "type": "response",
                    "text": content
                }))
        
        elif msg_type == "Error":
            error_msg = data.get("message", "Unknown error")
            logger.error(f"[{self.session_id}] Agent | Error: {error_msg}")
            await self.client_ws.send_text(json.dumps({
                "type": "error",
                "message": error_msg
            }))
            
        else:
            logger.debug(f"[{self.session_id}] Agent | {msg_type}: {data}")
    
    async def post_transcript(self) -> Dict:
        """
        Analyze the conversation transcript using DeepSeek LLM.
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
            client = GeminiClient(api_key=self.settings.GEMINI_API_KEY)
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert business analyst. Analyze conversations and extract structured insights. Always respond with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.4,
            )
            
            analysis_text = response.choices[0].message.content
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
            analysis_result["token_usage"] = client.get_usage(response)
            
            logger.info(f"[{self.session_id}] Analysis result: {json.dumps(analysis_result, indent=2)}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"[{self.session_id}] Error in post_transcript analysis: {e}")
            traceback.print_exc()
            return {"error": str(e), "conversation_history": self.conversation_history}
    
    async def close(self):
        """Close the Voice Agent connection."""
        self.is_active = False
        
        if self.agent_ws:
            try:
                await self.agent_ws.close()
            except Exception:
                pass
        logger.info(f"[{self.session_id}] Session closed")

