"""
Voice Agent Session management.
Handles WebSocket connection to xAI Grok Voice Agent API.
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
from tools.functions import get_current_weather, search_articles, retrieve_documents
import traceback


# xAI Grok Voice Agent WebSocket endpoint
VOICE_AGENT_URL = "wss://api.x.ai/v1/realtime"

# Default collection for voice document retrieval
DEFAULT_COLLECTION = "tool_calling_dev"


class VoiceAgentSession:
    """Manages a session with xAI Grok Voice Agent API."""
    
    def __init__(self, session_id: str, client_ws: WebSocket, settings: Settings):
        self.session_id = session_id
        self.client_ws = client_ws
        self.settings = settings
        self.agent_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_active = True
        self.start_time: Optional[float] = None
        self.audio_chunk_count = 0
        self.playback_started_sent = False
        self.pending_function_calls = {}  # Track pending function calls by call_id
    
    async def connect_to_agent(self) -> bool:
        """Connect to xAI Grok Voice Agent API."""
        try:
            xai_api_key = self.settings.XAI_API_KEY
            if not xai_api_key:
                logger.error(f"[{self.session_id}] XAI_API_KEY not configured")
                return False
            
            logger.info(f"[{self.session_id}] Connecting to Grok Voice Agent...")
            
            # Connect with Bearer token authentication
            self.agent_ws = await websockets.connect(
                VOICE_AGENT_URL,
                additional_headers={
                    "Authorization": f"Bearer {xai_api_key}",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
            logger.info(f"[{self.session_id}] ✓ Connected to Grok Voice Agent")
            
            # Send session.update message to configure the agent
            settings_dict = await get_voice_agent_settings(self.settings)
            await self.agent_ws.send(json.dumps(settings_dict))
            logger.info(f"[{self.session_id}] ✓ Sent session.update to Voice Agent")
            
            return True
        except Exception as e:
            logger.error(f"[{self.session_id}] ✗ Failed to connect to Voice Agent: {e}")
            traceback.print_exc()
            return False

    
    async def forward_audio_to_agent(self, audio_data: bytes):
        """Forward audio from client to Grok Voice Agent."""
        if self.agent_ws:
            try:
                # Grok expects audio in input_audio_buffer.append format
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                message = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_base64
                }
                await self.agent_ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"[{self.session_id}] Error sending audio to agent: {e}")
    
    async def commit_audio_buffer(self):
        """Commit the audio buffer to trigger processing."""
        if self.agent_ws:
            try:
                message = {"type": "input_audio_buffer.commit"}
                await self.agent_ws.send(json.dumps(message))
                logger.debug(f"[{self.session_id}] Audio buffer committed")
            except Exception as e:
                logger.error(f"[{self.session_id}] Error committing audio buffer: {e}")
    
    async def receive_from_agent(self):
        """Receive messages/audio from Grok Voice Agent and forward to client."""
        try:
            while self.is_active and self.agent_ws:
                try:
                    msg = await asyncio.wait_for(self.agent_ws.recv(), timeout=0.1)
                    
                    if isinstance(msg, bytes):
                        # Binary audio data - forward to client
                        self.audio_chunk_count += 1
                        
                        if not self.playback_started_sent:
                            self.playback_started_sent = True
                            if self.start_time:
                                latency_ms = int((time.perf_counter() - self.start_time) * 1000)
                                logger.info(f"[{self.session_id}] Agent | ⚡ First audio (latency: {latency_ms}ms)")
                            await self.client_ws.send_text(json.dumps({
                                "type": "playback_started"
                            }))
                        
                        if self.audio_chunk_count == 1:
                            logger.info(f"[{self.session_id}] Agent | Receiving audio chunks...")
                        
                        audio_base64 = base64.b64encode(msg).decode('utf-8')
                        await self.client_ws.send_text(json.dumps({
                            "type": "audio_chunk",
                            "audio": audio_base64,
                            "encoding": "pcm16",
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
            traceback.print_exc()
    
    async def _execute_function(self, function_name: str, arguments: dict) -> str:
        """
        Execute a function and return the result as a JSON string.
        Logs each step like the chat module for debugging.
        """
        start_time = time.perf_counter()
        
        logger.info(f"[VOICE_FUNCTION] [{self.session_id}] ▶ Starting execution: {function_name}")
        logger.info(f"[VOICE_FUNCTION] [{self.session_id}]   Arguments: {json.dumps(arguments)}")
        
        try:
            if function_name == "get_current_weather":
                location = arguments.get("location", "")
                unit = arguments.get("unit", "celsius")
                
                logger.info(f"[VOICE_FUNCTION] [{self.session_id}]   Weather lookup: location={location}, unit={unit}")
                
                result = get_current_weather(location=location, unit=unit)
                
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(f"[VOICE_FUNCTION] [{self.session_id}] ✓ Weather result: {result.get('description', 'N/A')}, temp={result.get('temperature', 'N/A')}° | took {elapsed_ms}ms")
                return json.dumps(result)
            
            elif function_name == "search_articles":
                query = arguments.get("query", "")
                max_results = arguments.get("max_results", 2)
                
                logger.info(f"[VOICE_FUNCTION] [{self.session_id}]   Web search: query='{query}', max_results={max_results}")
                
                result = search_articles(query=query, max_results=max_results)
                
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                content_preview = str(result)[:150] if result else "No results"
                logger.info(f"[VOICE_FUNCTION] [{self.session_id}] ✓ Search result: {content_preview}... | took {elapsed_ms}ms")
                return json.dumps(result)
            
            elif function_name == "retrieve_documents":
                query = arguments.get("query", "")
                file_ids = arguments.get("file_ids", None)
                
                logger.info(f"[VOICE_FUNCTION] [{self.session_id}]   Document retrieval: query='{query}', collection={DEFAULT_COLLECTION}")
                
                # Use retrieve_documents from tools/functions.py
                documents, token_usage = retrieve_documents(
                    query=query,
                    collection_name=DEFAULT_COLLECTION,
                    file_ids=file_ids,
                    top_k=5  # Fewer docs for voice to keep response concise
                )
                
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(f"[VOICE_FUNCTION] [{self.session_id}] ✓ Retrieved {len(documents)} documents | tokens={token_usage} | took {elapsed_ms}ms")
                
                if documents:
                    # Format documents for voice response
                    doc_summaries = []
                    for i, doc in enumerate(documents[:3]):  # Top 3 for voice
                        content_preview = doc.page_content[:200].replace('\n', ' ')
                        doc_summaries.append({
                            "index": i + 1,
                            "file": doc.metadata.get("file_name", "Unknown"),
                            "content": content_preview,
                            "score": round(doc.metadata.get("score", 0), 3)
                        })
                        logger.debug(f"[VOICE_FUNCTION] [{self.session_id}]   Doc {i+1}: {doc.metadata.get('file_name', 'Unknown')} (score={doc.metadata.get('score', 0):.3f})")
                    
                    result = {
                        "found": True,
                        "count": len(documents),
                        "documents": doc_summaries,
                        "message": f"Found {len(documents)} relevant documents"
                    }
                else:
                    result = {
                        "found": False,
                        "count": 0,
                        "message": "No relevant documents found for this query"
                    }
                
                return json.dumps(result)
            
            else:
                logger.warning(f"[VOICE_FUNCTION] [{self.session_id}] ✗ Unknown function: {function_name}")
                return json.dumps({"error": f"Unknown function: {function_name}"})
                
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"[VOICE_FUNCTION] [{self.session_id}] ✗ Error in {function_name} after {elapsed_ms}ms: {e}")
            traceback.print_exc()
            return json.dumps({"error": str(e)})
    
    async def _handle_agent_message(self, msg: str):
        """Handle JSON message from Grok Voice Agent."""
        data = json.loads(msg)
        msg_type = data.get("type")
        
        # Log all events for debugging
        logger.debug(f"[{self.session_id}] Agent | Event: {msg_type}")
        
        if msg_type == "session.created":
            logger.info(f"[{self.session_id}] Agent | ✓ Session created")
            await self.client_ws.send_text(json.dumps({
                "type": "agent_ready"
            }))
            
        elif msg_type == "session.updated":
            logger.info(f"[{self.session_id}] Agent | ✓ Session updated/configured")
            await self.client_ws.send_text(json.dumps({
                "type": "settings_applied"
            }))
            
        elif msg_type == "input_audio_buffer.speech_started":
            self.start_time = time.perf_counter()
            logger.info(f"[{self.session_id}] Agent | 🎤 User started speaking")
            await self.client_ws.send_text(json.dumps({
                "type": "speech_started"
            }))
            
        elif msg_type == "input_audio_buffer.speech_stopped":
            logger.info(f"[{self.session_id}] Agent | 🎤 User stopped speaking")
            await self.client_ws.send_text(json.dumps({
                "type": "speech_stopped"
            }))
            
        elif msg_type == "input_audio_buffer.committed":
            logger.info(f"[{self.session_id}] Agent | ✓ Audio buffer committed")
            
        elif msg_type == "conversation.item.created":
            item = data.get("item", {})
            role = item.get("role", "")
            item_type = item.get("type", "")
            logger.info(f"[{self.session_id}] Agent | 💬 Conversation item created (role={role}, type={item_type})")
            
            # When user's message is created, trigger response generation
            if role == "user" or item_type == "message":
                logger.info(f"[{self.session_id}] Agent | 🚀 Triggering response.create for user message")
                await self.agent_ws.send(json.dumps({"type": "response.create"}))
            
        elif msg_type == "response.created":
            logger.info(f"[{self.session_id}] Agent | 🤔 Response generation started")
            await self.client_ws.send_text(json.dumps({
                "type": "thinking"
            }))
            
        elif msg_type == "response.text.delta":
            # Streaming text response
            delta = data.get("delta", "")
            logger.debug(f"[{self.session_id}] Agent | Text delta: {delta}")
            
        elif msg_type == "response.text.done":
            text = data.get("text", "")
            logger.info(f"[{self.session_id}] Agent | 📝 Response text: {text}")
            await self.client_ws.send_text(json.dumps({
                "type": "response",
                "text": text
            }))
            
        elif msg_type == "response.audio_transcript.delta":
            # STT transcript of user's speech
            delta = data.get("delta", "")
            logger.debug(f"[{self.session_id}] Agent | Transcript delta: {delta}")
            
        elif msg_type == "response.audio_transcript.done":
            transcript = data.get("transcript", "")
            logger.info(f"[{self.session_id}] Agent | 📝 User transcript: {transcript}")
            await self.client_ws.send_text(json.dumps({
                "type": "transcript",
                "text": transcript
            }))
            
        elif msg_type == "response.audio.delta":
            # Audio chunk from TTS
            audio_base64 = data.get("delta", "")
            if audio_base64:
                self.audio_chunk_count += 1
                
                if not self.playback_started_sent:
                    self.playback_started_sent = True
                    if self.start_time:
                        latency_ms = int((time.perf_counter() - self.start_time) * 1000)
                        logger.info(f"[{self.session_id}] Agent | ⚡ First audio chunk (latency: {latency_ms}ms)")
                    await self.client_ws.send_text(json.dumps({
                        "type": "playback_started"
                    }))
                
                if self.audio_chunk_count == 1:
                    logger.info(f"[{self.session_id}] Agent | 🔊 Receiving audio chunks...")
                
                await self.client_ws.send_text(json.dumps({
                    "type": "audio_chunk",
                    "audio": audio_base64,
                    "encoding": "pcm16",
                    "sample_rate": 24000
                }))
            
        elif msg_type == "response.audio.done":
            logger.info(f"[{self.session_id}] Agent | 🔊 Audio complete (total chunks: {self.audio_chunk_count})")
            self.audio_chunk_count = 0
            self.playback_started_sent = False
            await self.client_ws.send_text(json.dumps({
                "type": "playback_finished"
            }))
            
        elif msg_type == "response.function_call_arguments.delta":
            # Function call arguments streaming
            call_id = data.get("call_id", "")
            delta = data.get("delta", "")
            logger.debug(f"[{self.session_id}] Agent | Function args delta for {call_id}: {delta}")
            
        elif msg_type == "response.function_call_arguments.done":
            # Function call ready to execute
            call_id = data.get("call_id", "")
            func_name = data.get("name", "")
            func_args_str = data.get("arguments", "{}")
            
            logger.info(f"[{self.session_id}] Agent | 🔧 Function call request: {func_name}")
            logger.info(f"[{self.session_id}] Agent |    call_id={call_id}")
            logger.info(f"[{self.session_id}] Agent |    arguments={func_args_str}")
            
            # Parse arguments
            try:
                func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
            except json.JSONDecodeError:
                func_args = {}
            
            # Execute the function
            result = await self._execute_function(func_name, func_args)
            
            logger.info(f"[{self.session_id}] Agent | 🔧 Function result: {result}")
            
            # Send function_call_output back to Grok
            output_message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result
                }
            }
            
            await self.agent_ws.send(json.dumps(output_message))
            logger.info(f"[{self.session_id}] Agent | ✓ Sent function output for {func_name}")
            
            # Trigger response generation after function output
            await self.agent_ws.send(json.dumps({"type": "response.create"}))
            logger.info(f"[{self.session_id}] Agent | ✓ Triggered response.create")
            
            # Notify client
            await self.client_ws.send_text(json.dumps({
                "type": "function_executed",
                "name": func_name,
                "result": result
            }))
            
        elif msg_type == "response.done":
            response = data.get("response", {})
            status = response.get("status", "unknown")
            logger.info(f"[{self.session_id}] Agent | ✓ Response complete (status={status})")
            
        elif msg_type == "error":
            error = data.get("error", {})
            error_msg = error.get("message", "Unknown error")
            error_type = error.get("type", "unknown")
            logger.error(f"[{self.session_id}] Agent | ✗ Error ({error_type}): {error_msg}")
            await self.client_ws.send_text(json.dumps({
                "type": "error",
                "message": error_msg
            }))
            
        elif msg_type == "rate_limits.updated":
            logger.debug(f"[{self.session_id}] Agent | Rate limits updated")
        
        elif msg_type == "ping":
            # Respond to ping with pong to keep connection alive
            event_id = data.get("event_id", "")
            logger.debug(f"[{self.session_id}] Agent | Ping received, sending pong")
            await self.agent_ws.send(json.dumps({
                "type": "pong",
                "event_id": event_id
            }))
        
        elif msg_type == "conversation.created":
            conversation_id = data.get("conversation", {}).get("id", "")
            logger.info(f"[{self.session_id}] Agent | ✓ Conversation created (id={conversation_id[:8]}...)")
            
        else:
            logger.debug(f"[{self.session_id}] Agent | Unhandled event: {msg_type}")
            logger.debug(f"[{self.session_id}] Agent |   Data: {json.dumps(data)[:200]}")
    
    async def close(self):
        """Close the Voice Agent connection."""
        self.is_active = False
        if self.agent_ws:
            try:
                await self.agent_ws.close()
                logger.info(f"[{self.session_id}] ✓ Agent WebSocket closed")
            except Exception:
                pass
        logger.info(f"[{self.session_id}] Session closed")
