import React, { useState, useEffect, useRef, useCallback } from 'react';
import MicButton from './components/MicButton';
import ResponseBubble from './components/ResponseBubble';
import ChatHistory from './components/ChatHistory';
import { v4 as uuidv4 } from 'uuid';

// Deepgram Voice Agent API backend
const WS_URL = "ws://localhost:8000";

// Audio configuration
const INPUT_SAMPLE_RATE = 16000;  // For recording
const OUTPUT_SAMPLE_RATE = 24000; // For playback
const AUDIO_CHANNELS = 1;

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [streamingResponse, setStreamingResponse] = useState('');
  const [response, setResponse] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isAssistantSpeaking, setIsAssistantSpeaking] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('initializing');
  const [transcriptAnalysis, setTranscriptAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const websocketRef = useRef(null);
  const audioContextRef = useRef(null);
  const playbackContextRef = useRef(null);
  const audioStreamRef = useRef(null);
  const workletNodeRef = useRef(null);
  const sessionIdRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const nextPlayTimeRef = useRef(0);
  const currentTranscriptRef = useRef('');
  const currentResponseRef = useRef('');

  useEffect(() => {
    sessionIdRef.current = uuidv4();
    setConnectionStatus('ready');
    console.log('Client session ID created:', sessionIdRef.current);

    // Initialize AudioContext for streaming playback
    playbackContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: OUTPUT_SAMPLE_RATE
    });

    return () => {
      cleanup();
      if (playbackContextRef.current) {
        playbackContextRef.current.close();
      }
    };
  }, []);

  // Process and play audio queue
  const playNextAudioChunk = useCallback(async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) {
      return;
    }

    isPlayingRef.current = true;

    while (audioQueueRef.current.length > 0) {
      const audioBase64 = audioQueueRef.current.shift();

      try {
        // Decode base64 to ArrayBuffer
        const binaryString = atob(audioBase64);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }

        // Convert Int16 PCM to Float32 for Web Audio API
        const pcmData = new Int16Array(bytes.buffer);
        const floatData = new Float32Array(pcmData.length);
        for (let i = 0; i < pcmData.length; i++) {
          floatData[i] = pcmData[i] / 32768.0;
        }

        // Create audio buffer
        const audioBuffer = playbackContextRef.current.createBuffer(
          AUDIO_CHANNELS,
          floatData.length,
          OUTPUT_SAMPLE_RATE
        );
        audioBuffer.getChannelData(0).set(floatData);

        // Schedule playback
        const source = playbackContextRef.current.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(playbackContextRef.current.destination);

        const currentTime = playbackContextRef.current.currentTime;
        const startTime = Math.max(currentTime, nextPlayTimeRef.current);
        source.start(startTime);
        nextPlayTimeRef.current = startTime + audioBuffer.duration;

        console.log(`Playing audio: ${floatData.length} samples, ${audioBuffer.duration.toFixed(2)}s`);
      } catch (error) {
        console.error('Error playing audio chunk:', error);
      }
    }

    isPlayingRef.current = false;
  }, []);

  const initializeWebSocket = () => {
    if (!sessionIdRef.current) {
      console.error("Session ID not initialized.");
      setConnectionStatus('error');
      return;
    }

    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    const wsUrl = `${WS_URL}/api/ws/voice/${sessionIdRef.current}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'session_started':
          console.log('Voice Agent session started');
          break;

        case 'agent_ready':
          console.log('Voice Agent ready');
          break;

        case 'settings_applied':
          console.log('Voice Agent settings applied');
          break;

        case 'speech_started':
          console.log('User speech detected');
          setIsProcessing(true);
          break;

        case 'transcript':
          console.log('Transcript:', data.text);
          setTranscript(data.text);
          currentTranscriptRef.current = data.text;
          break;

        case 'thinking':
          console.log('Agent thinking...');
          setIsProcessing(true);
          break;

        case 'response':
          console.log('Response:', data.text);
          setStreamingResponse(data.text);
          currentResponseRef.current = data.text;
          break;

        case 'playback_started':
          console.log('Agent speaking started');
          setIsAssistantSpeaking(true);
          setIsProcessing(false);
          break;

        case 'playback_finished':
          console.log('Agent speaking finished');
          setIsAssistantSpeaking(false);
          finalizeResponse();
          break;

        case 'audio_chunk':
          audioQueueRef.current.push(data.audio);
          if (playbackContextRef.current.state === 'suspended') {
            playbackContextRef.current.resume();
          }
          playNextAudioChunk();
          break;

        case 'error':
          console.error('Server error:', data.message);
          setIsProcessing(false);
          setIsAnalyzing(false);
          setConnectionStatus('error');
          break;

        case 'transcript_analysis':
          console.log('Transcript analysis received:', data.analysis);
          setTranscriptAnalysis(data.analysis);
          setIsAnalyzing(false);
          break;

        default:
          console.log('Unknown message type:', data.type, data);
          break;
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      if (connectionStatus !== 'error') {
        setConnectionStatus('ready');
      }
    };

    websocketRef.current = ws;
  };

  const finalizeResponse = () => {
    const userMsg = currentTranscriptRef.current;
    const assistantMsg = currentResponseRef.current;

    if (userMsg || assistantMsg) {
      const newChatItem = {
        id: uuidv4(),
        userMessage: userMsg,
        assistantMessage: assistantMsg,
        timestamp: new Date().toISOString()
      };

      setChatHistory(history => [...history, newChatItem]);
      setResponse({
        transcription: userMsg,
        text: assistantMsg
      });
    }

    setIsProcessing(false);
    setTranscript('');
    setStreamingResponse('');
    currentTranscriptRef.current = '';
    currentResponseRef.current = '';
  };

  // Convert ArrayBuffer to base64
  const arrayBufferToBase64 = (buffer) => {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
  };

  const handleRecordingStart = async () => {
    try {
      if (connectionStatus === 'error') {
        alert('Connection error. Please reconnect.');
        return;
      }

      setIsRecording(true);
      setResponse(null);
      setTranscript('');
      setStreamingResponse('');
      audioQueueRef.current = [];
      nextPlayTimeRef.current = 0;

      sessionIdRef.current = uuidv4();
      initializeWebSocket();

      await new Promise((resolve, reject) => {
        const checkConnection = () => {
          if (websocketRef.current?.readyState === WebSocket.OPEN) {
            resolve();
          } else if (websocketRef.current?.readyState === WebSocket.CLOSING || websocketRef.current?.readyState === WebSocket.CLOSED) {
            reject(new Error('WebSocket connection failed to open'));
          } else {
            setTimeout(checkConnection, 100);
          }
        };
        checkConnection();
      });

      // Start voice agent session
      websocketRef.current.send(JSON.stringify({ type: 'start_session' }));
      await new Promise(resolve => setTimeout(resolve, 500));

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: INPUT_SAMPLE_RATE,
          channelCount: 1,
        }
      });

      audioStreamRef.current = stream;

      // Create AudioContext for capturing raw PCM
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: INPUT_SAMPLE_RATE
      });

      const source = audioContextRef.current.createMediaStreamSource(stream);

      // Try AudioWorklet first, fall back to ScriptProcessor
      try {
        await audioContextRef.current.audioWorklet.addModule('/pcm-processor.js');

        const workletNode = new AudioWorkletNode(audioContextRef.current, 'pcm-processor');

        workletNode.port.onmessage = (event) => {
          if (event.data.type === 'audio' && websocketRef.current?.readyState === WebSocket.OPEN) {
            const base64Data = arrayBufferToBase64(event.data.buffer);
            websocketRef.current.send(JSON.stringify({
              type: 'audio_chunk',
              audio_data: base64Data
            }));
          }
        };

        source.connect(workletNode);
        workletNode.connect(audioContextRef.current.destination);
        workletNodeRef.current = workletNode;

        console.log('Recording started with AudioWorklet at 16kHz');
      } catch (workletError) {
        console.warn('AudioWorklet not available, using ScriptProcessor:', workletError);

        // Fallback to ScriptProcessor
        const bufferSize = 4096;
        const processorNode = audioContextRef.current.createScriptProcessor(bufferSize, 1, 1);

        processorNode.onaudioprocess = (e) => {
          if (websocketRef.current?.readyState === WebSocket.OPEN) {
            const inputData = e.inputBuffer.getChannelData(0);
            const int16Array = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
              const s = Math.max(-1, Math.min(1, inputData[i]));
              int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            const base64Data = arrayBufferToBase64(int16Array.buffer);
            websocketRef.current.send(JSON.stringify({
              type: 'audio_chunk',
              audio_data: base64Data
            }));
          }
        };

        source.connect(processorNode);
        // Connect through silent gain to make it work
        const silentGain = audioContextRef.current.createGain();
        silentGain.gain.value = 0;
        processorNode.connect(silentGain);
        silentGain.connect(audioContextRef.current.destination);

        workletNodeRef.current = processorNode;
        console.log('Recording started with ScriptProcessor at 16kHz');
      }

    } catch (error) {
      console.error('Failed to start recording:', error);
      alert('Failed to start recording: ' + error.message);
      setIsRecording(false);
      setConnectionStatus('error');
    }
  };

  const handleRecordingStop = async () => {
    setIsRecording(false);

    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach(track => track.stop());
      audioStreamRef.current = null;
    }

    // Send end_session to trigger transcript analysis
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      setIsAnalyzing(true);
      websocketRef.current.send(JSON.stringify({ type: 'end_session' }));
      console.log('Recording stopped, requesting transcript analysis...');
    } else {
      console.log('Recording stopped, WebSocket not open');
    }
  };

  const handleReconnect = () => {
    cleanup();
    sessionIdRef.current = uuidv4();
    setConnectionStatus('ready');
    setResponse(null);
    setTranscript('');
    setStreamingResponse('');
    setIsProcessing(false);
    console.log('Reconnected with new session ID:', sessionIdRef.current);
  };

  const cleanup = () => {
    if (websocketRef.current) {
      if (websocketRef.current.readyState === WebSocket.OPEN) {
        websocketRef.current.send(JSON.stringify({ type: 'end_session' }));
      }
      websocketRef.current.close();
      websocketRef.current = null;
    }

    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach(track => track.stop());
      audioStreamRef.current = null;
    }

    audioQueueRef.current = [];
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'ready': return 'Ready';
      case 'connected': return 'Connected';
      case 'error': return 'Connection Error';
      default: return 'Initializing...';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-blue-900 flex flex-col items-center justify-center p-4 font-sans">
      <div className="max-w-2xl w-full space-y-6">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-2">Voice Assistant</h1>
          <p className="text-white/80 mb-2">Low-latency voice agent powered by Deepgram</p>
          <div className="flex items-center justify-center space-x-2 text-sm text-white/60">
            <span>Status: {getConnectionStatusText()}</span>
            {isRecording && <span className="text-red-400 animate-pulse">🎙️ Recording</span>}
            {isProcessing && <span className="text-yellow-400 animate-pulse">🧠 Processing</span>}
            {isAssistantSpeaking && <span className="text-orange-400 animate-pulse">🔊 Speaking</span>}
          </div>
        </div>

        <ChatHistory chatHistory={chatHistory} />

        {transcript && (
          <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-4 text-white/90">
            <div className="flex items-start space-x-2">
              <span className="text-blue-400 text-sm">You:</span>
              <span>{transcript}</span>
            </div>
          </div>
        )}

        {streamingResponse && (
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-4">
            <div className="flex items-start space-x-2">
              <span className="text-purple-400 text-sm">
                {isAssistantSpeaking ? '🔊 Assistant:' : 'Assistant:'}
              </span>
              <span className="text-white">{streamingResponse}</span>
              {isAssistantSpeaking && <span className="animate-pulse text-purple-400">▋</span>}
            </div>
          </div>
        )}

        {response && !streamingResponse && !isAssistantSpeaking && <ResponseBubble response={response} />}

        {isProcessing && !streamingResponse && !transcript && (
          <div className="flex justify-center">
            <div className="bg-white/10 backdrop-blur-sm rounded-full px-6 py-3 text-white">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                <span>🧠 Thinking...</span>
              </div>
            </div>
          </div>
        )}

        <div className="flex justify-center pt-4">
          <MicButton
            isRecording={isRecording}
            onRecordingStart={handleRecordingStart}
            onRecordingStop={handleRecordingStop}
            disabled={isProcessing || isAssistantSpeaking || (connectionStatus === 'error' && !isRecording)}
          />
        </div>

        {connectionStatus === 'error' && (
          <div className="text-center">
            <button
              onClick={handleReconnect}
              className="bg-red-500 hover:bg-red-600 text-white font-semibold px-4 py-2 rounded-lg transition-colors"
            >
              Reconnect
            </button>
          </div>
        )}

        {isAnalyzing && (
          <div className="flex justify-center">
            <div className="bg-white/10 backdrop-blur-sm rounded-full px-6 py-3 text-white">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-purple-400"></div>
                <span>📊 Analyzing conversation...</span>
              </div>
            </div>
          </div>
        )}

        {transcriptAnalysis && !isAnalyzing && (
          <div className="bg-gradient-to-r from-purple-500/20 to-blue-500/20 backdrop-blur-sm rounded-2xl p-6 border border-purple-500/30">
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-xl font-semibold text-white flex items-center">
                <span className="mr-2">📊</span> Conversation Analysis
              </h3>
              <button
                onClick={() => setTranscriptAnalysis(null)}
                className="text-white/60 hover:text-white transition-colors text-xl"
              >
                ✕
              </button>
            </div>

            {transcriptAnalysis.error ? (
              <p className="text-red-400">{transcriptAnalysis.error}</p>
            ) : (
              <div className="space-y-4 text-white/90">
                {/* Summary Section */}
                {transcriptAnalysis.summary && (
                  <div className="bg-white/5 rounded-xl p-4">
                    <h4 className="text-purple-300 font-medium mb-2">📝 Summary</h4>
                    <p className="text-white/80 leading-relaxed">{transcriptAnalysis.summary}</p>
                  </div>
                )}

                {/* Industry & Position */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {transcriptAnalysis.industry && (
                    <div className="bg-white/5 rounded-lg p-3">
                      <span className="text-purple-400 text-xs block">Industry</span>
                      <span className="font-medium">{transcriptAnalysis.industry}</span>
                    </div>
                  )}
                  {transcriptAnalysis.position && (
                    <div className="bg-white/5 rounded-lg p-3">
                      <span className="text-purple-400 text-xs block">Position</span>
                      <span className="font-medium">{transcriptAnalysis.position}</span>
                    </div>
                  )}
                  {transcriptAnalysis.tenure && (
                    <div className="bg-white/5 rounded-lg p-3">
                      <span className="text-purple-400 text-xs block">Tenure</span>
                      <span className="font-medium">{transcriptAnalysis.tenure}</span>
                    </div>
                  )}
                  {transcriptAnalysis.sentiment && (
                    <div className="bg-white/5 rounded-lg p-3">
                      <span className="text-purple-400 text-xs block">Sentiment</span>
                      <span className={`font-medium capitalize ${transcriptAnalysis.sentiment === 'positive' ? 'text-green-400' :
                          transcriptAnalysis.sentiment === 'negative' ? 'text-red-400' :
                            'text-yellow-400'
                        }`}>{transcriptAnalysis.sentiment}</span>
                    </div>
                  )}
                </div>

                {/* Key Insights */}
                {transcriptAnalysis.key_insights && transcriptAnalysis.key_insights.length > 0 && (
                  <div className="bg-white/5 rounded-xl p-4">
                    <h4 className="text-green-300 font-medium mb-2">💡 Key Insights</h4>
                    <ul className="space-y-2">
                      {transcriptAnalysis.key_insights.map((insight, i) => (
                        <li key={i} className="flex items-start text-white/80">
                          <span className="text-green-400 mr-2">•</span>
                          <span>{insight}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Opportunities */}
                {transcriptAnalysis.opportunities && transcriptAnalysis.opportunities.length > 0 && (
                  <div className="bg-white/5 rounded-xl p-4">
                    <h4 className="text-blue-300 font-medium mb-2">🚀 Opportunities</h4>
                    <ul className="space-y-2">
                      {transcriptAnalysis.opportunities.map((opp, i) => (
                        <li key={i} className="flex items-start text-white/80">
                          <span className="text-blue-400 mr-2">•</span>
                          <span>{opp}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Red Flags */}
                {transcriptAnalysis.red_flags && transcriptAnalysis.red_flags.length > 0 && (
                  <div className="bg-red-500/10 rounded-xl p-4 border border-red-500/20">
                    <h4 className="text-red-300 font-medium mb-2">⚠️ Red Flags</h4>
                    <ul className="space-y-2">
                      {transcriptAnalysis.red_flags.map((flag, i) => (
                        <li key={i} className="flex items-start text-white/80">
                          <span className="text-red-400 mr-2">•</span>
                          <span>{flag}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Session Info */}
                <div className="flex items-center justify-between text-xs text-white/40 pt-2 border-t border-white/10">
                  <span>Session: {transcriptAnalysis.session_id?.slice(0, 8)}...</span>
                  <span>{transcriptAnalysis.message_count} messages</span>
                  {transcriptAnalysis.token_usage && (
                    <span>{transcriptAnalysis.token_usage.total_tokens} tokens</span>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;