import React, { useState, useEffect, useRef } from 'react';
import { 
  Mic, 
  MicOff, 
  Send, 
  Sparkles, 
  Activity, 
  ShieldCheck, 
  Layers, 
  CheckCircle2, 
  AlertTriangle, 
  BarChart3, 
  BookOpen, 
  Clock, 
  Upload, 
  Zap,
  Globe,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Terminal,
  Search,
  FileText,
  CheckCircle,
  XCircle,
  Cpu,
  Database,
  Sliders,
  Volume2,
  Copy,
  Check
} from 'lucide-react';
import './App.css';

interface RetrievedDoc {
  chunk_id: string;
  passage_id: string;
  text: string;
  raw_text: string;
  lang: string;
  rrf_score: number;
  dense_score: number;
}

interface LatencyBreakdown {
  stt_ms: number;
  input_guardrail_ms: number;
  embedding_ms: number;
  dense_search_ms: number;
  lexical_search_ms: number;
  fusion_ms: number;
  total_retrieval_ms: number;
  llm_generation_ms: number;
  output_guardrail_ms: number;
  total_end_to_end_ms: number;
}

interface PipelineTraceStep {
  step_num: number;
  step_id: string;
  step_name: string;
  time_ms: number;
  status: string;
  details: Record<string, any>;
}

interface PipelineResponse {
  query: string;
  detected_lang: string;
  answer: string;
  retrieved_documents: RetrievedDoc[];
  input_guard: {
    is_safe: boolean;
    reason: string;
    action: string;
    latency_ms: number;
  };
  output_guard: {
    is_grounded: boolean;
    grounding_score: number;
    is_abstention: boolean;
    hallucination_detected: boolean;
    reason: string;
    latency_ms: number;
  };
  latency: LatencyBreakdown;
  is_abstention: boolean;
  provider: string;
  pipeline_trace?: PipelineTraceStep[];
}

interface BackendStatus {
  status: string;
  supported_languages: string[];
  indexed_doc_counts: Record<string, number>;
  embedding_model: string;
  llm_provider: string;
  llm_model: string;
  latency_target_ms: number;
}

interface BenchmarkReport {
  num_queries: number;
  languages: string[];
  summary: Record<string, {
    min: number;
    p50: number;
    p70: number;
    p90: number;
    p100: number;
    mean: number;
  }>;
}

const API_BASE = import.meta.env.VITE_API_BASE ?? '';

export const App: React.FC = () => {
  // State
  const [activeTab, setActiveTab] = useState<'rag' | 'analytics' | 'architecture'>('rag');
  const [inputMode, setInputMode] = useState<'voice' | 'text'>('voice');
  const [selectedLang, setSelectedLang] = useState<string>('auto');
  const [queryText, setQueryText] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<PipelineResponse | null>(null);
  const [systemStatus, setSystemStatus] = useState<BackendStatus | null>(null);

  // Audio Recording State
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordingDuration, setRecordingDuration] = useState<number>(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const timerIntervalRef = useRef<number | null>(null);

  // Benchmark State
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkReport | null>(null);
  const [isBenchmarking, setIsBenchmarking] = useState<boolean>(false);
  const [benchQueriesCount, setBenchQueriesCount] = useState<number>(15);

  // Trace State
  const [expandedSteps, setExpandedSteps] = useState<Record<number, boolean>>({});
  const [copiedStep, setCopiedStep] = useState<number | null>(null);

  const toggleStep = (stepNum: number) => {
    setExpandedSteps((prev) => ({
      ...prev,
      [stepNum]: !prev[stepNum],
    }));
  };

  const expandAllSteps = () => {
    if (!response?.pipeline_trace) return;
    const all: Record<number, boolean> = {};
    response.pipeline_trace.forEach((s) => {
      all[s.step_num] = true;
    });
    setExpandedSteps(all);
  };

  const collapseAllSteps = () => {
    setExpandedSteps({});
  };

  const copyToClipboard = (text: string, stepNum: number) => {
    navigator.clipboard.writeText(text);
    setCopiedStep(stepNum);
    setTimeout(() => setCopiedStep(null), 2000);
  };

  const getStepIcon = (stepId: string) => {
    switch (stepId) {
      case 'language_detection':
        return <Globe size={18} color="#00F2FE" />;
      case 'stt_transcription':
        return <Volume2 size={18} color="#FF6B35" />;
      case 'input_guardrail':
        return <ShieldCheck size={18} color="#10B981" />;
      case 'query_embedding':
        return <Cpu size={18} color="#8B5CF6" />;
      case 'dense_search':
        return <Search size={18} color="#06B6D4" />;
      case 'lexical_search':
        return <FileText size={18} color="#3B82F6" />;
      case 'rrf_fusion':
        return <Sliders size={18} color="#F59E0B" />;
      case 'final_chunks':
        return <Database size={18} color="#EC4899" />;
      case 'final_prompt':
        return <Terminal size={18} color="#A78BFA" />;
      case 'llm_output':
        return <Sparkles size={18} color="#FF6B35" />;
      case 'output_guardrail':
        return <CheckCircle2 size={18} color="#14B8A6" />;
      default:
        return <Activity size={18} color="#00F2FE" />;
    }
  };

  // Fetch initial system status
  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(data);
      }
    } catch (e) {
      console.warn('Backend API connection check:', e);
    }
  };

  // Start Voice Recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      // Setup Web Audio Analyser for live visualizer
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      source.connect(analyser);

      audioContextRef.current = audioCtx;
      analyserRef.current = analyser;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        if (audioContextRef.current) {
          audioContextRef.current.close();
        }
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await handleAudioUpload(audioBlob);
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      setRecordingDuration(0);

      // Start duration counter
      timerIntervalRef.current = window.setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);

      // Start visualizer loop
      drawWaveform();
    } catch (err) {
      console.error('Error accessing microphone:', err);
      alert('Could not access microphone. Please grant permission or use text query mode.');
    }
  };

  // Stop Voice Recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
    }
  };

  // Draw Audio Waveform on Canvas
  const drawWaveform = () => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const render = () => {
      animationFrameRef.current = requestAnimationFrame(render);
      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const barWidth = (canvas.width / bufferLength) * 2;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height * 0.9;
        const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
        gradient.addColorStop(0, '#FF6B35');
        gradient.addColorStop(1, '#00F2FE');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
        x += barWidth;
      }
    };

    render();
  };

  // Submit Text Query
  const handleTextSubmit = async (customQuery?: string) => {
    const q = customQuery || queryText;
    if (!q.trim()) return;

    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, language: selectedLang }),
      });
      if (res.ok) {
        const data = await res.json();
        setResponse(data);
      } else {
        alert('Server returned an error.');
      }
    } catch (err) {
      console.error('Error submitting query:', err);
      alert('Could not connect to FastAPI server. Please check your network or server status.');
    } finally {
      setIsLoading(false);
    }
  };

  // Submit Audio Blob
  const handleAudioUpload = async (audioBlob: Blob) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', audioBlob, 'voice_query.wav');
    formData.append('language', selectedLang);

    try {
      const res = await fetch(`${API_BASE}/api/query_audio`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        setResponse(data);
        if (data.query) {
          setQueryText(data.query);
        }
      } else {
        alert('Error processing voice query.');
      }
    } catch (err) {
      console.error('Error uploading audio:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Run Benchmark
  const handleRunBenchmark = async () => {
    setIsBenchmarking(true);
    try {
      const res = await fetch(`${API_BASE}/api/benchmark?num_queries=${benchQueriesCount}&languages=en,hi,ta`);
      if (res.ok) {
        const data = await res.json();
        setBenchmarkResult(data);
      }
    } catch (err) {
      console.error('Benchmark failed:', err);
    } finally {
      setIsBenchmarking(false);
    }
  };



  const sampleQuestions = {
    en: [
      'What is the capital of India?',
      'What is Retrieval-Augmented Generation?',
      'Where is Goa located?',
    ],
    hi: [
      'भारत की राजधानी क्या है?',
      'गोवा कहाँ स्थित है?',
      'रिट्रीवल-ऑगमेंटेड जनरेशन क्या है?',
    ],
    ta: [
      'இந்தியாவின் தலைநகரம் எது?',
      'கோவா எங்கு அமைந்துள்ளது?',
      'உலகின் மிகப்பெரிய பெருங்கடல் எது?',
    ],
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div>
          <h1 className="brand-title">
            <span className="gradient-text">Indic Voice-RAG</span>
            <span className="badge badge-saffron">HH Goa 2026</span>
          </h1>
          <p className="brand-subtitle">
            Voice-Enabled Retrieval-Augmented Generation • Multi-Indic (EN / HI / TA) • Target &lt;200ms Latency Budget
          </p>
        </div>

        <div className="header-status-badge">
          <div className="status-dot" />
          <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>
            {systemStatus ? 'Pipeline Ready' : 'Connecting...'}
          </span>
          <span className="badge badge-cyan" style={{ marginLeft: 8 }}>
            FAISS HNSW + BM25s
          </span>
        </div>
      </header>

      {/* Main Tabs Navigation */}
      <nav className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === 'rag' ? 'active' : ''}`}
          onClick={() => setActiveTab('rag')}
        >
          <Mic size={18} /> Voice / Text Studio
        </button>
        <button
          className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
          onClick={() => setActiveTab('analytics')}
        >
          <BarChart3 size={18} /> Latency Analytics (P50/P70/P100)
        </button>

        <button
          className={`tab-btn ${activeTab === 'architecture' ? 'active' : ''}`}
          onClick={() => setActiveTab('architecture')}
        >
          <BookOpen size={18} /> System Architecture
        </button>
      </nav>

      {/* TAB 1: VOICE / TEXT RAG STUDIO */}
      {activeTab === 'rag' && (
        <div>
          {/* Query Studio Card */}
          <div className="glass-panel query-card">
            {/* Language Selector Bar */}
            <div className="lang-selector-bar">
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600 }}>LANGUAGE:</span>
              <button
                className={`lang-pill ${selectedLang === 'auto' ? 'active' : ''}`}
                onClick={() => setSelectedLang('auto')}
              >
                <Globe size={14} /> Auto-Detect
              </button>
              <button
                className={`lang-pill ${selectedLang === 'gu' ? 'active' : ''}`}
                onClick={() => setSelectedLang('gu')}
              >
                🇮🇳 ગુજરાતી (Gujarati)
              </button>
              <button
                className={`lang-pill ${selectedLang === 'hi' ? 'active' : ''}`}
                onClick={() => setSelectedLang('hi')}
              >
                🇮🇳 हिन्दी (Hindi)
              </button>
              <button
                className={`lang-pill ${selectedLang === 'te' ? 'active' : ''}`}
                onClick={() => setSelectedLang('te')}
              >
                🇮🇳 తెలుగు (Telugu)
              </button>

              <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
                <button
                  className={`btn-secondary ${inputMode === 'voice' ? 'active' : ''}`}
                  onClick={() => setInputMode('voice')}
                  style={{ fontSize: '0.82rem', padding: '5px 12px' }}
                >
                  <Mic size={14} /> Voice Mode
                </button>
                <button
                  className={`btn-secondary ${inputMode === 'text' ? 'active' : ''}`}
                  onClick={() => setInputMode('text')}
                  style={{ fontSize: '0.82rem', padding: '5px 12px' }}
                >
                  ✍️ Text Mode
                </button>
              </div>
            </div>

            {/* Voice Input Section */}
            {inputMode === 'voice' && (
              <div className="voice-studio">
                <button
                  className={`mic-button ${isRecording ? 'recording recording-pulse' : ''}`}
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={isLoading}
                >
                  {isRecording ? <MicOff size={32} /> : <Mic size={32} />}
                </button>
                <h3 style={{ fontSize: '1.15rem', marginBottom: 4 }}>
                  {isRecording ? `Recording... (${recordingDuration}s)` : 'Click to Speak (Sarvam AI STT)'}
                </h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  {isRecording ? 'Click the red button to stop and synthesize response' : 'Speak clearly in Gujarati, Hindi, or Telugu'}
                </p>
                <canvas ref={canvasRef} className="waveform-canvas" width={400} height={60} />

                {/* File Upload Option */}
                <div style={{ marginTop: 14 }}>
                  <label className="btn-secondary" style={{ cursor: 'pointer', fontSize: '0.82rem' }}>
                    <Upload size={14} /> Or Upload Audio File (.wav / .mp3)
                    <input
                      type="file"
                      accept="audio/*"
                      style={{ display: 'none' }}
                      onChange={(e) => {
                        if (e.target.files && e.target.files[0]) {
                          handleAudioUpload(e.target.files[0]);
                        }
                      }}
                    />
                  </label>
                </div>
              </div>
            )}

            {/* Text Input Row */}
            <div className="input-row">
              <input
                type="text"
                className="text-input-field"
                placeholder="Ask anything in Gujarati, Hindi, or Telugu..."
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleTextSubmit();
                }}
              />
              <button
                className="btn-primary"
                onClick={() => handleTextSubmit()}
                disabled={isLoading || !queryText.trim()}
              >
                {isLoading ? <RefreshCw className="spin" size={18} /> : <Send size={18} />}
                <span>{isLoading ? 'Processing...' : 'Ask RAG'}</span>
              </button>
            </div>

            {/* Quick Sample Questions */}
            <div className="sample-queries-section">
              <div className="sample-queries-label">Suggested Test Queries (Click to Run):</div>
              <div className="sample-pills-row">
                {sampleQuestions.en.map((q, idx) => (
                  <button
                    key={`en-${idx}`}
                    className="sample-pill-btn"
                    onClick={() => {
                      setQueryText(q);
                      handleTextSubmit(q);
                    }}
                  >
                    🇬🇧 {q}
                  </button>
                ))}
                {sampleQuestions.hi.map((q, idx) => (
                  <button
                    key={`hi-${idx}`}
                    className="sample-pill-btn"
                    onClick={() => {
                      setQueryText(q);
                      handleTextSubmit(q);
                    }}
                  >
                    🇮🇳 {q}
                  </button>
                ))}
                {sampleQuestions.ta.map((q, idx) => (
                  <button
                    key={`ta-${idx}`}
                    className="sample-pill-btn"
                    onClick={() => {
                      setQueryText(q);
                      handleTextSubmit(q);
                    }}
                  >
                    🇮🇳 {q}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Results Display */}
          {response && (
            <div className="animate-fade-in">
              {/* Metrics Dashboard */}
              <div className="metrics-grid">
                <div className="metric-box">
                  <div className="metric-title">
                    <Clock size={16} color="var(--accent-cyan)" /> Total End-to-End Latency
                  </div>
                  <div className="metric-val" style={{ color: response.latency.total_end_to_end_ms <= 200 ? '#10B981' : '#F59E0B' }}>
                    {response.latency.total_end_to_end_ms.toFixed(1)} <span style={{ fontSize: '1rem' }}>ms</span>
                  </div>
                  <div className="metric-sub">
                    Target: &lt;200ms {response.latency.total_end_to_end_ms <= 200 ? '✅ Budget Met' : '⚠️ Warning'}
                  </div>
                </div>

                <div className="metric-box">
                  <div className="metric-title">
                    <Zap size={16} color="var(--accent-saffron)" /> Hybrid Retrieval
                  </div>
                  <div className="metric-val" style={{ color: '#00F2FE' }}>
                    {response.latency.total_retrieval_ms.toFixed(1)} <span style={{ fontSize: '1rem' }}>ms</span>
                  </div>
                  <div className="metric-sub">
                    FAISS: {response.latency.dense_search_ms.toFixed(2)}ms | BM25: {response.latency.lexical_search_ms.toFixed(2)}ms
                  </div>
                </div>

                <div className="metric-box">
                  <div className="metric-title">
                    <ShieldCheck size={16} color="var(--accent-green)" /> Input Guardrail
                  </div>
                  <div className="metric-val" style={{ color: response.input_guard.is_safe ? '#10B981' : '#EF4444' }}>
                    {response.input_guard.is_safe ? 'PASSED' : 'BLOCKED'}
                  </div>
                  <div className="metric-sub">
                    Action: {response.input_guard.action} ({response.input_guard.latency_ms.toFixed(2)}ms)
                  </div>
                </div>

                <div className="metric-box">
                  <div className="metric-title">
                    <CheckCircle2 size={16} color="var(--accent-purple)" /> Context Grounding
                  </div>
                  <div className="metric-val" style={{ color: '#A78BFA' }}>
                    {(response.output_guard.grounding_score * 100).toFixed(0)}%
                  </div>
                  <div className="metric-sub">
                    Hallucination Check: {response.output_guard.hallucination_detected ? '⚠️ Flagged' : '✅ Verified'}
                  </div>
                </div>
              </div>

              {/* Grid with Answer & Waterfall */}
              <div className="results-grid">
                {/* Left Column: Answer Box */}
                <div>
                  <div className="glass-panel answer-box">
                    <div className="answer-header">
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <Sparkles size={20} color="var(--accent-saffron)" />
                        <h3 style={{ fontSize: '1.2rem' }}>Synthesized Answer</h3>
                      </div>
                      <div style={{ display: 'flex', gap: 8 }}>
                        <span className="badge badge-cyan">
                          Lang: {response.detected_lang.toUpperCase()}
                        </span>
                        <span className="badge badge-saffron">
                          {response.provider}
                        </span>
                      </div>
                    </div>

                    {response.is_abstention ? (
                      <div className="abstention-banner">
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                          <AlertTriangle size={18} />
                          <strong>Safe Abstention Triggered:</strong>
                        </div>
                        {response.answer}
                      </div>
                    ) : (
                      <div className="answer-text">
                        {response.answer}
                      </div>
                    )}

                    <div style={{ marginTop: 16, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                      <strong>Input Query:</strong> <em>"{response.query}"</em>
                    </div>
                  </div>

                  {/* Waterfall Latency Breakdown Chart */}
                  <div className="glass-panel waterfall-card">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                      <Activity size={18} color="var(--accent-cyan)" />
                      <h3 style={{ fontSize: '1.1rem' }}>Latency Waterfall Breakdown</h3>
                    </div>

                    {[
                      { name: 'STT (Speech-to-Text)', val: response.latency.stt_ms, color: '#FF6B35' },
                      { name: 'Input Guardrail Check', val: response.latency.input_guardrail_ms, color: '#10B981' },
                      { name: 'BGE-M3 Query Embedding', val: response.latency.embedding_ms, color: '#8B5CF6' },
                      { name: 'FAISS HNSW Dense Search', val: response.latency.dense_search_ms, color: '#06B6D4' },
                      { name: 'BM25s Lexical Search', val: response.latency.lexical_search_ms, color: '#3B82F6' },
                      { name: 'Reciprocal Rank Fusion (RRF)', val: response.latency.fusion_ms, color: '#F59E0B' },
                      { name: 'LLM Generation / Synthesis', val: response.latency.llm_generation_ms, color: '#EC4899' },
                      { name: 'Output Grounding Guardrail', val: response.latency.output_guardrail_ms, color: '#14B8A6' },
                    ].map((step, i) => {
                      const total = Math.max(response.latency.total_end_to_end_ms, 1);
                      const percent = Math.min((step.val / total) * 100, 100);
                      return (
                        <div key={i} className="waterfall-row">
                          <div className="waterfall-meta">
                            <span>{step.name}</span>
                            <span style={{ fontFamily: 'var(--font-mono)' }}>{step.val.toFixed(2)} ms</span>
                          </div>
                          <div className="progress-bar-bg">
                            <div
                              className="progress-bar-fill"
                              style={{ width: `${Math.max(percent, 2)}%`, backgroundColor: step.color }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Right Column: Retrieved Context Passages */}
                <div>
                  <div className="glass-panel passages-card">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                      <Layers size={18} color="var(--accent-purple)" />
                      <h3 style={{ fontSize: '1.1rem' }}>
                        Retrieved Context Passages ({response.retrieved_documents.length})
                      </h3>
                    </div>

                    {response.retrieved_documents.length === 0 ? (
                      <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>No passages retrieved.</p>
                    ) : (
                      response.retrieved_documents.map((doc, idx) => (
                        <div key={idx} className="passage-item">
                          <div className="passage-header">
                            <span style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--accent-cyan)' }}>
                              [{idx + 1}] Doc ID: {doc.passage_id}
                            </span>
                            <div style={{ display: 'flex', gap: 6 }}>
                              <span className="badge badge-green" style={{ fontSize: '0.7rem' }}>
                                Cosine: {doc.dense_score.toFixed(3)}
                              </span>
                              <span className="badge badge-cyan" style={{ fontSize: '0.7rem' }}>
                                RRF: {doc.rrf_score.toFixed(4)}
                              </span>
                            </div>
                          </div>
                          <p className="passage-content">{doc.raw_text}</p>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>

              {/* SECTION: Pipeline Execution Trace (11 Granular Stages with Latencies) */}
              {response.pipeline_trace && response.pipeline_trace.length > 0 && (
                <div className="glass-panel trace-container" style={{ marginTop: 24 }}>
                  <div className="trace-main-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <Activity size={22} color="var(--accent-cyan)" />
                      <div>
                        <h3 style={{ fontSize: '1.25rem', margin: 0, fontWeight: 700 }}>
                          Pipeline Execution Trace &amp; Inspection
                        </h3>
                        <p style={{ margin: '2px 0 0', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                          Step-by-step intermediate execution trace from audio / query ingestion to output guardrails ({response.pipeline_trace.length} stages)
                        </p>
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        className="btn-secondary"
                        onClick={expandAllSteps}
                        style={{ fontSize: '0.78rem', padding: '5px 10px' }}
                      >
                        Expand All
                      </button>
                      <button
                        className="btn-secondary"
                        onClick={collapseAllSteps}
                        style={{ fontSize: '0.78rem', padding: '5px 10px' }}
                      >
                        Collapse All
                      </button>
                    </div>
                  </div>

                  <div className="trace-timeline">
                    {response.pipeline_trace.map((step) => {
                      const isExpanded = !!expandedSteps[step.step_num];
                      const isPassed = step.status === 'passed' || step.status === 'completed';
                      const isBlocked = step.status === 'blocked' || step.status === 'failed';
                      const isFlagged = step.status === 'flagged' || step.status === 'warning';
                      const isAbstention = step.status === 'abstention';

                      return (
                        <div key={step.step_num} className={`trace-step-card ${isExpanded ? 'expanded' : ''}`}>
                          {/* Step Header Bar */}
                          <div
                            className="trace-step-header"
                            onClick={() => toggleStep(step.step_num)}
                          >
                            <div className="trace-step-left">
                              <span className="trace-step-num-badge">{step.step_num}</span>
                              <div className="trace-step-icon">{getStepIcon(step.step_id)}</div>
                              <span className="trace-step-title">{step.step_name}</span>
                            </div>

                            <div className="trace-step-right">
                              {/* Status Badge */}
                              <span
                                className={`badge ${
                                  isBlocked
                                    ? 'badge-red'
                                    : isFlagged
                                    ? 'badge-orange'
                                    : isAbstention
                                    ? 'badge-purple'
                                    : 'badge-green'
                                }`}
                                style={{ textTransform: 'uppercase', fontSize: '0.72rem', letterSpacing: 0.5 }}
                              >
                                {isBlocked && <XCircle size={12} style={{ marginRight: 4 }} />}
                                {isPassed && <CheckCircle size={12} style={{ marginRight: 4 }} />}
                                {isFlagged && <AlertTriangle size={12} style={{ marginRight: 4 }} />}
                                {step.status}
                              </span>

                              {/* Timing Badge */}
                              <span className="trace-step-time">
                                <Clock size={12} style={{ marginRight: 4 }} />
                                {step.time_ms.toFixed(2)} ms
                              </span>

                              {/* Chevron */}
                              <div className="trace-chevron">
                                {isExpanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                              </div>
                            </div>
                          </div>

                          {/* Step Detailed Drawer */}
                          {isExpanded && (
                            <div className="trace-step-body animate-fade-in">
                              {/* Custom rendering based on step_id */}
                              {step.step_id === 'final_prompt' && (
                                <div className="trace-prompt-view">
                                  <div className="trace-prompt-section">
                                    <div className="trace-prompt-header">
                                      <span>⚙️ System Prompt:</span>
                                      <button
                                        className="trace-copy-btn"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          copyToClipboard(step.details.system_prompt || '', step.step_num);
                                        }}
                                      >
                                        {copiedStep === step.step_num ? <Check size={12} /> : <Copy size={12} />}
                                        <span>{copiedStep === step.step_num ? 'Copied' : 'Copy'}</span>
                                      </button>
                                    </div>
                                    <pre className="trace-code-block">{step.details.system_prompt}</pre>
                                  </div>

                                  <div className="trace-prompt-section" style={{ marginTop: 12 }}>
                                    <div className="trace-prompt-header">
                                      <span>💬 User Prompt (Context + Question):</span>
                                      <button
                                        className="trace-copy-btn"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          copyToClipboard(step.details.user_prompt || '', step.step_num + 100);
                                        }}
                                      >
                                        {copiedStep === step.step_num + 100 ? <Check size={12} /> : <Copy size={12} />}
                                        <span>{copiedStep === step.step_num + 100 ? 'Copied' : 'Copy'}</span>
                                      </button>
                                    </div>
                                    <pre className="trace-code-block">{step.details.user_prompt}</pre>
                                  </div>
                                </div>
                              )}

                              {step.step_id === 'final_chunks' && (
                                <div className="trace-chunks-view">
                                  <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 8 }}>
                                    Retrieved <strong>{step.details.chunks_count}</strong> passages (Max Similarity: {step.details.max_similarity_score}):
                                  </div>
                                  <div className="trace-chunks-grid">
                                    {step.details.chunks && step.details.chunks.map((c: any, i: number) => (
                                      <div key={i} className="trace-chunk-card">
                                        <div className="trace-chunk-header">
                                          <span style={{ fontWeight: 700, color: 'var(--accent-cyan)' }}>
                                            #{c.rank} • Doc: {c.passage_id}
                                          </span>
                                          <div style={{ display: 'flex', gap: 4 }}>
                                            <span className="badge badge-green" style={{ fontSize: '0.68rem' }}>
                                              Cosine: {c.dense_score}
                                            </span>
                                            <span className="badge badge-cyan" style={{ fontSize: '0.68rem' }}>
                                              RRF: {c.rrf_score}
                                            </span>
                                          </div>
                                        </div>
                                        <div className="trace-chunk-sources">
                                          {c.match_sources && c.match_sources.map((src: string, si: number) => (
                                            <span key={si} className="trace-source-tag">🏷️ {src}</span>
                                          ))}
                                        </div>
                                        <p className="trace-chunk-preview">"{c.text_preview}"</p>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {step.step_id !== 'final_prompt' && step.step_id !== 'final_chunks' && (
                                <div className="trace-meta-table-wrap">
                                  <table className="trace-meta-table">
                                    <tbody>
                                      {Object.entries(step.details).map(([key, val]) => (
                                        <tr key={key}>
                                          <td className="trace-meta-key">{key.replace(/_/g, ' ')}</td>
                                          <td className="trace-meta-val">
                                            {typeof val === 'object' && val !== null ? (
                                              <pre className="trace-inline-json">{JSON.stringify(val, null, 2)}</pre>
                                            ) : typeof val === 'boolean' ? (
                                              <span style={{ fontWeight: 700, color: val ? '#10B981' : '#EF4444' }}>
                                                {val ? 'TRUE (PASSED)' : 'FALSE (FAILED)'}
                                              </span>
                                            ) : (
                                              String(val)
                                            )}
                                          </td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* TAB 2: LATENCY ANALYTICS & BENCHMARK */}
      {activeTab === 'analytics' && (
        <div className="animate-fade-in">
          <div className="glass-panel" style={{ padding: 24, marginBottom: 24 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16 }}>
              <div>
                <h2 style={{ fontSize: '1.5rem', marginBottom: 4 }}>📈 Pipeline Latency Benchmark Harness</h2>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem' }}>
                  Evaluates P50 / P70 / P90 / P100 latency percentiles across Gujarati, Hindi, and Telugu test queries.
                </p>
              </div>

              <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                <label style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
                  Queries:
                  <select
                    value={benchQueriesCount}
                    onChange={(e) => setBenchQueriesCount(Number(e.target.value))}
                    style={{
                      marginLeft: 8,
                      background: 'rgba(255,255,255,0.06)',
                      color: 'white',
                      border: '1px solid var(--border-color)',
                      padding: '6px 12px',
                      borderRadius: 6,
                    }}
                  >
                    <option value={10}>10 queries</option>
                    <option value={15}>15 queries</option>
                    <option value={30}>30 queries</option>
                  </select>
                </label>

                <button
                  className="btn-primary"
                  onClick={handleRunBenchmark}
                  disabled={isBenchmarking}
                >
                  <Zap size={16} />
                  {isBenchmarking ? 'Running Benchmark...' : '⚡ Run Full Benchmark'}
                </button>
              </div>
            </div>

            {/* Benchmark Summary Numbers */}
            {benchmarkResult && (
              <div style={{ marginTop: 24 }}>
                <div className="metrics-grid">
                  <div className="metric-box">
                    <div className="metric-title">P50 (Median) Latency</div>
                    <div className="metric-val" style={{ color: '#10B981' }}>
                      {benchmarkResult.summary.total_end_to_end.p50.toFixed(2)} ms
                    </div>
                    <div className="metric-sub">✅ Within &lt;200ms budget</div>
                  </div>

                  <div className="metric-box">
                    <div className="metric-title">P70 Latency</div>
                    <div className="metric-val" style={{ color: '#34D399' }}>
                      {benchmarkResult.summary.total_end_to_end.p70.toFixed(2)} ms
                    </div>
                    <div className="metric-sub">✅ Sub-60ms performance</div>
                  </div>

                  <div className="metric-box">
                    <div className="metric-title">P90 Latency</div>
                    <div className="metric-val" style={{ color: '#38BDF8' }}>
                      {benchmarkResult.summary.total_end_to_end.p90.toFixed(2)} ms
                    </div>
                    <div className="metric-sub">90th Percentile</div>
                  </div>

                  <div className="metric-box">
                    <div className="metric-title">P100 (Max) Latency</div>
                    <div className="metric-val" style={{ color: '#FB923C' }}>
                      {benchmarkResult.summary.total_end_to_end.p100.toFixed(2)} ms
                    </div>
                    <div className="metric-sub">Maximum observed time</div>
                  </div>
                </div>

                {/* Table of Component Percentiles */}
                <h3 style={{ fontSize: '1.2rem', marginTop: 20, marginBottom: 10 }}>
                  Detailed Component Latency Breakdown (ms)
                </h3>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Pipeline Component</th>
                      <th>Min (ms)</th>
                      <th>P50 (Median)</th>
                      <th>P70 (ms)</th>
                      <th>P90 (ms)</th>
                      <th>P100 (Max)</th>
                      <th>Mean (ms)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(benchmarkResult.summary).map(([key, s]) => (
                      <tr key={key}>
                        <td style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>
                          <code>{key}</code>
                        </td>
                        <td>{s.min.toFixed(2)}</td>
                        <td style={{ fontWeight: 700, color: '#10B981' }}>{s.p50.toFixed(2)}</td>
                        <td>{s.p70.toFixed(2)}</td>
                        <td>{s.p90.toFixed(2)}</td>
                        <td>{s.p100.toFixed(2)}</td>
                        <td>{s.mean.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}



      {/* TAB 4: ARCHITECTURE & SPECS */}
      {activeTab === 'architecture' && (
        <div className="animate-fade-in">
          <div className="glass-panel" style={{ padding: 24, lineHeight: 1.7 }}>
            <h2 style={{ fontSize: '1.5rem', marginBottom: 12 }}>📐 System Specifications &amp; Architecture</h2>

            <h3 style={{ fontSize: '1.2rem', color: 'var(--accent-saffron)', marginTop: 20 }}>1. Speech-to-Text (STT)</h3>
            <p style={{ color: 'var(--text-secondary)' }}>
              Integrated with <strong>Sarvam AI (saaras:v2)</strong> for native Gujarati, Hindi, and Telugu speech recognition with code-mixing support.
            </p>

            <h3 style={{ fontSize: '1.2rem', color: 'var(--accent-cyan)', marginTop: 20 }}>2. Multi-Tier Chunking Taxonomy</h3>
            <ul style={{ color: 'var(--text-secondary)', marginLeft: 20 }}>
              <li><strong>Atomic Passage Chunking:</strong> Preserves complete semantic integrity of source MSMARCO passages.</li>
              <li><strong>Hierarchical Sliding Window:</strong> 256-word chunking with 64-word sentence-boundary overlap.</li>
              <li><strong>Metadata Augmentation:</strong> Injects language, passage ID, and topical tags into dense embeddings.</li>
              <li><strong>Query-Anchor Pairing:</strong> Dual bidirectional matching between synthetic queries and passages.</li>
            </ul>

            <h3 style={{ fontSize: '1.2rem', color: 'var(--accent-purple)', marginTop: 20 }}>3. Sub-200ms Hybrid Retrieval</h3>
            <p style={{ color: 'var(--text-secondary)' }}>
              Combines <strong>FAISS HNSW</strong> dense vector search over L2-normalized <code>BAAI/bge-m3</code> embeddings with <strong>BM25s</strong> lexical keyword retrieval using Reciprocal Rank Fusion (RRF, k=60).
            </p>

            <h3 style={{ fontSize: '1.2rem', color: 'var(--accent-green)', marginTop: 20 }}>4. Guardrails &amp; Safe Abstention</h3>
            <p style={{ color: 'var(--text-secondary)' }}>
              Pre-retrieval input guardrails protect against prompt injections, toxic queries, and low-confidence audio. Post-generation output guardrails verify context grounding and trigger safe refusals when database relevance is insufficient.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
