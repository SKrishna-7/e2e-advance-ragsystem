import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, Plus, MoreHorizontal, Search, Filter, User, Bot, 
  Trash2, Copy, RefreshCw, Mic, Sparkles, 
  FileText, X, Settings, Folder, Menu, Sliders, Database, 
  Save, Loader2, BookOpen, UploadCloud, ChevronRight, AlertCircle, ExternalLink, Activity, Zap, Brain
} from 'lucide-react';

const API_URL = `${window.location.protocol}//${window.location.hostname}:8000`;

export default function App() {
  // --- STATE ---
  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: 'Hello! I am your Agentic RAG assistant. \n\nI can answer questions based on your uploaded documents. Upload a PDF to get started.',
    intent: 'GREETING',
    timestamp: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
  }]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [deletingFile, setDeletingFile] = useState(null);
  
  // Processing Status
  const [processingStatus, setProcessingStatus] = useState({ is_active: false, message: "Idle", step: "idle" });

  // Ingestion Mode
  const [ingestionMode, setIngestionMode] = useState("fast"); // 'fast' or 'advanced'

  // Modals & Data
  const [selectedSource, setSelectedSource] = useState(null);
  const [stats, setStats] = useState({ total_files: 0, total_pages: 0, files: [] });
  
  const [threadId] = useState(() => {
    const stored = localStorage.getItem("rag_thread_id");
    if (stored) return stored;
    const newId = "user_" + Math.random().toString(36).substr(2, 9);
    localStorage.setItem("rag_thread_id", newId);
    return newId;
  });

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // --- EFFECTS ---
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(() => {
      if (!document.hidden) {
        fetch(`${API_URL}/stats`).then(r => r.ok && r.json().then(setStats)).catch(()=>{});
        fetch(`${API_URL}/ingestion/status`).then(r => r.ok && r.json().then(setProcessingStatus)).catch(()=>{});
      }
    }, 3000); 
    return () => clearInterval(interval);
  }, []);

  // --- API ACTIONS ---
  const fetchData = async () => {
    try {
      const sRes = await fetch(`${API_URL}/stats`);
      if (sRes.ok) setStats(await sRes.json());
    } catch (e) { console.error("Init fetch failed", e); }
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;
    
    // 1. Add User Message
    const userMsg = { role: 'user', content: input, timestamp: new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) };
    setMessages(p => [...p, userMsg]);
    
    // 2. Add Placeholder for Assistant Message
    const assistantMsgId = Date.now();
    setMessages(p => [...p, {
      id: assistantMsgId,
      role: 'assistant', 
      content: '', // Start empty for streaming
      sources: [],
      timestamp: new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})
    }]);
    
    const currentInput = input;
    setInput('');
    setIsLoading(true);

    try {
      // 3. Call Streaming Endpoint
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question: currentInput, 
          thread_id: threadId,
          user_id: threadId 
        })
      });

      if (!response.ok) throw new Error("Stream Error");

      // 4. Read the Stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        // Split by SSE double newline
        const lines = chunk.split('\n\n');

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const jsonStr = line.replace("data: ", "").trim();
            if (jsonStr === "[DONE]") break;
            
            try {
              const event = JSON.parse(jsonStr);
              
              if (event.type === 'token') {
                accumulatedText += event.content;
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMsgId ? { ...msg, content: accumulatedText } : msg
                ));
              } else if (event.type === 'sources') {
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMsgId ? { ...msg, sources: event.data } : msg
                ));
              } else if (event.type === 'error') {
                 console.error("Stream Error Event:", event.content);
              }
            } catch (e) {
              // Partial JSON chunks can happen, ignore them
            }
          }
        }
      }

    } catch (e) {
      console.error(e);
      setMessages(p => [...p, { role: 'assistant', content: "⚠️ Connection error. Please check backend.", isError: true }]);
    } finally { 
      setIsLoading(false); 
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setProcessingStatus({ is_active: true, message: `Uploading (${ingestionMode} mode)...`, step: "processing" });

    const fd = new FormData();
    fd.append('file', file);
    fd.append('mode', ingestionMode); // Pass the selected mode to backend

    try {
      const res = await fetch(`${API_URL}/upload`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error("Upload failed");
    } catch (e) { 
      alert("Upload failed. Check backend logs."); 
      setProcessingStatus({ is_active: false, message: "Failed", step: "error" });
    } finally { 
      if (fileInputRef.current) fileInputRef.current.value=""; 
    }
  };

  const handleDeleteFile = async (filename) => {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) return;
    setDeletingFile(filename);
    try {
      const res = await fetch(`${API_URL}/delete/${filename}`, { method: 'DELETE' });
      if (res.ok) {
        setStats(prev => ({
            ...prev,
            files: prev.files.filter(f => f.name !== filename),
            total_files: prev.total_files - 1
        }));
      } else {
        alert("Failed to delete file.");
      }
    } catch (e) { console.error(e); } 
    finally { setDeletingFile(null); fetchData(); }
  };

  // --- RENDER ---
  return (
    <div className="flex h-dvh w-dvw bg-[#F3F4F6] font-sans text-gray-800 overflow-hidden">
      {isSidebarOpen && <div className="fixed inset-0 bg-black/20 z-20 md:hidden backdrop-blur-sm" onClick={() => setIsSidebarOpen(false)} />}

      {/* --- SIDEBAR --- */}
      <div className={`fixed inset-y-0 left-0 z-30 w-80 bg-white border-r border-gray-200 flex flex-col shadow-2xl transition-transform duration-300 md:relative md:translate-x-0 ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        
        {/* Header */}
        <div className="p-6 border-b border-gray-100 bg-white/50 backdrop-blur-md sticky top-0 z-10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
               <div className="w-10 h-10 bg-gradient-to-tr from-[#00A67E] to-[#008c6b] rounded-xl flex items-center justify-center shadow-lg shadow-[#00A67E]/20 text-white">
                 <Sparkles fill="white" size={20} />
               </div>
               <div>
                 <p className="font-bold text-xl text-gray-900 tracking-tight leading-tight">Agentic RAG</p>
               </div>
            </div>
            <button className="md:hidden p-2 text-gray-400" onClick={() => setIsSidebarOpen(false)}><X size={20} /></button>
          </div>
        </div>

        {/* Stats */}
        <div className="p-6 grid grid-cols-2 gap-3">
           <div className="bg-gray-50 border border-gray-100 p-4 rounded-2xl flex flex-col items-center justify-center gap-1">
              <span className="text-2xl font-bold text-gray-900">{stats.total_files}</span>
              <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest">Docs</span>
           </div>
           <div className="bg-gray-50 border border-gray-100 p-4 rounded-2xl flex flex-col items-center justify-center gap-1">
              <span className="text-2xl font-bold text-gray-900">{stats.total_pages}</span>
              <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest">Pages</span>
           </div>
        </div>

        {/* File List */}
        <div className="flex-1 overflow-y-auto px-4 pb-4">
          <div className="flex items-center justify-between mb-4 px-2">
            <h3 className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-2">
              <Database size={12} /> Knowledge Base
            </h3>
          </div>
          <div className="space-y-2">
            {stats.files.map((f, i) => (
              <div key={i} className="group flex items-center justify-between p-3 bg-white border border-gray-100 hover:border-[#00A67E]/30 hover:shadow-sm hover:bg-[#F0FDF9] rounded-xl transition-all duration-200">
                 <div className="flex items-center gap-3 overflow-hidden">
                   <div className="p-2 bg-gray-50 group-hover:bg-white rounded-lg">
                     <FileText size={16} className="text-gray-400 group-hover:text-[#00A67E]" />
                   </div>
                   <div className="flex flex-col min-w-0">
                     <span className="text-sm font-medium text-gray-700 truncate group-hover:text-gray-900" title={f.name}>{f.name}</span>
                     <span className="text-[10px] text-gray-400">{f.pages} pages</span>
                   </div>
                 </div>
                 <button onClick={(e) => { e.stopPropagation(); handleDeleteFile(f.name); }} disabled={deletingFile === f.name} className="opacity-0 group-hover:opacity-100 p-2 text-gray-400 hover:text-red-500 rounded-lg transition-all">
                    {deletingFile === f.name ? <Loader2 size={16} className="animate-spin text-red-500"/> : <Trash2 size={16} />}
                 </button>
              </div>
            ))}
            {stats.files.length === 0 && (
              <div className="text-center py-10 px-4 border-2 border-dashed border-gray-100 rounded-2xl">
                <div className="w-12 h-12 bg-gray-50 rounded-full flex items-center justify-center mx-auto mb-3"><Folder size={20} className="text-gray-300"/></div>
                <p className="text-xs text-gray-400 font-medium">No documents yet.</p>
              </div>
            )}
          </div>
        </div>

        {/* Upload / Progress Area */}
        <div className="p-6 bg-white border-t border-gray-100 z-10">
          {processingStatus.is_active ? (
            <div className="w-full bg-[#F0FDF9] border border-[#00A67E]/20 p-4 rounded-2xl animate-pulse">
               <div className="flex items-center gap-3 mb-2">
                 <Loader2 className="w-5 h-5 text-[#00A67E] animate-spin" />
                 <span className="text-sm font-bold text-[#00A67E]">{processingStatus.step === 'complete' ? 'Success' : 'Processing'}</span>
               </div>
               <p className="text-xs text-gray-600 font-medium">{processingStatus.message}</p>
            </div>
          ) : (
            <>
              {/* --- NEW: Ingestion Mode Toggle --- */}
              <div className="flex bg-gray-100 p-1 rounded-xl mb-4">
                <div 
                  onClick={() => setIngestionMode("fast")}
                  className={`cursor-pointer flex-1 flex items-center justify-center gap-2 py-2 text-xs font-semibold rounded-lg transition-all ${ingestionMode === 'fast' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                  title="Text only. Instant."
                >
                  <Zap size={14} className={ingestionMode === 'fast' ? "text-amber-500 fill-amber-500" : ""} /> Fast
                </div>
                <div 
                  onClick={() => setIngestionMode("advanced")}
                  className={`cursor-pointer flex-1 flex items-center justify-center gap-2 py-2 text-xs font-semibold rounded-lg transition-all ${ingestionMode === 'advanced' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                  title="OCR & Images. Slower."
                >
                  <Brain size={14} className={ingestionMode === 'advanced' ? "text-purple-500" : ""} /> Advanced
                </div>
              </div>

              <div 
                onClick={() => fileInputRef.current?.click()}
                className="cursor-pointer w-full flex items-center justify-center gap-3 bg-gray-900 hover:bg-black text-white text-sm font-semibold py-4 rounded-2xl shadow-lg shadow-gray-900/10 transition-all hover:scale-[1.02] active:scale-[0.98]"
              >
                <UploadCloud className="w-5 h-5" /> Upload New PDF
              </div>
              <input type="file" ref={fileInputRef} className="hidden" accept=".pdf" onChange={handleFileUpload} />
            </>
          )}
          
          <div className="mt-6 text-center">
            <a href="https://sureshkrishnan.vercel.app/" target="_blank" className="group inline-flex items-center gap-1.5 text-[10px] font-medium text-gray-400 hover:text-[#00A67E]">
              Built by Suresh Krishnan <ExternalLink size={10} className="opacity-0 group-hover:opacity-100" />
            </a>
          </div>
        </div>
      </div>

      {/* --- CHAT AREA --- */}
      <div className="flex-1 flex flex-col relative w-full h-full">
        <div className="h-20 flex items-center justify-between px-6 bg-transparent z-10">
          <div className="flex items-center gap-3">
             <button onClick={() => setIsSidebarOpen(true)} className="md:hidden p-2 -ml-2 text-gray-500"><Menu size={24}/></button>
             <div className="hidden md:flex flex-col">
                <h2 className="font-bold text-gray-800">Session ID: <span className="font-mono text-gray-500 font-normal">{threadId.slice(-6)}</span></h2>
                <span className="text-xs text-[#00A67E] flex items-center gap-1 font-medium"><span className="w-1.5 h-1.5 rounded-full bg-[#00A67E] animate-pulse"/> Active</span>
             </div>
          </div>
          <button className="p-2 text-gray-400 hover:text-gray-600 hover:bg-white rounded-xl" onClick={() => setMessages([])}><RefreshCw size={20} /></button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 md:px-8 py-4 space-y-8 scroll-smooth pb-40">
          {messages.map((msg, i) => (
            <div key={i} className={`flex gap-4 max-w-4xl mx-auto ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-4 duration-500`}>
              {msg.role === 'assistant' && <div className="w-10 h-10 rounded-2xl bg-white border border-gray-100 shadow-sm flex-shrink-0 flex items-center justify-center text-[#00A67E]"><Bot size={20} /></div>}
              <div className={`flex flex-col max-w-[85%] md:max-w-[75%] space-y-1 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                <div className="flex items-center gap-2 px-1">
                  <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">{msg.role === 'assistant' ? 'AI Agent' : 'You'}</span>
                  <span className="text-[10px] text-gray-300">{msg.timestamp}</span>
                </div>
                {msg.role === 'system' ? (
                  <div className="inline-flex items-center gap-2 text-xs bg-[#00A67E]/10 text-[#00A67E] px-4 py-2 rounded-full font-medium border border-[#00A67E]/20">
                    <Loader2 size={12} className="animate-spin"/> {msg.content}
                  </div>
                ) : (
                  <div className={`p-6 rounded-3xl shadow-sm text-[15px] leading-relaxed relative group ${msg.role === 'assistant' ? 'bg-white border border-gray-100 text-gray-700 rounded-tl-none' : 'bg-gray-900 text-white rounded-tr-none'}`}>
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                    {msg.sources?.length > 0 && (
                      <div className="mt-6 pt-4 border-t border-dashed border-gray-100">
                        <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3 flex items-center gap-1"><BookOpen size={12}/> Sources & Evidence</p>
                        <div className="flex flex-wrap gap-2">
                          {msg.sources.map((src, k) => (
                            <div key={k} onClick={() => setSelectedSource(src)} className="flex items-center gap-2 pl-2 pr-3 py-1.5 bg-gray-50 hover:bg-[#F0FDF9] border border-gray-200 rounded-lg text-left group/card cursor-pointer">
                               <div className="w-6 h-6 bg-white rounded-md border border-gray-100 flex items-center justify-center text-gray-400 group-hover/card:text-[#00A67E]"><FileText size={12}/></div>
                               <div><div className="text-[11px] font-semibold text-gray-700 max-w-[100px] truncate">{src.filename}</div><div className="text-[9px] text-gray-400">Page {src.page}</div></div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {msg.sources?.some(s => s.images?.length > 0) && (
                      <div className="mt-4 grid grid-cols-2 gap-2">
                         {msg.sources.flatMap(s => s.images).map((img, idx) => <div key={idx} className="relative aspect-video rounded-xl overflow-hidden border border-gray-100 bg-gray-50"><img src={img} className="w-full h-full object-contain" /></div>)}
                      </div>
                    )}
                  </div>
                )}
              </div>
              {msg.role === 'user' && <div className="w-10 h-10 rounded-2xl bg-gray-200 flex-shrink-0 flex items-center justify-center text-gray-500 overflow-hidden"><User size={20} /></div>}
            </div>
          ))}
          {isLoading && !messages[messages.length - 1]?.content && (
             <div className="flex gap-4 max-w-4xl mx-auto">
               <div className="w-10 h-10 rounded-2xl bg-white border border-gray-100 shadow-sm flex-shrink-0 flex items-center justify-center text-[#00A67E] animate-pulse">
                 <Bot size={20} />
               </div>
               <div className="bg-white px-6 py-4 rounded-3xl rounded-tl-none border border-gray-100 shadow-sm flex items-center gap-2">
                 <span className="w-1.5 h-1.5 bg-[#00A67E] rounded-full animate-bounce"/>
                 <span className="w-1.5 h-1.5 bg-[#00A67E] rounded-full animate-bounce delay-100"/>
                 <span className="w-1.5 h-1.5 bg-[#00A67E] rounded-full animate-bounce delay-200"/>
               </div>
             </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="absolute bottom-6 left-0 right-0 px-4 md:px-0 z-20">
          <div className="max-w-3xl mx-auto">
             <div className="bg-white/80 backdrop-blur-xl border border-white/40 shadow-2xl shadow-gray-900/10 rounded-3xl p-2 flex items-center gap-2 relative ring-1 ring-gray-900/5 focus-within:ring-[#00A67E]/50 transition-all duration-300">
                <div className="pl-4"><Sparkles size={20} className="text-[#00A67E] animate-pulse" /></div>
                <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()} placeholder="Ask a question..." className="flex-1 bg-transparent border-none outline-none text-gray-800 placeholder-gray-400 h-12 text-[15px] font-medium" disabled={isLoading}/>
                <div onClick={handleSendMessage} disabled={!input.trim() || isLoading} className="h-12 w-12 bg-gray-900 hover:bg-[#00A67E] text-white rounded-2xl flex items-center justify-center transition-all duration-300 hover:rotate-12 disabled:bg-gray-200 disabled:text-gray-400 disabled:rotate-0 disabled:cursor-not-allowed shadow-lg cursor-pointer"><Send size={15} /></div>
             </div>
             <p className="text-center text-[10px] text-gray-400 font-medium mt-3">AI can make mistakes. Verify information from sources.</p>
          </div>
        </div>
      </div>

      {selectedSource && (
        <div className="fixed inset-0 bg-gray-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-4 md:p-10 animate-in fade-in duration-200" onClick={(e) => e.target === e.currentTarget && setSelectedSource(null)}>
          <div className="bg-white w-full max-w-4xl h-[85vh] rounded-3xl shadow-2xl flex flex-col overflow-hidden ring-1 ring-white/50 scale-100 animate-in zoom-in-95 duration-300">
            <div className="p-5 border-b border-gray-100 flex justify-between items-center bg-gray-50/50">
               <div className="flex items-center gap-4">
                 <div className="w-12 h-12 bg-white rounded-2xl border border-gray-100 shadow-sm flex items-center justify-center"><FileText className="text-[#00A67E]" size={24}/></div>
                 <div><h3 className="font-bold text-gray-900 text-lg leading-tight">{selectedSource.filename}</h3><div className="flex items-center gap-2 mt-1"><span className="text-xs font-semibold bg-gray-100 text-gray-600 px-2 py-0.5 rounded-md">Page {selectedSource.page}</span></div></div>
               </div>
               <div onClick={() => setSelectedSource(null)} className="w-10 h-10 flex items-center justify-center rounded-xl bg-gray-100 hover:bg-gray-200 text-gray-500 transition-colors"><X size={20}/></div>
            </div>
            <div className="flex-1 overflow-y-auto p-8 bg-white">
               {selectedSource.images?.map((img, i) => <div key={i} className="mb-8 p-2 bg-gray-50 rounded-2xl border border-gray-100 shadow-inner"><img src={img} className="w-full rounded-xl" /></div>)}
               <div className="prose prose-sm max-w-none"><h4 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Extracted Text Content</h4><div className="text-gray-700 text-lg leading-8 font-serif bg-[#FDFBF7] p-8 rounded-xl border border-[#F5EFE6]">{selectedSource.content_snippet}</div></div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}