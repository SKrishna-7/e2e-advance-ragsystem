import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, Plus, MoreHorizontal, Search, Filter, User, Bot, 
  ThumbsUp, ThumbsDown, Copy, RefreshCw, Mic, Sparkles, 
  FileText, X, Settings, Folder, Menu, Sliders, Database, 
  Save, Loader2, BookOpen, UploadCloud
} from 'lucide-react';

// FIX: Use Env Var for Docker compatibility with safe fallback.
// We use a conditional check to avoid crashes in environments where import.meta.env is undefined.
const API_URL = (import.meta.env && import.meta.env.VITE_API_URL) || "http://localhost:000";

export default function App() {
  // --- STATE ---
  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: 'Hello! I am your AI Assistant. Upload a PDF to get started.',
    intent: 'GREETING',
    timestamp: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
  }]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  
  // Modals & Data
  const [selectedSource, setSelectedSource] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [stats, setStats] = useState({ total_files: 0, total_pages: 0, files: [] });
  const [settings, setSettings] = useState({
    chunk_size: 1024, chunk_overlap: 200, top_k_retrieval: 15, top_k_rerank: 5, temperature: 0.0
  });

  // FIX: Generate unique ID per browser session so users don't see each other's chats
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
    // Polling stats (only if window is visible to save server resources)
    const interval = setInterval(() => {
      if (!document.hidden) {
        fetch(`${API_URL}/stats`).then(r => r.ok && r.json().then(setStats)).catch(()=>{});
      }
    }, 8000);
    return () => clearInterval(interval);
  }, []);

  // --- API CALLS ---
  const fetchData = async () => {
    try {
      // settings endpoint might not exist in your new backend, handled gracefully
      const sRes = await fetch(`${API_URL}/stats`);
      if (sRes.ok) setStats(await sRes.json());
    } catch (e) { console.error("Init fetch failed", e); }
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { role: 'user', content: input, timestamp: new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) };
    setMessages(p => [...p, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question: userMsg.content, 
          thread_id: threadId,
          user_id: threadId 
        })
      });
      
      if (!res.ok) throw new Error("API Error");
      const data = await res.json();
      
      setMessages(p => [...p, {
        role: 'assistant', 
        content: data.answer, 
        sources: data.sources, 
        intent: data.intent,
        timestamp: new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})
      }]);
    } catch (e) {
      setMessages(p => [...p, { role: 'assistant', content: "Server error. Is the backend running?", isError: true }]);
    } finally { setIsLoading(false); }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setIsUploading(true);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch(`${API_URL}/upload`, { method: 'POST', body: fd });
      if (res.ok) {
        setMessages(p => [...p, { role: 'system', content: `✅ Uploaded ${file.name}. Processing...` }]);
        setTimeout(fetchData, 2000);
      } else {
        throw new Error("Upload failed");
      }
    } catch (e) { 
      alert("Upload failed. Check backend logs."); 
    } finally { 
      setIsUploading(false); 
      if (fileInputRef.current) fileInputRef.current.value=""; 
    }
  };

  // --- UI RENDER ---
  return (
    <div className="flex h-dvh w-dvw  bg-[#F9FAFB] font-sans text-gray-800 overflow-hidden selection:bg-[#00A67E]/20">
      
      {/* Mobile Overlay */}
      {isSidebarOpen && <div className="fixed inset-0 bg-black/40 z-20 md:hidden backdrop-blur-sm" onClick={() => setIsSidebarOpen(false)} />}

      {/* --- SIDEBAR --- */}
      <div className={`fixed inset-y-0 left-0 z-30 w-72 bg-white border-r border-gray-200 flex flex-col shadow-xl transition-transform duration-300 md:relative md:translate-x-0 ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        
        {/* Header */}
        <div className="p-5 flex items-center justify-between border-b border-gray-50 mt-10">
          <p className="text-xl font-bold text-gray-800 tracking-tight flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-[#00A67E]" />
            Agentic RAG
          </p>
          <button className="md:hidden p-2 text-gray-400 hover:bg-gray-100 rounded-lg" onClick={() => setIsSidebarOpen(false)}><X size={20} /></button>
        </div>

        {/* Upload Button */}
        <div className="p-5">
          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="w-full flex items-center justify-center gap-2 bg-[#00A67E] hover:bg-[#008c6b] text-white text-sm font-medium py-3 rounded-xl shadow-lg shadow-[#00A67E]/20 transition-all active:scale-[0.98]"
          >
            {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <UploadCloud className="w-4 h-4" />}
            {isUploading ? "Uploading..." : "New Document"}
          </button>
          <input type="file" ref={fileInputRef} className="hidden" accept=".pdf" onChange={handleFileUpload} />
        </div>

        {/* Stats Panel */}
        <div className="px-5 pb-5 flex-1 overflow-y-auto">
          <div className="mb-6">
            <h3 className="flex items-center gap-2 text-[11px] font-bold text-gray-400 uppercase tracking-wider mb-4"><Database size={12} /> Knowledge Base</h3>
            
            {/* Quick Stats */}
            <div className="flex gap-2 mb-4">
              <div className="flex-1 bg-gray-50 p-3 rounded-lg border border-gray-100 text-center">
                <div className="text-xl font-bold text-gray-800">{stats.total_files}</div>
                <div className="text-[10px] text-gray-500 font-medium uppercase">Docs</div>
              </div>
              <div className="flex-1 bg-gray-50 p-3 rounded-lg border border-gray-100 text-center">
                <div className="text-xl font-bold text-gray-800">{stats.total_pages}</div>
                <div className="text-[10px] text-gray-500 font-medium uppercase">Pages</div>
              </div>
            </div>

            {/* File List */}
            <div className="space-y-1">
              {stats.files.map((f, i) => (
                <div key={i} className="flex justify-between items-center p-2 hover:bg-gray-50 rounded-lg group transition-colors border border-transparent hover:border-gray-100 cursor-default">
                   <div className="flex items-center gap-2 overflow-hidden">
                     <FileText size={14} className="text-gray-400 group-hover:text-[#00A67E]" />
                     <span className="text-xs text-gray-600 truncate max-w-[120px]" title={f.name}>{f.name}</span>
                   </div>
                   <span className="text-[10px] bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded font-medium">{f.pages}p</span>
                </div>
              ))}
              {stats.files.length === 0 && (
                <div className="text-center py-8 border-2 border-dashed border-gray-100 rounded-xl">
                  <p className="text-xs text-gray-400">No documents yet.</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Bottom Nav */}
        <div className="p-4 border-t border-gray-100 bg-gray-50 flex justify-between items-center text-gray-500">
           <div className="flex items-center gap-2">
             <div className="relative">
               <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"/>
               <div className="absolute inset-0 w-2 h-2 bg-green-500 rounded-full animate-ping opacity-75"/>
             </div>
             <span className="text-xs font-medium">System Online</span>
           </div>
        </div>
      </div>

      {/* --- MAIN CHAT --- */}
      <div className="flex-1 flex flex-col relative bg-white w-full">
        {/* Header */}
        <div className="h-16 border-b border-gray-100 flex items-center justify-between px-4 md:px-6 bg-white z-10 sticky top-0">
          <div className="flex items-center gap-3">
             <button onClick={() => setIsSidebarOpen(true)} className="md:hidden p-2 -ml-2 text-gray-500 hover:bg-gray-50 rounded-lg"><Menu size={24}/></button>
             <h2 className="font-bold text-lg text-gray-800"></h2>
          </div>
          
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 pb-32 scrollbar-thin scrollbar-thumb-gray-200 scrollbar-track-transparent mb-20">
          {messages.map((msg, i) => (
            <div key={i} className="flex gap-4 max-w-3xl mx-auto group animate-in fade-in slide-in-from-bottom-2 duration-300">
              <div className={`w-8 h-8 rounded-xl flex-shrink-0 flex items-center justify-center shadow-sm ${msg.role === 'assistant' ? 'bg-[#00A67E] text-white' : 'bg-gray-100 text-gray-600'}`}>
                 {msg.role === 'assistant' ? <Bot size={18} /> : <User size={18} />}
              </div>
              <div className="flex-1 space-y-2 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-sm text-gray-900">{msg.role === 'assistant' ? 'AI Assistant' : 'You'}</span> 
                  <span className="text-[10px] text-gray-400 font-medium">{msg.timestamp}</span>
                </div>
                
                {msg.role === 'system' ? (
                  <span className="inline-flex items-center gap-2 text-xs bg-gray-50 border border-gray-200 text-gray-600 px-3 py-1.5 rounded-full font-medium">
                    <Loader2 size={12} className="animate-spin"/> {msg.content}
                  </span>
                ) : (
                  <div className={`p-5 rounded-2xl ${msg.role === 'assistant' ? ' text-gray-800 rounded-tl-none' : 'text-gray-800 bg-white border border-gray-100 shadow-sm rounded-tr-none'}`}>
                    <div className="whitespace-pre-wrap text-[15px] leading-7">{msg.content}</div>
                    
                    {/* Better Image Grid */}
                    {msg.sources?.some(s => s.images?.length > 0) && (
                      <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 gap-2">
                        {msg.sources.flatMap(s => s.images).map((img, k) => (
                           <div key={k} className="relative aspect-video rounded-lg overflow-hidden border border-[#00A67E]/20 bg-white shadow-sm cursor-zoom-in group/img transition-transform hover:scale-[1.02]">
                             <img src={img} className="w-full h-full object-contain p-1" />
                             <div className="absolute inset-0 bg-black/0 group-hover/img:bg-black/5 transition-colors"/>
                           </div>
                        ))}
                      </div>
                    )}

                    {/* Citations */}
                    {msg.sources?.length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-4 pt-3 border-t border-[#00A67E]/20">
                        {msg.sources.map((src, k) => (
                          <div key={k} onClick={() => setSelectedSource(src)} className="flex items-center gap-1.5 px-3 py-2 text-white text-[10px] font-light bg-black rounded-[10px] justify-center cursor-pointer transition-transform hover:scale-105">
                            <BookOpen size={15}/><p className='text-sm font-light text-center'> page: {src.page} </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                
              </div>
            </div>
          ))}
          {isLoading && <div className="flex gap-4 max-w-3xl mx-auto pl-12"><div className="flex items-center gap-1 bg-gray-100 px-4 py-2 rounded-full"><span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"/><span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce delay-75"/><span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce delay-150"/></div></div>}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="absolute bottom-0 left-0 right-0 p-4 md:p-6 bg-gradient-to-t from-white via-white to-transparent pt-10">
          <div className="max-w-3xl mx-auto relative shadow-2xl rounded-2xl bg-white border border-gray-200 focus-within:border-[#00A67E] focus-within:ring-4 focus-within:ring-[#00A67E]/10 transition-all duration-300">
             <div className="absolute left-4 top-1/2 -translate-y-1/2 text-[#00A67E]">
               <Sparkles size={18} />
             </div>
             <input 
                value={input} 
                onChange={(e) => setInput(e.target.value)} 
                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()} 
                placeholder="Ask anything about your documents..." 
                className="w-full bg-transparent text-gray-800 placeholder-gray-400 rounded-2xl pl-12 pr-14 py-4 outline-none text-[15px]" 
                disabled={isLoading}
             />
             <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
              
               <button onClick={handleSendMessage} disabled={!input.trim() || isLoading} className="p-2 bg-[#00A67E] text-white rounded-xl hover:bg-[#008c6b] disabled:bg-gray-100 disabled:text-gray-300 transition-all shadow-md active:scale-95"><Send size={18}/></button>
             </div>
          </div>
          <div className="text-center mt-3 text-[11px] text-gray-400 font-medium">
             AI can make mistakes. Please verify important information.
          </div>
        </div>
      </div>

      {/* --- CITATION MODAL --- */}
      {selectedSource && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-in fade-in duration-200" onClick={(e) => e.target === e.currentTarget && setSelectedSource(null)}>
          <div className="bg-white w-full max-w-3xl max-h-[85vh] rounded-2xl shadow-2xl flex flex-col overflow-hidden border border-gray-100 scale-100 animate-in zoom-in-95 duration-200">
            <div className="p-4 border-b border-gray-100 flex justify-between bg-gray-50 items-center">
               <div className="flex gap-3 items-center">
                 <div className="p-2 bg-white border border-gray-200 rounded-lg shadow-sm"><FileText className="text-[#00A67E]" size={20}/></div>
                 <div><h3 className="font-semibold text-sm text-gray-800">{selectedSource.filename}</h3><span className="text-xs text-gray-500 font-medium">Page {selectedSource.page} • Context Viewer</span></div>
               </div>
               <button onClick={() => setSelectedSource(null)} className="p-2 hover:bg-white hover:shadow-sm rounded-full transition-all"><X className="text-gray-400 hover:text-gray-600" size={20}/></button>
            </div>
            <div className="p-8 overflow-y-auto text-gray-700 leading-relaxed font-serif text-[17px]">
               {selectedSource.images?.map((img, i) => <img key={i} src={img} className="w-full mb-6 rounded-lg border border-gray-100 shadow-md"/>)}
               <div className="whitespace-pre-wrap">{selectedSource.content_snippet}</div>
            </div>
            <div className="p-4 border-t border-gray-100 bg-gray-50 flex justify-end">
              <button onClick={() => setSelectedSource(null)} className="px-6 py-2.5 bg-gray-900 text-white rounded-xl text-sm font-medium hover:bg-black transition-colors shadow-lg shadow-gray-900/10">Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}