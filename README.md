# 🚀 End-to-End Agentic RAG System
### A High-Performance, Dockerized Retrieval-Augmented Generation Pipeline

---

## 1. **Executive Summary**

The **End-to-End Agentic RAG System** - designed for intelligent document retrieval and question answering.  
Unlike standard RAG pipelines that rely on linear vector search, this system implements an **Agentic Workflow** using **LangGraph**, featuring:

- Intelligent query routing  
- Hybrid retrieval (Dense + BM25)  
- Cross-encoder reranking  
- Self-correction loops  
- Streaming responses (SSE)  
- Advanced PDF parsing (OCR, tables, layout) via **Docling**

Fully containerized with Docker, the system integrates a modern **React frontend**, a **FastAPI backend**, and optimized low-latency inference through **Redis caching** and hardware-accelerated embeddings.

---

### **Problems in Standard RAG**
- **Low Accuracy**: Irrelevant chunk retrieval.  
- **High Latency**: >10 seconds due to multiple AI calls.  
- **Static Behavior**: Every query treated the same (Hello vs technical query).

An **Agentic System** that:
- Detects intent (Chat vs Search)  
- Rewrites unclear queries  
- Runs multi-stage retrieval  
- Reranks for relevance  
- Caches for sub-millisecond repeats  

---

## 3. **System Architecture**

### **High-Level Data Flow**

---

## 4. **Complete Tech Stack**

| Component | Technology | Description |
|----------|------------|-------------|
| **Frontend** | React 19, Vite | Fast SPA with streaming support |
| **Styling** | TailwindCSS, Lucide | Modern UI components |
| **Backend API** | FastAPI, Uvicorn | Async, high-performance API layer |
| **Agent Workflow** | LangChain, LangGraph | State-machine-driven graph |
| **Vector DB** | ChromaDB | Persistent embedding store |
| **Caching** | Redis (Alpine) | Millisecond-level caching |
| **Ingestion** | Docling, PyPDF | OCR, tables, layout extraction |
| **Embeddings** | BGE-M3 | Dense vector embeddings |
| **Reranker** | TinyBERT Cross-Encoder | Lightweight ONNX reranking |
| **LLMs** | Groq (Llama-3), Ollama | Cloud + local inference options |
| **DevOps** | Docker, Nginx, GitHub Actions | CI/CD + reverse proxy |

---

## 5. **Detailed Architecture Explanation**

### **5.1 Agentic Graph**
Built using **LangGraph**, the system is a **state machine**, not a linear pipeline.

- **State:** Stores user query, documents, conversation history  
- **Nodes:** Retrieve, grade, rewrite, generate  
- **Conditional Edges:**  
  - If irrelevant → Rewrite query  
  - If casual → Route to Chat  
  - If no results → Fallback response  

---

### **5.2 Hybrid Retrieval Strategy**
A combination of:

#### **Dense Retrieval (Chroma + BGE-M3)**
- Captures semantic similarity  
- Good for conceptual matches  

#### **Sparse Retrieval (BM25)**
- Captures exact keywords  
- Critical for logs, PDFs, technical queries  

#### **Reciprocal Rank Fusion (RRF)**
Combines both lists for balanced relevance.

---

### **5.3 Caching Layer (Redis)**
- Queries hashed and checked  
- Cache Hit → **<50ms** response  
- Cache Miss → Output stored for 24h TTL  

---

### **5.4 Streaming (Server-Sent Events)**
- FastAPI uses Python generators to stream tokens  
- React frontend renders tokens using **ReadableStream**  
- Greatly reduces *Time-To-First-Token* (TTFT)

---

## 6. **Features**

- ⚡ **Instant Caching**  
- 🧠 **Intelligent Query Routing**  
- 📄 **Advanced Docling Ingestion**  
  - OCR for scanned text  
  - Table extraction  
  - Full layout parsing  
- 🎯 **Cross-Encoder Reranking (TinyBERT)**  
- 🌊 **Real-Time Token Streaming**  
- 📊 **Live File Statistics Dashboard**  
- 🐳 **Dockerized Deployment** (`docker-compose up`)

---

## 7. **Model & Pipeline Details**

### **Embedding Model**
- **BAAI/bge-m3**  
- High multilingual & long-context performance  

### **Reranker**
- **cross-encoder/ms-marco-TinyBERT-L-2-v2**  
- ONNX optimized  
- ~15% retrieval accuracy boost  

### **LLM Providers**
- **Groq Llama-3.3-70B** → 300+ tokens/sec  
- **Ollama Gemma-2B** → offline inference  

---

## Author

**Suresh Krishna S**  
AI Engineer | LLM Systems | Retrieval Engineering

- GitHub: https://github.com/SKrishna-7
- LinkedIn: https://www.linkedin.com/in/suresh-krishnan-s/
- Portfolio : https://sureshkrishnan.vercel.app/

---


