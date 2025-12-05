import uvicorn
import shutil
import os
import sys
import logging
import json
import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# --- INTERNAL MODULES ---
from src.graph.rag_graph import RAGGraph
from src.ingestion.data_ingestion import DataIngestionPipeline
from src.utils.config import ModelConfig
from logger.logger_config import get_logger

load_dotenv()

# Initialize Logger
logger = get_logger("API_LOG")

# --- GLOBAL STATE ---
rag_bot = None
ingestion_pipeline = None
INGESTION_STATUS = {
    "is_active": False,
    "message": "System Ready",
    "step": "idle" # idle, processing, complete, error
}

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Core AI Engines on Startup"""
    global rag_bot, ingestion_pipeline
    logger.info("🚀 Starting RAG Server...")
    try:
        # Initialize Components
        rag_bot = RAGGraph()
        ingestion_pipeline = DataIngestionPipeline()
        
        logger.info(f"✅ System Core Loaded. Ingestion Config: {ingestion_pipeline.config.CHROMA_PATH}")
    except Exception as e:
        logger.critical(f"❌ CRITICAL INITIALIZATION ERROR: {e}")
        logger.critical("Server cannot start without core components. Exiting...")
        sys.exit(1)
    
    yield
    
    logger.info("🛑 Server Shutting Down...")

# --- APP SETUP ---
app = FastAPI(
    title="Agentic RAG API",
    description="Multi-Modal System with Streaming & Caching",
    version="3.4.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class QueryRequest(BaseModel):
    question: str
    thread_id: Optional[str] = "default_session"
    user_id: Optional[str] = "me-1"

class SourceDoc(BaseModel):
    filename: str
    page: int
    content_snippet: str
    images: List[str] = []

class QueryResponse(BaseModel):
    answer: str
    intent: str
    sources: List[SourceDoc]
    chat_history: List[str] = []

class DocStats(BaseModel):
    total_files: int
    total_pages: int
    files: List[Dict[str, Any]]

# --- HELPER FUNCTIONS ---

def process_upload_task(directory: str, mode: str):
    """Runs ingestion with the selected mode in background."""
    global INGESTION_STATUS
    if not ingestion_pipeline: return

    INGESTION_STATUS["is_active"] = True
    INGESTION_STATUS["step"] = "processing"
    
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        count = len(files)
        
        mode_text = "Fast (Text Only)" if mode == "fast" else "Advanced (OCR + Images)"
        INGESTION_STATUS["message"] = f"Ingesting {count} docs using {mode_text} mode..."
        
        # Pass mode to run
        ingestion_pipeline.run(directory, mode=mode)
        
        INGESTION_STATUS["message"] = "Ingestion Complete!"
        INGESTION_STATUS["step"] = "complete"
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        INGESTION_STATUS["message"] = f"Error: {str(e)}"
        INGESTION_STATUS["step"] = "error"
    finally:
        time.sleep(5)
        INGESTION_STATUS["is_active"] = False
        INGESTION_STATUS["message"] = "System Ready"
        INGESTION_STATUS["step"] = "idle"

# --- ENDPOINTS ---

@app.get("/health")
def health_check():
    status = "online" if rag_bot and ingestion_pipeline else "initializing_or_broken"
    return {"status": status, "model": ModelConfig.OLLAMA_MODEL_NAME}

@app.get("/ingestion/status")
def get_ingestion_status():
    """Endpoint for UI to poll ingestion progress."""
    return INGESTION_STATUS

@app.get("/stats", response_model=DocStats)
def get_stats():
    """Retrieve KB Statistics from ChromaDB."""
    if not ingestion_pipeline: 
        return {"total_files": 0, "total_pages": 0, "files": []}
        
    try:
        collection = ingestion_pipeline.chroma_client.get_collection(
            name=ingestion_pipeline.config.COLLECTION_NAME
        )
        data = collection.get(include=["metadatas"])
        
        unique_files = {}
        for m in data.get("metadatas", []):
            if not m: continue
            fname = m.get("filename", "unknown")
            page = m.get("page", 0)
            if fname not in unique_files: unique_files[fname] = 0
            if page > unique_files[fname]: unique_files[fname] = page
            
        file_list = [{"name": k, "pages": v} for k,v in unique_files.items()]
        
        return {
            "total_files": len(unique_files),
            "total_pages": sum(unique_files.values()),
            "files": file_list
        }
    except Exception as e:
        # It's common for stats to fail if DB is empty, just return 0
        return {"total_files": 0, "total_pages": 0, "files": []}

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    mode: str = Form("fast") # Default to fast if not specified
):
    if not ingestion_pipeline:
        raise HTTPException(status_code=503, detail="System initializing...")
    try:
        save_dir = os.path.join("docs", "raw")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Pass the mode to the background task
        background_tasks.add_task(process_upload_task, save_dir, mode)
        
        return {"message": f"Uploaded {file.filename}. Mode: {mode}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete/{filename}")
def delete_file(filename: str):
    """Deletes a file from the system (DB + Disk)."""
    if not ingestion_pipeline:
        raise HTTPException(status_code=503, detail="System initializing...")
        
    try:
        success = ingestion_pipeline.delete_document(filename)
        return {"message": f"Deleted {filename}", "status": "success"}
    except Exception as e:
        logger.error(f"Delete Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- CHAT ENDPOINTS ---

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """Standard Synchronous Chat Endpoint (Legacy/Fallback)."""
    if not rag_bot: 
        raise HTTPException(status_code=503, detail="System initializing.")
    
    logger.info(f"📥 Chat Query: {request.question}")
    try:
        inputs = {"question": request.question}
        config = {
            "configurable": {"thread_id": request.thread_id},
            "run_name": "RAG_Chat_Request"
        }
        
        result = rag_bot.app.invoke(inputs, config=config)
        
        raw_docs = result.get("documents", [])
        formatted_sources = []
        
        for doc in raw_docs:
            img_data = doc.metadata.get("images", [])
            final_imgs = []
            
            if isinstance(img_data, list): final_imgs = img_data
            elif isinstance(img_data, str) and img_data:
                final_imgs = img_data.split("|||") if "|||" in img_data else [img_data]
                
            formatted_sources.append(SourceDoc(
                filename=doc.metadata.get("filename", "unknown"),
                page=doc.metadata.get("page", -1),
                content_snippet=doc.page_content,
                images=final_imgs
            ))
            
        return QueryResponse(
            answer=result.get("answer", "No response."),
            intent=result.get("intent", "UNKNOWN"),
            sources=formatted_sources,
            chat_history=result.get("chat_history", [])
        )
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream_endpoint(request: QueryRequest):
    """
    Streaming Endpoint (SSE).
    Returns a stream of events:
    1. 'sources': JSON list of citations (emitted first).
    2. 'token': Text chunks of the answer (emitted incrementally).
    """
    if not rag_bot:
        raise HTTPException(status_code=503, detail="System initializing.")

    # 1. Quick Intent Check
    # We do this outside the generator to catch errors early
    try:
        intent = rag_bot.generator.check_intent(request.question)
    except Exception as e:
        logger.error(f"Intent Check Failed: {e}")
        intent = "SEARCH"

    async def event_generator():
        try:
            # --- PATH A: CASUAL CHAT ---
            if intent == "CHAT":
                # Just stream the chat response
                for token in rag_bot.generator.chat_chain.stream({"question": request.question}):
                    payload = json.dumps({"type": "token", "content": token})
                    yield f"data: {payload}\n\n"
                    await asyncio.sleep(0) # Yield control
                yield "data: [DONE]\n\n"
                return

            # --- PATH B: RAG SEARCH ---
            
            # 1. Check Cache First (Manual check for streaming)
            cached = rag_bot.cache.get_answer(request.question)
            if cached:
                # Emit Cached Sources
                sources_payload = json.dumps({"type": "sources", "data": cached.get("sources", [])})
                yield f"data: {sources_payload}\n\n"
                
                # Emit Cached Answer (Simulate stream or send bulk)
                token_payload = json.dumps({"type": "token", "content": cached["answer"]})
                yield f"data: {token_payload}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 2. Retrieve Documents
            # Using the optimized search (Hybrid -> Rerank)
            docs = rag_bot.retriever.search(request.question)
            
            # 3. Format & Send Sources (ASAP)
            formatted_sources = []
            for doc in docs:
                img_data = doc.metadata.get("images", [])
                final_imgs = []
                if isinstance(img_data, list): final_imgs = img_data
                elif isinstance(img_data, str) and img_data:
                    final_imgs = img_data.split("|||") if "|||" in img_data else [img_data]
                
                formatted_sources.append({
                    "filename": doc.metadata.get("filename", "unknown"),
                    "page": doc.metadata.get("page", -1),
                    "content_snippet": doc.page_content[:200],
                    "images": final_imgs
                })
            
            sources_payload = json.dumps({"type": "sources", "data": formatted_sources})
            yield f"data: {sources_payload}\n\n"

            # 4. Generate & Stream Answer
            full_answer = ""
            for token in rag_bot.generator.stream_generate_answer(request.question, docs):
                full_answer += token
                token_payload = json.dumps({"type": "token", "content": token})
                yield f"data: {token_payload}\n\n"
                await asyncio.sleep(0)
            
            # 5. Update Cache (Background)
            if docs:
                rag_bot.cache.set_answer(request.question, full_answer, formatted_sources)

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming Error: {e}")
            err_payload = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)