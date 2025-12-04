import uvicorn
import shutil
import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi import Form
# --- INTERNAL MODULES ---
from src.graph.rag_graph import RAGGraph
from src.ingestion.data_ingestion import DataIngestionPipeline
from src.utils.config import ModelConfig
from logger.logger_config import get_logger

from dotenv import load_dotenv
import time


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
        # NOTE: If this fails, we catch it below.
        rag_bot = RAGGraph()
        ingestion_pipeline = DataIngestionPipeline()
        
        logger.info(f"✅ System Core Loaded. Ingestion Config: {ingestion_pipeline.config.CHROMA_PATH}")
    except Exception as e:
        logger.critical(f"❌ CRITICAL INITIALIZATION ERROR: {e}")
        logger.critical("Server cannot start without core components. Exiting...")
        # Force exit so Docker restarts the container instead of hanging in a broken state
        sys.exit(1)
    
    yield
    
    logger.info("🛑 Server Shutting Down...")

# --- APP SETUP ---
app = FastAPI(
    title="Agentic RAG API",
    description="Multi-Modal System with Dynamic Settings & Stats",
    version="3.3.0",
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
    user_id:Optional[str] = "me-1"

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

# --- BACKGROUND TASKS ---
# def process_upload_task(directory: str):
#     """Runs ingestion with the ACTIVE settings."""
#     if not ingestion_pipeline:
#         logger.error("❌ Cannot run ingestion: Pipeline is not initialized.")
#         return

#     logger.info(f"🔄 Starting Ingestion in {directory} with Chunk Size {ingestion_pipeline.config.CHUNK_SIZE}...")
#     try:
#         ingestion_pipeline.run(directory)
#         logger.info("✅ Ingestion Complete.")
#     except Exception as e:
#         logger.error(f"❌ Ingestion Failed: {e}")

# --- ENDPOINTS ---
# def process_upload_task(directory: str):
#     global INGESTION_STATUS
#     if not ingestion_pipeline:
#         return

#     INGESTION_STATUS["is_active"] = True
#     INGESTION_STATUS["step"] = "processing"
    
#     try:
#         # 1. Count files to give better feedback
#         files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
#         file_count = len(files)
#         INGESTION_STATUS["message"] = f"Reading {file_count} document(s)..."
        
#         # 2. Run Pipeline
#         # Note: In a real prod app, you'd pass a callback to .run() for granular updates
#         ingestion_pipeline.run(directory)
        
#         INGESTION_STATUS["message"] = "Indexing Vector Database..."
#         # Simulate a small delay or check if run is done
        
#         INGESTION_STATUS["message"] = "Ingestion Complete!"
#         INGESTION_STATUS["step"] = "complete"
        
#     except Exception as e:
#         logger.error(f"Ingestion failed: {e}")
#         INGESTION_STATUS["message"] = f"Error: {str(e)}"
#         INGESTION_STATUS["step"] = "error"
#     finally:
#         # Keep the "Complete" message visible for 5 seconds, then reset
#         time.sleep(5)
#         INGESTION_STATUS["is_active"] = False
#         INGESTION_STATUS["message"] = "System Ready"
#         INGESTION_STATUS["step"] = "idle"

@app.get("/health")
def health_check():
    status = "online" if rag_bot and ingestion_pipeline else "initializing_or_broken"
    return {"status": status, "model": ModelConfig.OLLAMA_MODEL_NAME}

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
        logger.error(f"Stats Error: {e}")
        return {"total_files": 0, "total_pages": 0, "files": []}

# @app.post("/upload")
# async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
#     if not ingestion_pipeline:
#         raise HTTPException(status_code=503, detail="System is not fully initialized.")

#     try:
#         save_dir = "docs/raw"
#         os.makedirs(save_dir, exist_ok=True)
#         file_path = os.path.join(save_dir, file.filename)
        
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
            
#         background_tasks.add_task(process_upload_task, save_dir)
#         return {"message": f"Uploaded {file.filename}. Processing in background."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

def process_upload_task(directory: str, mode: str):
    """Runs ingestion with the selected mode."""
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

# ... endpoints ...

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
    """
    Deletes a file from the system (DB + Disk).
    """
    if not ingestion_pipeline:
        raise HTTPException(status_code=503, detail="System initializing...")
        
    try:
        success = ingestion_pipeline.delete_document(filename)
        return {"message": f"Deleted {filename}", "status": "success"}
    except Exception as e:
        logger.error(f"Delete Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not rag_bot: 
        raise HTTPException(status_code=503, detail="System initializing or failed to start.")
    
    logger.info(f"📥 Chat Query: {request.question}")
    try:
        inputs = {"question": request.question}
        config = {
            "configurable": {"thread_id": request.thread_id},
            "run_name": "RAG_Chat_Request",
            "metadata": {
                "user_id": request.user_id,
                "thread_id": request.thread_id,
                "environment": "production",
                "source": "api_server"
            }
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

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)