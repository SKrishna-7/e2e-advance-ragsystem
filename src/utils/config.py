import os

class DataIngestionConfig:
    def __init__(self):
        self.CHUNK_SIZE = 1024
        self.CHUNK_OVERLAP = 200
        self.EMBEDDING_MODEL = "Bylaw/BAAI-bge-m3"
        
        # FIX: Cross-platform path handling
        # In Docker, this will map to /app/chroma/chroma_db
        default_chroma = os.path.join("chroma", "chroma_db")
        self.CHROMA_PATH = os.getenv("CHROMA_PATH", default_chroma)
        
        self.COLLECTION_NAME = "rag_production"
        self.DEVICE = os.getenv("DEVICE", "cuda")

class RetrievalConfig:
    def __init__(self):
        default_chroma = os.path.join("chroma", "chroma_db")
        self.CHROMA_PATH = os.getenv("CHROMA_PATH", default_chroma)
        
        self.COLLECTION_NAME = "rag_production"
        
        # Models
        self.EMBEDDING_MODEL = "BAAI/bge-m3"
        # self.RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.RERANKER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        self.DEVICE = os.getenv("DEVICE", "cuda")
        
        # Tuning
        self.TOP_K_RETRIEVAL = 30  
        self.TOP_K_RERANK = 10      
        self.HYBRID_WEIGHTS = [0.5, 0.5] 
        
        # FIX: Remove hardcoded Windows path. 
        # Docker will use the default /root/.cache unless specified.
        self.CACHE_DIR = os.getenv("HF_HOME", None)

class ModelConfig:
    # Options: "groq" or "ollama"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
    
    # Models
    MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma3:1b")
    
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") 
    # Note: host.docker.internal allows Docker to see Ollama on your host machine
    
    TEMP = 0



class CacheConfig:
    def __init__(self):
        # Default to localhost for local testing, 'rag_redis' for Docker
        self.REDIS_URL = "redis://localhost:6379/0"
        self.CACHE_TTL = 3600 * 24  # Cache expires in 24 hours


