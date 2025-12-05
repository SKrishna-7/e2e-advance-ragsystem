import redis
import json
import hashlib
from src.utils.config import CacheConfig
from logger.logger_config import get_logger

logger = get_logger("CACHE")

class CacheManager:
    def __init__(self):
        self.config = CacheConfig()
        try:
            self.client = redis.from_url(self.config.REDIS_URL, decode_responses=True)
            self.client.ping() # Test connection
            logger.info(f"✅ Redis Cache Connected: {self.config.REDIS_URL}")
            self.enabled = True
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed. Caching disabled. Error: {e}")
            self.enabled = False

    def _generate_key(self, text: str) -> str:
        """Creates a unique hash for the question to use as a key."""
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()

    def get_answer(self, question: str):
        if not self.enabled: return None
        
        key = f"rag_answer:{self._generate_key(question)}"
        cached_data = self.client.get(key)
        
        if cached_data:
            logger.info(f"⚡ Cache Hit for: '{question[:30]}...'")
            return json.loads(cached_data)
        return None

    def set_answer(self, question: str, answer: str, sources: list = []):
        if not self.enabled: return
        
        key = f"rag_answer:{self._generate_key(question)}"
        payload = json.dumps({
            "answer": answer,
            "sources": sources, # We cache sources too!
            "cached": True
        })
        
        self.client.setex(key, self.config.CACHE_TTL, payload)