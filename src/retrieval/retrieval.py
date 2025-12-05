




# import logging
# from typing import List
# import string

# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.documents import Document
# from sentence_transformers import CrossEncoder

# from langchain_classic.retrievers import EnsembleRetriever
# from langchain_community.retrievers import BM25Retriever
# import chromadb

# from src.utils.config import RetrievalConfig
# from logger.logger_config import get_logger
# import torch
# logger = get_logger("Retriever_LOG")

# class RetrievalEngine:
#     def __init__(self, config: RetrievalConfig = RetrievalConfig()):
#         self.config = config
        
#         DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Debug print to see what's actually picked
#         print(f"🖥️  Running on Device: {DEVICE}")
#         # 1. Initialize Embedding Model
#         logger.info(f"Loading Embedding Model: {config.EMBEDDING_MODEL}")
#         self.embedding_model = HuggingFaceEmbeddings(
#             model_name=config.EMBEDDING_MODEL,
#             model_kwargs={'device': config.DEVICE},
#             encode_kwargs={'normalize_embeddings': True},
#         )
        
#         # 2. Connect to ChromaDB (Dense Store)
#         logger.info(f"Connecting to ChromaDB at: {config.CHROMA_PATH}")
#         self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        
#         self.vector_store = Chroma(
#             client=self.chroma_client,
#             collection_name=config.COLLECTION_NAME,
#             embedding_function=self.embedding_model
#         )
        
#         # 3. Initialize Hybrid System (Dense + Sparse)
#         self.ensemble_retriever = self._build_hybrid_retriever()
        
#         # 4. Initialize Reranker
#         logger.info(f"Loading Reranker Model: {config.RERANKER_MODEL}")
#         self.reranker = CrossEncoder(
#             config.RERANKER_MODEL, 
#             device=config.DEVICE,
#             max_length=512,
#             # cache_folder=config.CACHE_DIR
#         )

#     def _preprocess_func(self, text: str) -> List[str]:
#         """
#         Simple text normalization for BM25.
#         Converts to lowercase and removes punctuation.
#         """
#         return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

#     def _build_hybrid_retriever(self):
#         """
#         Constructs an EnsembleRetriever combining:
#         1. Chroma (MMR Search for Diversity)
#         2. BM25 (Sparse Keyword Search)
#         """
#         logger.info("Building Hybrid Retriever with MMR...")
        
#         # A. Dense Retriever (Vector) - UPDATED TO MMR
#         # MMR (Maximal Marginal Relevance) is critical for multi-doc retrieval.
#         # It balances relevance (similarity) with diversity (novelty).
#         dense_retriever = self.vector_store.as_retriever(
#             search_type="mmr",
#             search_kwargs={
#                 "k": self.config.TOP_K_RETRIEVAL,
#                 "fetch_k": self.config.TOP_K_RETRIEVAL * 2, # Fetch more to filter from
#                 "lambda_mult": 0.5 # 0.5 balances diversity vs relevance
#             }
#         )
        
#         # B. Sparse Retriever (BM25)
#         # Fetch all docs from Chroma to build BM25 index in-memory
#         try:
#              # --- SAFETY CHECK: Prevent Memory Overflow ---
#             # BM25Retriever.from_documents loads ALL text into RAM to build the index.
#             # If we have > 10k docs, this might crash the server on startup.
#             # We access the underlying collection to get the count first.
#             doc_count = self.vector_store._collection.count()
#             MAX_DOCS_FOR_BM25 = 10000 
            
#             if doc_count == 0:
#                 logger.warning("ChromaDB is empty! BM25 disabled, using Dense only.")
#                 return dense_retriever
                
#             if doc_count > MAX_DOCS_FOR_BM25:
#                 logger.warning(
#                     f"⚠️ Document count ({doc_count}) exceeds memory safety limit for in-memory BM25 "
#                     f"({MAX_DOCS_FOR_BM25}). Reverting to Dense-Only search to prevent crash."
#                 )
#                 return dense_retriever

#             # If safe, Fetch all docs from Chroma to build BM25 index in-memory
#             logger.info(f"Fetching {doc_count} documents for BM25 indexing...")
#             all_docs_data = self.vector_store.get() 
#             texts = all_docs_data['documents']
#             metadatas = all_docs_data['metadatas']
            
#             docs_for_bm25 = []
#             for t, m in zip(texts, metadatas):
#                 meta = m if m is not None else {}
#                 docs_for_bm25.append(Document(page_content=t, metadata=meta))
            
#             # Initialize BM25 with text normalization
#             bm25_retriever = BM25Retriever.from_documents(
#                 docs_for_bm25, 
#                 preprocess_func=self._preprocess_func
#             )
#             bm25_retriever.k = self.config.TOP_K_RETRIEVAL
            
#             # C. Ensemble
#             # Typical weights: 0.5/0.5 or 0.7 (Dense) / 0.3 (Sparse)
#             ensemble_retriever = EnsembleRetriever(
#                 retrievers=[dense_retriever, bm25_retriever],
#                 weights=self.config.HYBRID_WEIGHTS
#             )
#             logger.info("✅ Hybrid Retriever built successfully.")
#             return ensemble_retriever
#             return ensemble_retriever
            
#         except Exception as e:
#             logger.error(f"Failed to build BM25: {e}")
#             return dense_retriever

#     def search(self, query: str) -> List[Document]:
#         """
#         Execute Hybrid Search -> Deduplicate -> Debug Print -> Rerank
#         """
#         logger.info(f"Processing Query: '{query}'")
        
#         # --- Step 1: Hybrid Retrieval ---
#         try:
#             raw_docs = self.ensemble_retriever.invoke(query)
#         except Exception as e:
#             logger.error(f"Hybrid retrieval failed: {e}")
#             return []
        
#         # --- Step 1.5: Deduplication ---
#         # Ensemble might return the same document from both retrievers.
#         # We deduplicate based on (Source File + Page Number)
#         unique_docs_map = {}
#         for doc in raw_docs:
#             source = doc.metadata.get("filename", "unknown")
#             page = doc.metadata.get("page", "unknown")
#             # Create a unique key for this chunk content location
#             key = f"{source}_{page}"
            
#             # If we haven't seen this chunk, or if this version is longer/better (optional logic), keep it
#             if key not in unique_docs_map:
#                 unique_docs_map[key] = doc
        
#         initial_docs = list(unique_docs_map.values())

#         print("\n" + "="*80)
#         print(f"")
#         print(f" 🔹 PHASE 1: HYBRID RETRIEVAL ({len(initial_docs)} Unique Candidates)")
#         print("="*80)
        
#         # Print Phase 1 Results
#         for i, doc in enumerate(initial_docs[:5]): 
#             source = doc.metadata.get("filename", "Unknown")
#             page = doc.metadata.get("page", "??")
#             print(f"[{i+1}] Page {page} | {source}")

#         # --- Step 2: Reranking ---
#         if not initial_docs:
#             return []
            
#         ranked_docs = self._rerank_documents(query, initial_docs)
        
#         print("\n" + "="*80)
#         print(f" 🔸 PHASE 2: RERANKED RESULTS (Top {self.config.TOP_K_RERANK})")
#         print("="*80)
        
#         for i, doc in enumerate(ranked_docs):
#             score = doc.metadata.get("relevance_score", 0.0)
#             source = doc.metadata.get("filename", "Unknown")
#             page = doc.metadata.get("page", "?")
            
#             snippet = doc.page_content.replace('\n', ' ').strip()[:150]
            
#             print(f"[{i+1}] Score: {score:.4f} | Page: {page} | File: {source}")
#             print(f"    Context: \"{snippet}...\"\n")
            
#         return ranked_docs

#     def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
#         """
#         Score and sort documents using the CrossEncoder.
#         """
#         if not docs:
#             return []
            
#         # Create pairs [Query, Content]
#         pairs = [[query, doc.page_content] for doc in docs]
        
#         # Get Scores (logits)
#         scores = self.reranker.predict(pairs)
        
#         # Attach score and sort
#         doc_score_pairs = list(zip(docs, scores))
#         sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
#         # Select Top K
#         final_results = []
#         for doc, score in sorted_pairs[:self.config.TOP_K_RERANK]:
#             doc.metadata["relevance_score"] = float(score)
#             final_results.append(doc)
            
#         return final_results

# if __name__ == "__main__":
#     # Test
#     engine = RetrievalEngine()
#     test_query = "What is Recurrent Networks ? "
#     results = engine.search(test_query)

















import logging
import time  # <--- ADDED FOR TIMING
from typing import List
import string
import torch # <--- ENSURE TORCH IS IMPORTED
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import chromadb

from src.utils.config import RetrievalConfig
from logger.logger_config import get_logger

logger = get_logger("Retriever_LOG")

class RetrievalEngine:
    def __init__(self, config: RetrievalConfig = RetrievalConfig()):
        self.config = config
        
        # --- 🚨 CRITICAL FIX: FORCE DEVICE DETECTION ---
        # Your previous code defined 'DEVICE' locally but didn't update the config!
        detected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config.DEVICE = detected_device 
        
        print("\n" + "="*50)
        print(f"🚀 INITIALIZING RAG ENGINE")
        print(f"🖥️  Actual Running Device: {self.config.DEVICE.upper()}")
        if self.config.DEVICE == "cpu":
            print("⚠️  WARNING: Running on CPU. Latency will be high!")
        print("="*50 + "\n")

        # 1. Initialize Embedding Model
        logger.info(f"Loading Embedding Model: {config.EMBEDDING_MODEL}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': self.config.DEVICE}, # Use the updated config
            encode_kwargs={'normalize_embeddings': True},
        )
        
        # 2. Connect to ChromaDB (Dense Store)
        logger.info(f"Connecting to ChromaDB at: {config.CHROMA_PATH}")
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embedding_model
        )
        
        # 3. Initialize Hybrid System (Dense + Sparse)
        # We time this because BM25 building is slow
        t0 = time.time()
        self.ensemble_retriever = self._build_hybrid_retriever()
        logger.info(f"⏱️ BM25/Hybrid Index Build Time: {time.time() - t0:.2f}s")
        
        # 4. Initialize Reranker
        logger.info(f"Loading Reranker Model: {config.RERANKER_MODEL}")
        self.reranker = CrossEncoder(
            config.RERANKER_MODEL, 
            device=self.config.DEVICE, # Use the updated config
            max_length=512,
        )

    def _preprocess_func(self, text: str) -> List[str]:
        return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

    def _build_hybrid_retriever(self):
        logger.info("Building Hybrid Retriever with MMR...")
        
        dense_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.config.TOP_K_RETRIEVAL,
                "fetch_k": self.config.TOP_K_RETRIEVAL * 2,
                "lambda_mult": 0.5
            }
        )
        
        try:
            doc_count = self.vector_store._collection.count()
            MAX_DOCS_FOR_BM25 = 10000 
            
            if doc_count == 0:
                logger.warning("ChromaDB is empty! BM25 disabled.")
                return dense_retriever
                
            if doc_count > MAX_DOCS_FOR_BM25:
                logger.warning(f"⚠️ Doc count {doc_count} too high for RAM. Using Dense only.")
                return dense_retriever

            logger.info(f"Fetching {doc_count} documents for BM25 indexing...")
            all_docs_data = self.vector_store.get() 
            texts = all_docs_data['documents']
            metadatas = all_docs_data['metadatas']
            
            docs_for_bm25 = [
                Document(page_content=t, metadata=m if m else {}) 
                for t, m in zip(texts, metadatas)
            ]
            
            bm25_retriever = BM25Retriever.from_documents(
                docs_for_bm25, 
                preprocess_func=self._preprocess_func
            )
            bm25_retriever.k = self.config.TOP_K_RETRIEVAL
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[dense_retriever, bm25_retriever],
                weights=self.config.HYBRID_WEIGHTS
            )
            return ensemble_retriever
            
        except Exception as e:
            logger.error(f"Failed to build BM25: {e}")
            return dense_retriever

    def search(self, query: str) -> List[Document]:
        """
        Execute Hybrid Search -> Deduplicate -> Debug Print -> Rerank
        """
        start_total = time.time()
        logger.info(f"Processing Query: '{query}'")
        
        # --- Step 1: Hybrid Retrieval ---
        t0 = time.time()
        try:
            raw_docs = self.ensemble_retriever.invoke(query)
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []
        t1 = time.time()
        
        # Deduplication
        unique_docs_map = {}
        for doc in raw_docs:
            source = doc.metadata.get("filename", "unknown")
            page = doc.metadata.get("page", "unknown")
            key = f"{source}_{page}"
            if key not in unique_docs_map:
                unique_docs_map[key] = doc
        
        initial_docs = list(unique_docs_map.values())

        print("\n" + "="*80)
        print(f" 🔹 PHASE 1: HYBRID RETRIEVAL ({len(initial_docs)} Candidates) | Time: {t1-t0:.4f}s")
        print("="*80)
        
        # --- Step 2: Reranking ---
        if not initial_docs:
            return []
            
        t2 = time.time()
        ranked_docs = self._rerank_documents(query, initial_docs)
        t3 = time.time()
        
        print("\n" + "="*80)
        print(f" 🔸 PHASE 2: RERANKED RESULTS | Time: {t3-t2:.4f}s")
        print("="*80)
        
        for i, doc in enumerate(ranked_docs):
            score = doc.metadata.get("relevance_score", 0.0)
            source = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page", "?")
            snippet = doc.page_content.replace('\n', ' ').strip()[:150]
            print(f"[{i+1}] Score: {score:.4f} | Page: {page} | File: {source}")
            print(f"    Context: \"{snippet}...\"\n")
            
        print(f"⏱️  TOTAL PIPELINE TIME: {time.time() - start_total:.4f}s")
        return ranked_docs


    # def search(self, query: str) -> List[Document]:
    #     import time
    #     logger.info(f"Processing Query: '{query}'")
        
    #     # --- DIAGNOSTIC START ---
        
    #     # 1. Test Embedding Speed (Is the model slow?)
    #     t_start = time.time()
    #     query_embedding = self.embedding_model.embed_query(query)
    #     t_embed = time.time()
        
    #     # 2. Test Vector Search Speed (Is the DB slow?)
    #     # We pass the pre-computed embedding directly to skip re-embedding
    #     results = self.vector_store.similarity_search_by_vector(query_embedding, k=5)
    #     t_search = time.time()
        
    #     print("\n" + "="*40)
    #     print(f"🕵️ DIAGNOSTIC REPORT")
    #     print(f"----------------------------------------")
    #     print(f"🔤 Embedding Time:  {t_embed - t_start:.4f}s")
    #     print(f"🔍 DB Search Time:  {t_search - t_embed:.4f}s")
    #     print(f"----------------------------------------")
    #     print(f"⚡ TOTAL TIME:      {t_search - t_start:.4f}s")
    #     print("="*40 + "\n")
        
    #     return results
    
    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs: return []
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        doc_score_pairs = list(zip(docs, scores))
        sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        final_results = []
        for doc, score in sorted_pairs[:self.config.TOP_K_RERANK]:
            doc.metadata["relevance_score"] = float(score)
            final_results.append(doc)
        return final_results

if __name__ == "__main__":
    # Test
    try:
        engine = RetrievalEngine()
        test_query = "What is Recurrent Networks?"
        results = engine.search(test_query)
    except Exception as e:
        logger.error(f"Runtime Error: {e}")