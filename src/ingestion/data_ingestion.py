import os
import hashlib
import base64
import shutil
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional, Literal
from PIL import Image

# Fast Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

import chromadb

# Advanced Loaders (Docling)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice
)
from docling.datamodel.base_models import InputFormat

from src.utils.config import DataIngestionConfig
from logger.logger_config import get_logger

logger = get_logger("DataIngestion_LOG")

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class DataIngestionPipeline:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config
        self.docling_converter = None  # Lazy load this
        
        logger.info(f"Loading Embedding Model: {config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info(f"Initializing ChromaDB at: {config.CHROMA_PATH}")
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    def _get_docling(self) -> DocumentConverter:
        """Lazy initialization of Docling to save RAM if not used."""
        if self.docling_converter:
            return self.docling_converter
            
        logger.info(f"Initializing Docling with device: {self.config.DEVICE}")
        accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CUDA if self.config.DEVICE == "cuda" else AcceleratorDevice.CPU
        )
        pipeline_options = PdfPipelineOptions(accelerator_options=accelerator_options)
        pipeline_options.do_ocr = True 
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.generate_page_images = True 
        pipeline_options.generate_picture_images = True

        self.docling_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        return self.docling_converter

    def run(self, data_dir: str, mode: Literal["fast", "advanced"] = "fast"):
        """
        Main execution method.
        mode="fast": Uses PyPDF (Text only, 10x speed)
        mode="advanced": Uses Docling (OCR, Tables, Images, Slower)
        """
        data_path = Path(data_dir)
        if not data_path.exists(): return

        all_files = [str(p) for p in data_path.glob(f"**/*.pdf")] # Focus on PDF for now
        if not all_files: return

        files_to_process = self._filter_processed_files(all_files)
        if not files_to_process:
            logger.info("✅ All documents are already processed.")
            return

        logger.info(f"🚀 Starting ingestion. Mode: {mode.upper()} | Files: {len(files_to_process)}")

        if mode == "advanced":
            raw_docs = self._load_advanced(files_to_process)
        else:
            raw_docs = self._load_fast(files_to_process)
        
        chunks = self._chunk_documents(raw_docs)
        self._index_documents(chunks)
        
        logger.info("Ingestion Pipeline Completed Successfully.")

    def _load_fast(self, file_paths: List[str]) -> List[LangChainDocument]:
        """Fast extraction using PyPDF. Text only."""
        docs = []
        for fp in file_paths:
            try:
                logger.info(f"⚡ Fast-Parsing: {fp}")
                loader = PyPDFLoader(fp)
                pages = loader.load()
                
                # Normalize Metadata
                for p in pages:
                    p.metadata["filename"] = Path(fp).name
                    p.metadata["source"] = fp
                    # PyPDF uses 'page' (0-indexed). 
                    # Ensure consistency if needed, but usually it's fine.
                docs.extend(pages)
            except Exception as e:
                logger.error(f"Fast parse failed for {fp}: {e}")
        return docs

    def _load_advanced(self, file_paths: List[str]) -> List[LangChainDocument]:
        """Deep extraction using Docling. Includes Images & Layout."""
        converter = self._get_docling()
        langchain_docs = []
        
        for file_path in file_paths:
            try:
                logger.info(f"🧠 Advanced-Parsing (OCR/Layout): {file_path}")
                result = converter.convert(file_path)
                docling_doc = result.document

                page_content_map: Dict[int, str] = {}
                page_images_map: Dict[int, List[str]] = {}
                
                for item, level in docling_doc.iterate_items():
                    if not hasattr(item, 'prov') or not item.prov: continue
                    page_no = item.prov[0].page_no
                    
                    # Image Handling
                    if hasattr(item, "image") and item.image:
                        try:
                            pil_image = None
                            if isinstance(item.image, Image.Image): pil_image = item.image
                            elif hasattr(item.image, "pil_image") and item.image.pil_image: pil_image = item.image.pil_image
                            
                            if pil_image:
                                b64_str = image_to_base64(pil_image)
                                if page_no not in page_images_map: page_images_map[page_no] = []
                                page_images_map[page_no].append(f"data:image/png;base64,{b64_str}")
                        except Exception: pass

                    # Text Handling
                    try:
                        item_text = docling_doc.export_to_markdown(from_element=item, to_element=item)
                    except: item_text = getattr(item, 'text', '')

                    if page_no not in page_content_map: page_content_map[page_no] = ""
                    page_content_map[page_no] += item_text + "\n\n"

                for page_num, content in page_content_map.items():
                    if not content.strip() and page_num not in page_images_map: continue
                    doc = LangChainDocument(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "filename": Path(file_path).name,
                            "page": page_num,
                            "images": page_images_map.get(page_num, []) 
                        }
                    )
                    langchain_docs.append(doc)
            except Exception as e:
                logger.error(f"Advanced parse failed for {file_path}: {e}")
                
        return langchain_docs

    def _chunk_documents(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        logger.info("Chunking documents...")
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        )
        semantic_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile"
        )

        final_chunks = []
        for doc in documents:
            # 1. Try Markdown split first (works best for structured/advanced docs)
            md_chunks = markdown_splitter.split_text(doc.page_content)
            for chunk in md_chunks: chunk.metadata.update(doc.metadata)
            
            # 2. If Markdown split didn't do much (common in Fast mode), use Semantic
            # Or refine the markdown chunks
            try:
                if not md_chunks: # Fallback if no headers found
                    md_chunks = [doc]
                
                semantic_chunks = semantic_splitter.split_documents(md_chunks)
                final_chunks.extend(semantic_chunks)
            except Exception:
                final_chunks.extend(md_chunks)
            
        logger.info(f"Generated {len(final_chunks)} chunks.")
        return final_chunks

    def _index_documents(self, chunks: List[LangChainDocument]):
        if not chunks: return
        logger.info(f"Indexing {len(chunks)} chunks...")
        
        collection = self.chroma_client.get_or_create_collection(
            name=self.config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

        # Batching and Deduplication Logic (Simplified for brevity, keep your existing logic here)
        # ... (Reuse the logic from previous version regarding unique_batch_map) ...
        # For brevity in this response, I'm pasting the critical update part:
        
        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            # Simple Hash for now to allow the code to run
            chunk_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()
            
            meta = chunk.metadata.copy()
            if "images" in meta and isinstance(meta["images"], list):
                meta["images"] = "|||".join(meta["images"])
            
            ids.append(chunk_hash)
            docs.append(chunk.page_content)
            metas.append(meta)

        if ids:
            embeddings = self.embeddings.embed_documents(docs)
            BATCH_SIZE = 50
            for i in range(0, len(ids), BATCH_SIZE):
                collection.upsert(
                    ids=ids[i:i+BATCH_SIZE],
                    embeddings=embeddings[i:i+BATCH_SIZE],
                    metadatas=metas[i:i+BATCH_SIZE],
                    documents=docs[i:i+BATCH_SIZE]
                )

    def _filter_processed_files(self, all_files: List[str]) -> List[str]:
        collection = self.chroma_client.get_or_create_collection(name=self.config.COLLECTION_NAME)
        new_files = []
        for fp in all_files:
            fname = Path(fp).name
            existing = collection.get(where={"filename": fname}, limit=1)
            if not existing['ids']: new_files.append(fp)
        return new_files

    def delete_document(self, filename: str) -> bool:
        try:
            col = self.chroma_client.get_collection(name=self.config.COLLECTION_NAME)
            col.delete(where={"filename": filename})
            fp = Path("docs/raw") / filename
            if fp.exists(): os.remove(fp)
            return True
        except Exception: return False

if __name__ == "__main__":
    pipeline = DataIngestionPipeline()