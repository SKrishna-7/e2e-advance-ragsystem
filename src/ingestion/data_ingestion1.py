# import os
# import hashlib
# from pathlib import Path
# from typing import List, Optional, Dict, Any

# from langchain_core.documents import Document as LangChainDocument
# from langchain_text_splitters import MarkdownHeaderTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_experimental.text_splitter import SemanticChunker

# import chromadb
# # We use the standard client for the type hint, but PersistentClient for implementation
# from chromadb import ClientAPI 

# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.datamodel.pipeline_options import (
#     PdfPipelineOptions,
#     AcceleratorOptions,
#     AcceleratorDevice
# )
# from docling.datamodel.base_models import InputFormat

# # Assumed internal modules (mocked for this context if you don't have them)
# from src.utils.config import PipelineConfig
# from logger.logger_config import get_logger

# logger = get_logger("DataIngestion_LOG")

# class DataIngestionPipeline:
#     def __init__(self, config: PipelineConfig = PipelineConfig()):
#         self.config = config
        
#         # 1. Initialize Docling Converter (With GPU Acceleration)
#         self.converter = self._init_docling()
        
#         # 2. Initialize Embedding Model
#         logger.info(f"Loading Embedding Model: {config.EMBEDDING_MODEL}")
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name=config.EMBEDDING_MODEL,
#             model_kwargs={'device': config.DEVICE},
#             encode_kwargs={'normalize_embeddings': True}
#         )

#         # 3. Initialize Chroma Persistent Client
#         logger.info(f"Initializing ChromaDB at: {config.CHROMA_PATH}")
#         self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)

#     def _init_docling(self) -> DocumentConverter:
#         """
#         Sets up the Docling converter.
#         - PDF: Optimized with CUDA acceleration and OCR.
#         """
#         logger.info(f"Initializing Docling with device: {self.config.DEVICE}")
        
#         accelerator_options = AcceleratorOptions(
#             num_threads=4,
#             device=AcceleratorDevice.CUDA if self.config.DEVICE == "cuda" else AcceleratorDevice.CPU
#         )
        
#         pipeline_options = PdfPipelineOptions(accelerator_options=accelerator_options)
#         pipeline_options.do_ocr = True 
#         pipeline_options.do_table_structure = True
#         pipeline_options.table_structure_options.do_cell_matching = True

#         return DocumentConverter(
#             format_options={
#                 InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
#             }
#         )

#     def run(self, data_dir: str):
#         """
#         Main execution method: Load -> Split -> Embed -> Store
#         """
#         supported_extensions = ["pdf", "docx", "pptx", "html", "md"]
#         files = []
#         for ext in supported_extensions:
#             files.extend([str(p) for p in Path(data_dir).glob(f"**/*.{ext}")])
            
#         if not files:
#             logger.warning(f"No supported documents found in {data_dir}")
#             return

#         logger.info(f"Found {len(files)} documents to process.")

#         # 1. Load (Grouped by Page)
#         raw_docs = self._load_documents(files)
        
#         # 2. Chunk (Semantically, respecting page boundaries)
#         chunks = self._chunk_documents(raw_docs)
        
#         # 3. Index (Deduplicated Upsert)
#         self._index_documents(chunks)
        
#         logger.info("Ingestion Pipeline Completed Successfully.")

#     def _load_documents(self, file_paths: List[str]) -> List[LangChainDocument]:
#         """
#         Parses files using Docling and converts them to LangChain Documents.
#         CRITICAL CHANGE: This now creates one Document *per page* to preserve page numbers.
#         """
#         langchain_docs = []
        
#         for file_path in file_paths:
#             try:
#                 logger.info(f"Parsing: {file_path}")
#                 result = self.converter.convert(file_path)
#                 docling_doc = result.document

#                 # --- NEW LOGIC: Iterating items to group by Page Number ---
#                 # This ensures we know exactly which page every sentence came from.
#                 page_content_map: Dict[int, str] = {}
                
#                 # iterate_items() yields (DocItem, level)
#                 for item, level in docling_doc.iterate_items():
#                     # Skip items without provenance (rare, but possible)
#                     if not item.prov or len(item.prov) == 0:
#                         continue
                    
#                     # Get the page number (Docling pages are 1-indexed)
#                     page_no = item.prov[0].page_no
                    
#                     # Convert this specific item (header/text/table) to Markdown
#                     # We use the document's exporter to handle the item format correctly
#                     try:
#                         # Depending on Docling version, we might extract text or export
#                         # serialize() is the most robust way to get markdown for an item
#                         item_text = docling_doc.export_to_markdown(from_element=item, to_element=item)
#                     except Exception:
#                         # Fallback if specific export fails, just grab text
#                         item_text = getattr(item, 'text', '')

#                     if page_no not in page_content_map:
#                         page_content_map[page_no] = ""
                    
#                     page_content_map[page_no] += item_text + "\n\n"

#                 # --- Construct LangChain Documents (One per Page) ---
#                 for page_num, content in page_content_map.items():
#                     if not content.strip():
#                         continue
                        
#                     doc = LangChainDocument(
#                         page_content=content,
#                         metadata={
#                             "source": file_path,
#                             "filename": Path(file_path).name,
#                             "page": page_num, # <--- THIS IS THE KEY
#                             "total_pages": len(page_content_map)
#                         }
#                     )
#                     langchain_docs.append(doc)
                
#             except Exception as e:
#                 logger.error(f"Failed to parse {file_path}: {e}")
#                 continue
                
#         return langchain_docs

#     def _chunk_documents(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
#         """
#         1. Hierarchical Split: Use Markdown headers.
#         2. Semantic Split: Break text by meaning.
#         """
#         logger.info("Chunking documents (Hybrid: Markdown + Semantic)...")
        
#         # 1. Markdown Splitter
#         headers_to_split_on = [
#             ("#", "Header 1"),
#             ("##", "Header 2"),
#             ("###", "Header 3"),
#         ]
#         markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
#         # 2. Semantic Chunker
#         semantic_splitter = SemanticChunker(
#             embeddings=self.embeddings,
#             breakpoint_threshold_type="percentile", 
#             breakpoint_threshold_amount=95
#         )

#         final_chunks = []
        
#         # We process each "Page Document" individually.
#         # This guarantees that a chunk will never span across Page 1 and Page 2,
#         # keeping the "page" metadata accurate.
#         for doc in documents:
#             # First pass: Markdown structure within the page
#             md_chunks = markdown_splitter.split_text(doc.page_content)
            
#             # Carry over metadata (including PAGE NUMBER) to each markdown chunk
#             for chunk in md_chunks:
#                 chunk.metadata.update(doc.metadata)
            
#             # Second pass: Semantic analysis
#             try:
#                 # Semantic splitter usually takes a list, but we feed it chunks from ONE page
#                 semantic_chunks = semantic_splitter.split_documents(md_chunks)
#                 final_chunks.extend(semantic_chunks)
#             except Exception as e:
#                 # Fallback if text is too short for semantic analysis
#                 logger.warning(f"Semantic split failed for page {doc.metadata.get('page')}, keeping MD chunks.")
#                 final_chunks.extend(md_chunks)
            
#         logger.info(f"Generated {len(final_chunks)} semantic chunks.")
#         return final_chunks

#     def _index_documents(self, chunks: List[LangChainDocument]):
#         """
#         Embeds chunks and upserts them to ChromaDB with Deduplication.
#         """
#         if not chunks:
#             logger.warning("No chunks to index.")
#             return

#         logger.info(f"Indexing {len(chunks)} chunks into ChromaDB with Deduplication...")
        
#         collection = self.chroma_client.get_or_create_collection(
#             name=self.config.COLLECTION_NAME,
#             metadata={"hnsw:space": "cosine"}
#         )

#         # --- Phase 1: Local Batch Aggregation ---
#         unique_batch_map: Dict[str, dict] = {}

#         for chunk in chunks:
#             # Hash content to check for exact duplicates
#             chunk_hash = hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()
#             source_file = chunk.metadata.get("source", "unknown")

#             if chunk_hash not in unique_batch_map:
#                 unique_batch_map[chunk_hash] = {
#                     "page_content": chunk.page_content,
#                     "base_metadata": chunk.metadata.copy(),
#                     "sources_in_batch": {source_file},
#                     "count_in_batch": 1
#                 }
#             else:
#                 unique_batch_map[chunk_hash]["sources_in_batch"].add(source_file)
#                 unique_batch_map[chunk_hash]["count_in_batch"] += 1

#         batch_ids = list(unique_batch_map.keys())
        
#         # --- Phase 2: DB Lookup & Merge ---
#         existing_records = collection.get(ids=batch_ids, include=["metadatas"])
#         existing_meta_map = {
#             id_: meta for id_, meta in zip(existing_records["ids"], existing_records["metadatas"]) 
#             if meta is not None
#         }

#         # --- Phase 3: Prepare Upsert Data ---
#         ids_to_upsert = []
#         docs_to_upsert = []
#         metas_to_upsert = []

#         for chunk_id, batch_data in unique_batch_map.items():
#             final_metadata = batch_data["base_metadata"]
#             new_sources = batch_data["sources_in_batch"]
#             batch_count = batch_data["count_in_batch"]

#             if chunk_id in existing_meta_map:
#                 existing_meta = existing_meta_map[chunk_id]
                
#                 # Merge Counts
#                 prev_count = existing_meta.get("copy_count", 1)
#                 final_metadata["copy_count"] = prev_count + batch_count
                
#                 # Merge Sources
#                 prev_sources_str = existing_meta.get("all_sources", existing_meta.get("source", ""))
#                 prev_sources_set = set(prev_sources_str.split(",")) if prev_sources_str else set()
#                 combined_sources = prev_sources_set.union(new_sources)
#                 final_metadata["all_sources"] = ",".join(sorted(list(combined_sources)))
#             else:
#                 # New Record
#                 final_metadata["copy_count"] = batch_count
#                 final_metadata["all_sources"] = ",".join(sorted(list(new_sources)))

#             # Clean up metadata (Set is not JSON serializable)
#             if "sources_in_batch" in final_metadata: del final_metadata["sources_in_batch"]
            
#             ids_to_upsert.append(chunk_id)
#             docs_to_upsert.append(batch_data["page_content"])
#             metas_to_upsert.append(final_metadata)

#         # --- Phase 4: Embedding & Upsert ---
#         if ids_to_upsert:
#             logger.info(f"Generating embeddings for {len(docs_to_upsert)} unique chunks...")
#             embeddings = self.embeddings.embed_documents(docs_to_upsert)
            
#             collection.upsert(
#                 ids=ids_to_upsert,
#                 embeddings=embeddings,
#                 metadatas=metas_to_upsert,
#                 documents=docs_to_upsert
#             )
#             logger.info(f"Upserted {len(ids_to_upsert)} records. Deduplication active.")

# if __name__ == "__main__":
#     pipeline = DataIngestionPipeline()
#     # Windows path style provided in prompt
#     pipeline.run(r"docs\raw")
















import os
import hashlib
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

import chromadb
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice
)
from docling.datamodel.base_models import InputFormat

# Assumed internal modules
from src.utils.config import DataIngestionConfig
from logger.logger_config import get_logger

logger = get_logger("DataIngestion_LOG")

# --- HELPER FUNCTION FOR IMAGES ---
def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a Base64 string."""
    buffered = BytesIO()
    # Save as PNG to preserve quality/transparency
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class DataIngestionPipeline:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config
        
        # 1. Lazy Initialize Docling
        self.converter = None
        
        # 2. Initialize Embedding Model
        logger.info(f"Loading Embedding Model: {config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 3. Initialize Chroma Persistent Client
        logger.info(f"Initializing ChromaDB at: {config.CHROMA_PATH}")
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    def _init_docling(self) -> DocumentConverter:
        """
        Sets up the Docling converter.
        - PDF: Optimized with CUDA acceleration and OCR.
        """
        logger.info(f"Initializing Docling with device: {self.config.DEVICE}")
        
        accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CUDA if self.config.DEVICE == "cuda" else AcceleratorDevice.CPU
        )
        
        pipeline_options = PdfPipelineOptions(accelerator_options=accelerator_options)
        pipeline_options.do_ocr = True 
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        # Enable image extraction in Docling options if required by your version
        pipeline_options.generate_page_images = True 
        pipeline_options.generate_picture_images = True

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def run(self, data_dir: str):
        """
        Main execution method: Check DB -> Load -> Split -> Embed -> Store
        """
        supported_extensions = ["pdf", "docx", "pptx", "html", "md"]
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.error(f"Directory not found: {data_dir}")
            return

        all_files = []
        for ext in supported_extensions:
            all_files.extend([str(p) for p in data_path.glob(f"**/*.{ext}")])
            
        if not all_files:
            logger.warning(f"No supported documents found in {data_dir}")
            return

        # --- STEP 1: Filter already processed files ---
        files_to_process = self._filter_processed_files(all_files)

        if not files_to_process:
            logger.info("✅ All documents are already processed. Skipping ingestion.")
            return

        logger.info(f"🚀 Starting processing for {len(files_to_process)} new document(s)...")

        # --- STEP 2: Initialize Docling (Only if needed) ---
        if self.converter is None:
            self.converter = self._init_docling()

        # --- STEP 3: Load (Grouped by Page with Images) ---
        raw_docs = self._load_documents(files_to_process)
        
        # --- STEP 4: Chunk ---
        chunks = self._chunk_documents(raw_docs)
        
        # --- STEP 5: Index ---
        self._index_documents(chunks)
        
        logger.info("Ingestion Pipeline Completed Successfully.")

    def delete_document(self, filename: str) -> bool:
        """
        Deletes a document from ChromaDB and the local filesystem.
        """
        logger.info(f"🗑️ Attempting to delete: {filename}")
        
        try:
            # 1. Delete from ChromaDB
            # We target vectors where metadata 'filename' matches
            collection = self.chroma_client.get_collection(name=self.config.COLLECTION_NAME)
            collection.delete(where={"filename": filename})
            logger.info(f"✅ Deleted vectors for {filename} from ChromaDB")
            
            # 2. Delete File from Disk
            # Assuming standard path structure
            file_path = Path("docs/raw") / filename
            if file_path.exists():
                os.remove(file_path)
                logger.info(f"✅ Deleted physical file: {file_path}")
            else:
                logger.warning(f"⚠️ File not found on disk: {file_path}")
                
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting document {filename}: {e}")
            raise e

    def _filter_processed_files(self, all_files: List[str]) -> List[str]:
        collection = self.chroma_client.get_or_create_collection(
            name=self.config.COLLECTION_NAME
        )
        
        new_files = []
        for file_path in all_files:
            filename = Path(file_path).name
            existing = collection.get(
                where={"filename": filename},
                limit=1,
                include=[] 
            )
            
            if existing['ids']:
                print(f"⏩ Skipping: {filename} (Already in DB)")
            else:
                new_files.append(file_path)
                
        return new_files

    def _load_documents(self, file_paths: List[str]) -> List[LangChainDocument]:
        """
        Parses files using Docling and converts them to LangChain Documents.
        NOW INCLUDES IMAGE EXTRACTION (FIXED for ImageRef).
        """
        langchain_docs = []
        
        for file_path in file_paths:
            try:
                logger.info(f"Parsing: {file_path}")
                result = self.converter.convert(file_path)
                docling_doc = result.document

                # Maps to store Content and Images per page
                page_content_map: Dict[int, str] = {}
                page_images_map: Dict[int, List[str]] = {}
                
                # iterate_items() yields (DocItem, level)
                for item, level in docling_doc.iterate_items():
                    
                    if not hasattr(item, 'prov') or not item.prov or len(item.prov) == 0:
                        continue
                    
                    # Get page number
                    page_no = item.prov[0].page_no
                    
                    # --- A. HANDLE IMAGES ---
                    # Check if this item is an image/picture
                    if hasattr(item, "image") and item.image:
                        try:
                            # 1. Resolve the PIL Image from the ImageRef
                            pil_image = None
                            
                            # Case A: It's already a PIL Image (older versions)
                            if isinstance(item.image, Image.Image):
                                pil_image = item.image
                            # Case B: It's an ImageRef wrapper (newer versions)
                            elif hasattr(item.image, "pil_image") and item.image.pil_image:
                                pil_image = item.image.pil_image
                            
                            # 2. Convert if we found a valid image
                            if pil_image:
                                b64_str = image_to_base64(pil_image)
                                full_img_str = f"data:image/png;base64,{b64_str}"
                                
                                if page_no not in page_images_map:
                                    page_images_map[page_no] = []
                                page_images_map[page_no].append(full_img_str)
                            
                        except Exception as img_err:
                            logger.warning(f"Failed to process image on page {page_no}: {img_err}")

                    # --- B. HANDLE TEXT ---
                    try:
                        item_text = docling_doc.export_to_markdown(from_element=item, to_element=item)
                    except Exception:
                        item_text = getattr(item, 'text', '')

                    if page_no not in page_content_map:
                        page_content_map[page_no] = ""
                    
                    page_content_map[page_no] += item_text + "\n\n"

                # --- Construct LangChain Documents ---
                for page_num, content in page_content_map.items():
                    # Only skip if BOTH text and images are empty
                    if not content.strip() and page_num not in page_images_map:
                        continue
                        
                    doc = LangChainDocument(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "filename": Path(file_path).name,
                            "page": page_num,
                            "total_pages": len(page_content_map),
                            # ✅ Store images in metadata
                            "images": page_images_map.get(page_num, []) 
                        }
                    )
                    langchain_docs.append(doc)
                
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                continue
                
        return langchain_docs

    def _chunk_documents(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        logger.info("Chunking documents (Hybrid: Markdown + Semantic)...")
        
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        semantic_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile", 
            breakpoint_threshold_amount=95
        )

        final_chunks = []
        
        for doc in documents:
            md_chunks = markdown_splitter.split_text(doc.page_content)
            
            # Carry over metadata (including IMAGES) to chunks
            for chunk in md_chunks:
                chunk.metadata.update(doc.metadata)
            
            try:
                semantic_chunks = semantic_splitter.split_documents(md_chunks)
                final_chunks.extend(semantic_chunks)
            except Exception:
                final_chunks.extend(md_chunks)
            
        logger.info(f"Generated {len(final_chunks)} semantic chunks.")
        return final_chunks

    def _index_documents(self, chunks: List[LangChainDocument]):
        """
        Embeds chunks and upserts them to ChromaDB with Deduplication.
        """
        if not chunks:
            logger.warning("No chunks to index.")
            return

        logger.info(f"Indexing {len(chunks)} chunks into ChromaDB with Deduplication...")
        
        collection = self.chroma_client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # --- Phase 1: Local Batch Aggregation ---
        unique_batch_map: Dict[str, dict] = {}

        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()
            source_file = chunk.metadata.get("source", "unknown")

            if chunk_hash not in unique_batch_map:
                # IMPORTANT: Deep copy metadata so we don't accidentally mutate the original doc
                meta_copy = chunk.metadata.copy()
                
                # Convert list of images to string (or handle appropriately for Chroma)
                # Chroma doesn't support Lists in metadata well, so we often join them or skip if too large.
                # For this implementation, we will NOT deduplicate image lists here to be safe, 
                # or we convert list to string representation if needed.
                # Best practice: keep 'images' list in the response, but for Chroma metadata, we might need to stringify.
                if "images" in meta_copy and isinstance(meta_copy["images"], list):
                    # WARNING: Storing Base64 in metadata might hit size limits. 
                    # If this crashes, comment out the next line.
                    # For now, we assume it's okay for small-medium docs.
                    pass 

                unique_batch_map[chunk_hash] = {
                    "page_content": chunk.page_content,
                    "base_metadata": meta_copy,
                    "sources_in_batch": {source_file},
                    "count_in_batch": 1
                }
            else:
                unique_batch_map[chunk_hash]["sources_in_batch"].add(source_file)
                unique_batch_map[chunk_hash]["count_in_batch"] += 1

        batch_ids = list(unique_batch_map.keys())
        
        # --- Phase 2: DB Lookup & Merge ---
        existing_records = collection.get(ids=batch_ids, include=["metadatas"])
        existing_meta_map = {
            id_: meta for id_, meta in zip(existing_records["ids"], existing_records["metadatas"]) 
            if meta is not None
        }

        # --- Phase 3: Prepare Upsert Data ---
        ids_to_upsert = []
        docs_to_upsert = []
        metas_to_upsert = []

        for chunk_id, batch_data in unique_batch_map.items():
            final_metadata = batch_data["base_metadata"]
            new_sources = batch_data["sources_in_batch"]
            batch_count = batch_data["count_in_batch"]

            # Handle List Metadata (Images) for Chroma Compatibility
            # ChromaDB metadata values must be int, float, str, or bool. It does NOT support lists.
            # We must convert the list of images to a single string (or ignore it for the DB index).
            if "images" in final_metadata and isinstance(final_metadata["images"], list):
                # We join them with a delimiter or just pick the first one to avoid massive metadata
                # Alternatively, store them as "image_0", "image_1", etc.
                # Here, we will flatten to a string to prevent errors.
                # NOTE: This creates very large metadata values.
                final_metadata["images"] = "|||".join(final_metadata["images"])

            if chunk_id in existing_meta_map:
                existing_meta = existing_meta_map[chunk_id]
                prev_count = existing_meta.get("copy_count", 1)
                final_metadata["copy_count"] = prev_count + batch_count
                
                prev_sources_str = existing_meta.get("all_sources", existing_meta.get("source", ""))
                prev_sources_set = set(prev_sources_str.split(",")) if prev_sources_str else set()
                combined_sources = prev_sources_set.union(new_sources)
                final_metadata["all_sources"] = ",".join(sorted(list(combined_sources)))
            else:
                final_metadata["copy_count"] = batch_count
                final_metadata["all_sources"] = ",".join(sorted(list(new_sources)))

            if "sources_in_batch" in final_metadata: del final_metadata["sources_in_batch"]
            
            ids_to_upsert.append(chunk_id)
            docs_to_upsert.append(batch_data["page_content"])
            metas_to_upsert.append(final_metadata)

        # --- Phase 4: Embedding & Upsert ---
        if ids_to_upsert:
            logger.info(f"Generating embeddings for {len(docs_to_upsert)} unique chunks...")
            embeddings = self.embeddings.embed_documents(docs_to_upsert)
            
            BATCH_SIZE = 100
            for i in range(0, len(ids_to_upsert), BATCH_SIZE):
                batch_ids = ids_to_upsert[i : i + BATCH_SIZE]
                batch_embeds = embeddings[i : i + BATCH_SIZE]
                batch_metas = metas_to_upsert[i : i + BATCH_SIZE]
                batch_docs = docs_to_upsert[i : i + BATCH_SIZE]
                
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeds,
                    metadatas=batch_metas,
                    documents=batch_docs
                )
            logger.info(f"Upserted {len(ids_to_upsert)} records. Deduplication active.")

if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    pipeline.run(r"docs\raw")