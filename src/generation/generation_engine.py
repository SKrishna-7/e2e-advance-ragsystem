from typing import List, Dict, Literal, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from logger.logger_config import get_logger

from langchain_groq import ChatGroq

from src.utils.config import ModelConfig

import os
logger = get_logger("Generator_LOG")



# --- DATA MODELS ---

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# --- ENGINE CLASS ---
base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

class GenerationEngine:
    def __init__(self, model_name: Optional[str] = None):
        """
        Initializes the Generator with a local Ollama model.
        """
        logger.info(f"Initializing Generator with model: {model_name}")
        self.config = ModelConfig()
        # model_name = model_name or self.config.MODEL_NAME
        # Main LLM for text generation
        
        print("Model Config : ",self.config.MODEL_NAME)

        if not model_name:
            if self.config.LLM_PROVIDER == "ollama":
                model_name = self.config.OLLAMA_MODEL_NAME
            else:
                model_name = self.config.MODEL_NAME
                
        if self.config.LLM_PROVIDER == "ollama":
            self.llm = ChatOllama(
                model=model_name,
                temperature=self.config.TEMP,
                base_url=self.config.OLLAMA_BASE_URL
            )
        else:
            self.llm = ChatGroq(
                model=model_name,
                temperature=self.config.TEMP
            )
        # --- 1. ROUTER CHAIN ---
        # Decides if we should chat or search
        router_system_prompt = (
            "You are an intent classifier. Determine if the user's query is 'CHAT' "
            "(greetings, small talk, compliments) or 'SEARCH' (questions needing "
            "information, context, or facts).\n"
            "Return ONLY one word: 'CHAT' or 'SEARCH'."
        )
        self.router_chain = (
            ChatPromptTemplate.from_messages([
                ("system", router_system_prompt),
                ("human", "{question}")
            ]) 
            | self.llm 
            | StrOutputParser()
        )

        # --- 2. CASUAL CHAT CHAIN ---
        # Handles small talk
        chat_system_prompt = (
            "You are a helpful AI assistant. Respond politely to the user's greeting "
            "or conversational query. Do not hallucinate technical information."
        )
        self.chat_chain = (
            ChatPromptTemplate.from_messages([
                ("system", chat_system_prompt),
                ("human", "{question}")
            ]) 
            | self.llm 
            | StrOutputParser()
        )

        # --- 3. RAG ANSWER CHAIN ---
        # Generates the final answer from documents
        rag_system_prompt = (
            "You are a technical assistant. Answer the user's question using ONLY "
            "the provided context snippets.\n"
            "If the answer is not in the context, return the top ranked retrieved document'.\n\n"
            "CITATION FORMAT: [Source: filename (Page X)]"
        )
        self.rag_chain = (
            ChatPromptTemplate.from_messages([
                ("system", rag_system_prompt),
                # Insert History here
                ("system", "Conversation History:\n{chat_history}"), 
                ("human", "Context:\n{context}\n\nQuestion: {question}")
            ])
            | self.llm
            | StrOutputParser()
        )

        # --- 4. GRADER CHAIN ---
        # Checks if retrieved docs are actually relevant
        grader_system_prompt = (
            "You are a grader assessing relevance of a retrieved document to a user question. "
            "If the document contains keyword(s) or semantic meaning related to the question, "
            "grade it as relevant. Give a binary score 'yes' or 'no'."
        )

        
        self.grader_chain = (
            ChatPromptTemplate.from_messages([
                ("system", grader_system_prompt),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
            ])
            | self.llm.with_structured_output(GradeDocuments)
        )

        # --- 5. REWRITER CHAIN ---
        # Transforms bad queries into good ones
        rewriter_system_prompt = (
            "You are a question re-writer that converts an input question to a better version "
            "that is optimized for vectorstore retrieval. Look at the input and try to reason "
            "about the underlying semantic intent."
        )
        self.rewriter_chain = (
            ChatPromptTemplate.from_messages([
                ("system", rewriter_system_prompt),
                ("human", "Initial Question: {question}\nFormulate an improved question.")
            ])
            | self.llm
            | StrOutputParser()
        )

    # --- PUBLIC METHODS ---

    def check_intent(self, query: str) -> str:
        """Determines if query is CHAT or SEARCH."""
        try:
            intent = self.router_chain.invoke({"question": query}).strip().upper()
            if "SEARCH" in intent: return "SEARCH"
            return "CHAT"
        except Exception as e:
            logger.warning(f"Router failed: {e}")
            return "SEARCH" # Default safe option

    def chat_casually(self, query: str) -> str:
        """Handles casual conversation."""
        return self.chat_chain.invoke({"question": query})

    def grade_document(self, question: str, document: str) -> str:
        """Determines if a document is relevant (yes/no)."""
        try:
            score = self.grader_chain.invoke({
                "question": question, 
                "document": document
            })
            return score.binary_score
        except Exception as e:
            logger.warning(f"Grading failed: {e}")
            return "yes" # Default to keeping it if grading fails

    def rewrite_query(self, question: str) -> str:
        """Rewrites the question for better retrieval."""
        return self.rewriter_chain.invoke({"question": question})

    # def generate_answer(self, query: str, retrieved_docs: List[Document]) -> str:
    #     """Generates the final RAG response."""
    #     if not retrieved_docs:
    #         return "I could not find any relevant documents to answer your question."

    #     logger.info(f"Generating answer for query: '{query}' using {len(retrieved_docs)} docs")
        
    #     context_str = self._format_context(retrieved_docs)
        
    #     try:
    #         response = self.rag_chain.invoke({
    #             "context": context_str,
    #             "question": query
    #         })
    #         return response
    #     except Exception as e:
    #         logger.error(f"Generation failed: {e}")
    #         return "Sorry, I encountered an error while generating the response."

    def generate_answer(self, query: str, retrieved_docs: List[Document], chat_history: List[str] = []) -> str:
        if not retrieved_docs:
            return "I could not find any relevant documents to answer your question."

        context_str = self._format_context(retrieved_docs)
        history_str = "\n".join(chat_history) if chat_history else "No previous history."
        
        try:
            response = self.rag_chain.invoke({
                "context": context_str,
                "question": query,
                "chat_history": history_str  # Pass formatted history
            })
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Sorry, I encountered an error while generating the response."

    def stream_generate_answer(self, query: str, retrieved_docs: List[Document], chat_history: List[str] = []):
        """Streams the RAG response token by token."""
        if not retrieved_docs:
            yield "I could not find any relevant documents to answer your question."
            return

        context_str = self._format_context(retrieved_docs)
        history_str = "\n".join(chat_history) if chat_history else "No previous history."
        
        try:
            # Using .stream() instead of .invoke()
            for chunk in self.rag_chain.stream({
                "context": context_str,
                "question": query,
                "chat_history": history_str
            }):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield "Error generating response."

    def _format_context(self, docs: List[Document]) -> str:
        """Helper to format docs into a string with metadata."""
        formatted_text = ""
        for doc in docs:
            source = doc.metadata.get("filename", "Unknown File")
            page = doc.metadata.get("page", "?")
            
            formatted_text += f"--- Source: {source} (Page {page}) ---\n"
            formatted_text += f"{doc.page_content}\n\n"
            
        return formatted_text
    
        