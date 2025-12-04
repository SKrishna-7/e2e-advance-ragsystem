from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
# Define the structure for the evaluation output

import os
base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# model = os.getenv("OLLAMA_MODEL", "gemma3:1b")

class EvalScore(BaseModel):
    faithfulness_score: float = Field(description="Score from 0.0 to 1.0. 1.0 means the answer is fully derived from the provided context.")
    relevance_score: float = Field(description="Score from 0.0 to 1.0. 1.0 means the answer directly addresses the user's question.")
    reasoning: str = Field(description="Brief explanation of why these scores were given.")

class RAGEvaluator:
    def __init__(self, model_name):
        # We use temperature=0 for the judge to ensure consistent, logical grading
        self.llm = ChatGroq(model=model_name, temperature=0)
        # self.llm = ChatOllama(model=model_name, temperature=0,base_url=base)
        
        # System prompt for the "Judge"
        eval_system_prompt = (
            "You are an impartial judge evaluating a RAG system. "
            "You will be given a QUESTION, a set of RETRIEVED CONTEXTS, and the GENERATED ANSWER.\n\n"
            "Evaluate on two criteria:\n"
            "1. Faithfulness: Is the answer derived *solely* from the provided context? (0.0 = Hallucination, 1.0 = Faithful)\n"
            "2. Relevance: Does the answer directly address the user's question? (0.0 = Irrelevant, 1.0 = Perfect)\n\n"
            "Return your evaluation in JSON format with fields: 'faithfulness_score', 'relevance_score', and 'reasoning'."
        )
        
        self.eval_chain = (
            ChatPromptTemplate.from_messages([
                ("system", eval_system_prompt),
                ("human", "QUESTION: {question}\n\nCONTEXT: {context}\n\nANSWER: {answer}")
            ])
            | self.llm.with_structured_output(EvalScore)
        )

    def evaluate(self, question: str, answer: str, documents: List[Document]) -> dict:
        """Runs the evaluation chain."""
        if not documents:
            return {"error": "No documents retrieved to evaluate against."}

        # Format context for the judge
        context_text = "\n\n".join([doc.page_content for doc in documents])
        
        try:
            result = self.eval_chain.invoke({
                "question": question,
                "context": context_text,
                "answer": answer
            })
            return {
                "faithfulness": result.faithfulness_score,
                "relevance": result.relevance_score,
                "reasoning": result.reasoning
            }
        except Exception as e:
            return {"error": str(e)}