from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver 

from src.retrieval.retrieval import RetrievalEngine
from src.generation.generation_engine import GenerationEngine
from src.utils.config import RetrievalConfig , ModelConfig
from langchain_core.documents import Document


# 1. Define the Graph State
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: The user's original or rewritten question.
        intent: 'CHAT' or 'SEARCH'.
        documents: List of retrieved documents.
        answer: The final generated response.
        loop_count: Safety counter to prevent infinite rewriting loops.
    """
    question: str
    intent: str
    documents: List[Document]
    answer: str
    loop_count: int
    chat_history: List[str]

class RAGGraph:
    def __init__(self):
        # Initialize Engines
        config = RetrievalConfig() 
        self.retriever = RetrievalEngine(config)
        self.generator = GenerationEngine()
        
        # Build the Workflow
        self.app = self._build_workflow()

    # --- NODES ---

    def decide_intent(self, state: GraphState):
        """Node 1: Analyze user intent (Chat vs Search)."""
        print("   ↳ 🧠 Checking Intent...", end="\r")
        intent = self.generator.check_intent(state["question"])
        return {"intent": intent, "loop_count": 0} # Initialize loop count

    def run_chat(self, state: GraphState):
        """Node 2A: Handle casual conversation."""
        print("   ↳ 💬 Chatting...", end="\r")
        response = self.generator.chat_casually(state["question"])
        return {"answer": response}

    def retrieve_documents(self, state: GraphState):
        """Node 2B: Retrieve documents based on the question."""
        print("   ↳ 🔍 Retrieving...", end="\r")
        docs = self.retriever.search(state["question"])
        return {"documents": docs}

    def grade_documents(self, state: GraphState):
        """
        Node 3: Filter retrieved documents.
        Checks if the retrieved documents are actually relevant to the question.
        """
        print("   ↳ ⚖️  Grading Docs...", end="\r")
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs = []
        for d in documents:
            score = self.generator.grade_document(question, d.page_content)
            if score == "yes":
                filtered_docs.append(d)
        
        return {"documents": filtered_docs}

    def transform_query(self, state: GraphState):
        """
        Node 4: Rewrite the query (Self-Correction).
        If documents were irrelevant, we rewrite the question to try again.
        """
        question = state["question"]
        loop_count = state.get("loop_count", 0)
        
        print(f"   ↳ 🔄 Rewriting query (Attempt {loop_count + 1})...", end="\r")
        better_question = self.generator.rewrite_query(question)
        
        return {"question": better_question, "loop_count": loop_count + 1}

    def generate_rag_answer(self, state: GraphState):
        """Node 5: Generate the final answer using valid documents."""
        print("   ↳ 📝 Generating...", end="\r")
        response = self.generator.generate_answer(
            state["question"], 
            state["documents"],
            state.get("chat_history", [])
        )

        new_history = state.get("chat_history", []) + [
            f"User: {state['question']}",
            f"Assistant: {response}"
        ]
        
        if len(new_history) > 10: 
            new_history = new_history[-10:]

        return {"answer": response, "chat_history": new_history}
    # --- EDGES ---

    def route_query(self, state: GraphState):
        """Conditional Edge: Route based on intent."""
        if state["intent"] == "CHAT":
            return "chat_node"
        return "retrieve_node"

    def decide_to_generate(self, state: GraphState):
        """
        Conditional Edge: Decide whether to generate or retry.
        """
        filtered_docs = state["documents"]
        loop_count = state.get("loop_count", 0)

        # Logic:
        # 1. If we have relevant docs -> Generate
        # 2. If no relevant docs AND we haven't looped too many times -> Rewrite Query
        # 3. If no relevant docs AND max loops reached -> Generate (will likely say "I don't know")
        
        if not filtered_docs:
            if loop_count < 3: # Max 3 retries
                return "transform_query"
            else:
                return "generate_node"
        
        return "generate_node"

    # --- WORKFLOW ---

    def _build_workflow(self):
        workflow = StateGraph(GraphState)

        # 1. Add Nodes
        workflow.add_node("intent_analyzer", self.decide_intent)
        workflow.add_node("chat_node", self.run_chat)
        workflow.add_node("retrieve_node", self.retrieve_documents)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("generate_node", self.generate_rag_answer)

        # 2. Add Edges
        workflow.set_entry_point("intent_analyzer")

        # Routing Intent
        workflow.add_conditional_edges(
            "intent_analyzer",
            self.route_query,
            {
                "chat_node": "chat_node",
                "retrieve_node": "retrieve_node"
            }
        )

        # Retrieval Flow
        workflow.add_edge("retrieve_node", "grade_documents")

        # Self-Correction Loop
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query", # Loop back
                "generate_node": "generate_node"      # Proceed
            }
        )

        # Connect Loop back to Retriever
        workflow.add_edge("transform_query", "retrieve_node")

        # End points
        workflow.add_edge("generate_node", END)
        workflow.add_edge("chat_node", END)

        # 3. Persistence
        
        checkpointer = MemorySaver()
        
        return workflow.compile(checkpointer=checkpointer)