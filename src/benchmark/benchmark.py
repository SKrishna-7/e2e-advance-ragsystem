import time
import pandas as pd
import torch
import gc
import os
from tqdm import tqdm
from src.graph.rag_graph import RAGGraph
from src.evaluator.rag_evaluator import RAGEvaluator
from src.utils.config import ModelConfig
from langchain_core.tracers.context import tracing_v2_enabled

from dotenv import load_dotenv
load_dotenv()

# --- OPTIMIZATION FOR LOW VRAM ---
# Helps avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- CONFIGURATION ---
# Models to Benchmark
MODELS_TO_TEST = [
    # {
    #     "name": "Ollama (Gemma)", 
    #     "provider": "ollama", 
    #     "model_id": "gemma3:1b" # Adjust if you specifically have 'gemma3' installed
    # },
    # {
    #     "name": "Groq (openai/gpt-oss-20b)", 
    #     "provider": "groq", 
    #     "model_id": "openai/gpt-oss-20b"
    # },
    {
        "name": "Groq (llama-3.1-8b-instant)", 
        "provider": "groq", 
        "model_id": "llama-3.1-8b-instant"
    }
]

# Test Questions

# Test Questions
TEST_DATASET = [
    "Who translated this specific edition of 'Beyond Good and Evil'?",
    "Who wrote the Introduction to this edition?",
    "According to the Publishers' Note, what other book did Willard Huntington Wright write about Nietzsche?",
    "How does the computational complexity of a Self-Attention layer compare to a Recurrent layer?",
    "What is the title of the 'Aftersong' or poem found at the end of the book?",
    "Nietzshe's opinion on Epicurus ?",
    "Describe the architecture of the Encoder stack in the Transformer.",

    "How does Nietzsche describe the 'Will to Truth' in the first chapter?",
    "What optimizer was used for training the Transformer, and what were its parameters?",

    "What does Nietzsche say is the fundamental drive of all philosophy?",
    "How does the author characterize the 'Free Spirit'?",
    "What is Nietzsche's criticism of the 'English Psychologists'?",

    "Complete the quote: 'He who fights with monsters should look to it that he himself does not become a...'",
    "What does Nietzsche say happens when you gaze long into an abyss?",


    "What is the primary advantage of the Transformer over RNNs regarding sequential computation?",
    "What is the formula for the attention mechanism used in the paper?",
    "Why is the dot product scaled by 1/sqrt(dk) in the attention mechanism?",
    
    # Implementation Details
    "How does the model handle the lack of recurrence and convolution to perceive order?",
    "What label smoothing value was used during training?",
    
    # Results & Performance
    "What BLEU score did the Transformer achieve on the WMT 2014 English-to-German task?",
    "What hardware was used to train the models and how long did it take?"
]

def clear_gpu_memory():
    """Aggressively clears CUDA cache to prevent OOM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def run_comparative_benchmark():
    print("🚀 Starting Comparative Model Benchmark...\n")
    
    all_results = []
    
    # Initialize Judge ONCE globally? 
    # NO: The Judge (if local) takes VRAM. If using Groq, it's fine.
    # Assuming Judge uses Groq (Llama 70b), it uses API, not local VRAM.
    print("⚖️  Initializing Judge (Llama-3.3-70b via Groq)...")
    try:
        evaluator = RAGEvaluator(model_name="llama-3.1-8b-instant")
        # evaluator = RAGEvaluator(model_name="gemma3:1b")
    except Exception as e:
        print(f"⚠️ Failed to init judge: {e}")
        return

    for model_conf in MODELS_TO_TEST:
        display_name = model_conf["name"]
        provider = model_conf["provider"]
        model_id = model_conf["model_id"]
        
        print(f"\n--------------------------------------------------")
        print(f"🧪 Testing Model: {display_name}")
        print(f"   ID: {model_id} | Provider: {provider}")
        print(f"--------------------------------------------------")

        # 1. Clean up previous model artifacts from VRAM
        clear_gpu_memory()

        # 2. Hot-Swap Configuration
        ModelConfig.LLM_PROVIDER = provider
        if provider == "ollama":
            ModelConfig.OLLAMA_MODEL_NAME = model_id
        else:
            ModelConfig.MODEL_NAME = model_id
            
        # 3. Re-Initialize Graph
        try:
            rag_bot = RAGGraph()
        except Exception as e:
            print(f"❌ Failed to initialize {display_name}: {e}")
            continue

        model_metrics = {"latencies": [], "faithfulness": [], "relevance": []}

        # 4. Run Test Queries
        for query in tqdm(TEST_DATASET, desc=f"Querying {display_name}"):
            try:
                start_time = time.time()
                inputs = {"question": query}
                # Unique thread ID to ensure clean state
                config = {"configurable": {"thread_id": f"bench_{provider}_{model_id}"}}
                
                response = rag_bot.app.invoke(inputs, config=config)
                
                end_time = time.time()
                latency = end_time - start_time
                
                generated_answer = response.get("answer", "No answer")
                retrieved_docs = response.get("documents", [])

                # Evaluate
                if retrieved_docs:
                    # Judge is external (Groq), so it shouldn't impact VRAM significantly
                    score = evaluator.evaluate(
                        question=query, 
                        answer=generated_answer, 
                        documents=retrieved_docs
                    )
                else:
                    score = {"faithfulness": 0.0, "relevance": 0.0}

                result_entry = {
                    "Model": display_name,
                    "Provider": provider,
                    "Question": query,
                    "Latency (s)": round(latency, 2),
                    "Faithfulness": score.get("faithfulness", 0.0),
                    "Relevance": score.get("relevance", 0.0),
                    "Answer": generated_answer[:100] + "..."
                }
                all_results.append(result_entry)
                
                model_metrics["latencies"].append(latency)
                model_metrics["faithfulness"].append(score.get("faithfulness", 0.0))
                model_metrics["relevance"].append(score.get("relevance", 0.0))

            except Exception as e:
                print(f"  ❌ Error on query '{query}': {e}")
                # If OOM happens during a query, try to recover for next query
                clear_gpu_memory()

        # Summary for this model
        avg_lat = sum(model_metrics["latencies"]) / len(model_metrics["latencies"]) if model_metrics["latencies"] else 0
        avg_faith = sum(model_metrics["faithfulness"]) / len(model_metrics["faithfulness"]) if model_metrics["faithfulness"] else 0
        avg_rel = sum(model_metrics["relevance"]) / len(model_metrics["relevance"]) if model_metrics["relevance"] else 0
        
        print(f"   👉 Avg Latency: {avg_lat:.2f}s")
        print(f"   👉 Avg Faithfulness: {avg_faith:.2f}")
        print(f"   👉 Avg Relevance: {avg_rel:.2f}")
        
        # Aggressive cleanup after model is done
        del rag_bot
        clear_gpu_memory()

    # --- FINAL REPORT ---
    if not all_results:
        print("\n❌ No results generated.")
        return

    df = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print("🏆 FINAL BENCHMARK LEADERBOARD")
    print("="*60)
    
    leaderboard = df.groupby("Model")[["Latency (s)", "Faithfulness", "Relevance"]].mean().reset_index()
    print(leaderboard.to_string(index=False))
    
    csv_filename = "multi_model_benchmark_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n✅ Detailed results saved to '{csv_filename}'")

if __name__ == "__main__":
    run_comparative_benchmark()