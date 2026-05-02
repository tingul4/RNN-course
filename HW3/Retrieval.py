#%%
import pandas as pd
import time
from sentence_transformers import CrossEncoder
from DB import vector_db_fixed_500, vector_db_fixed_1000, vector_db_semantic

# --- Load Cross-Encoder (The Re-ranker) ---
# This model takes (Query, Document) pairs and outputs a similarity score.
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"Loading Cross-Encoder: {RERANK_MODEL_NAME} on CUDA...")

reranker = CrossEncoder(RERANK_MODEL_NAME, device='cuda')

def advanced_rag_retrieve(query, db, top_k_retrieval=20, top_k_rerank=3):
    """
    Stage 1: Vector Search (Fast, High Recall)
    Stage 2: Cross-Encoder Re-ranking (Slow, High Precision)
    """
    
    # --- Stage 1: Dense Retrieval ---
    print(f"\nQuery: {query}")
    start_vs = time.time()
    results = db.similarity_search_with_score(query, k=top_k_retrieval)
    initial_docs = [doc for doc, score in results]
    vs_latency = time.time() - start_vs
    print(f"[Latency] Vector Search: {vs_latency:.4f} seconds")
    print(f"Stage 1: Retrieved {len(initial_docs)} candidates via Vector Search.")
    print("Stage 1: Top Results:")
    for i, (doc, score) in enumerate(results[:top_k_rerank]):
        print(f"  [{i+1}] Score: {score:.4f}\n  Text: {doc.page_content}...\n  ------------------------")

    # --- Stage 2: Re-ranking ---
    # Prepare pairs for the Cross-Encoder: [[Query, Doc1], [Query, Doc2]...]
    doc_texts = [d.page_content for d in initial_docs]
    pairs = [[query, doc_text] for doc_text in doc_texts]
    
    # Predict scores (higher is better)
    
    start_rr = time.time()
    scores = reranker.predict(pairs)
    rr_latency = time.time() - start_rr
    print(f"[Latency] Re-ranking (Cross-Encoder): {rr_latency:.4f} seconds")
    
    # Combine docs with scores
    scored_docs = list(zip(initial_docs, scores))
    
    # Sort by score descending
    scored_docs_sorted = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    # Select Top-N
    final_docs = [doc for doc, score in scored_docs_sorted[:top_k_rerank]]
    
    # Debug Print: Show re-ranking effect
    print("Stage 2: Top Re-ranked Results:")
    for i, (doc, score) in enumerate(scored_docs_sorted[:top_k_rerank]):
        print(f"  [{i+1}] Score: {score:.4f}\n  Text: {doc.page_content}...\n  ------------------------")
        
    return final_docs

def run_experiment_1(query, top_k_retrieval=20, top_k_rerank=3):
    print("\n🔍 Running Experiment 1: Chunk Size Analysis...")
    print("\n--- Using fixed Chunking strategy with chunk_size=500 ---")
    advanced_rag_retrieve(query, vector_db_fixed_500)
    print("\n--- Using fixed Chunking strategy with chunk_size=1000 ---")
    advanced_rag_retrieve(query, vector_db_fixed_1000)
    
def run_experiment_2(db, _DATA_PATH="dataset/train.csv", top_k_retrieval=20, top_k_rerank=3):
    df = pd.read_csv(_DATA_PATH)
    found_examples = 0

    print("\n🔍 Running Experiment 2: Re-ranking Impact...")
    
    for _, row in df.iterrows():
        query = row['prompt']
        q_id = int(row['id'])

        # --- Stage 1: Dense Retrieval ---
        print(f"\nQuery: {query}")
        
        initial_docs = db.similarity_search(query, k=top_k_retrieval)
        
        print(f"Response (Stage 1 | top-1): {initial_docs[0].page_content[:200]}")
        
        stage1_is_correct = initial_docs[0].metadata.get('is_correct') and initial_docs[0].metadata.get('question_id') == q_id if initial_docs else False

        # --- Stage 2: Re-ranking ---
        # Prepare pairs for the Cross-Encoder: [[Query, Doc1], [Query, Doc2]...]
        doc_texts = [d.page_content for d in initial_docs]
        pairs = [[query, doc_text] for doc_text in doc_texts]
        
        # Predict scores (higher is better)
        scores = reranker.predict(pairs)
        
        # Combine docs with scores
        scored_docs = list(zip(initial_docs, scores))
        
        # Sort by score descending
        scored_docs_sorted = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        # Select Top-N
        final_docs = [doc for doc, score in scored_docs_sorted[:top_k_rerank]]
        
        stage2_is_correct = final_docs[0].metadata.get('is_correct') and final_docs[0].metadata.get('question_id') == q_id if final_docs else False

        print(f"Response (Stage 2 | top-1): {final_docs[0].page_content[:200]}")

        found = not stage1_is_correct and stage2_is_correct

        if (found):
            found_examples += 1
            print(f"\n✅ Q_ID {q_id}: {query} | Found after re-ranking!")
        if (found_examples == 2):
            print(f"\n--- Completed experiment with {found_examples} examples found after re-ranking. ---")
            break
        print("\n" + "-" * 50)


if __name__ == "__main__":
    # run experiment 1
    test_query = "What generates energy in the cell?"
    run_experiment_1(test_query, vector_db_semantic)
    # run experiment 2
    run_experiment_2(vector_db_semantic)