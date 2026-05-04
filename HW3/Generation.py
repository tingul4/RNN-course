# Terminal commands
# run `ollama serve` in terminal before executing this script
# %%
import re
import requests
import time
from Retrieval import advanced_rag_retrieve
from DB import vector_db_semantic
import pandas as pd


def query_ollama(prompt, model="llama3:8b"):
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=data)
        return response.json()["response"]
    except Exception as e:
        return f"Error calling Ollama: {e}"


def parse_answer_letter(text):
    """Extract the first A-E letter from the LLM response."""
    match = re.search(r"\b([A-E])\b", text.strip())
    return match.group(1) if match else None


def run_rag_pipeline(query, options_text):
    # 1. Retrieve & Rerank
    retrieved_docs = advanced_rag_retrieve(query, vector_db_semantic)

    # 2. Construct Prompt
    context_text = "\n\n".join([d.page_content for d in retrieved_docs])

    prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful science assistant. Use the context below to answer the multiple-choice question.
If the correct answer cannot be deduced from the context, output 'F' meaning 'I do not know'.
Reply with ONLY the letter of the correct answer (A, B, C, D, or E). No explanation.

Context:
{context_text}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {query}

{options_text}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    # 3. Generate
    print("\nGenerating Answer...")
    start_gen = time.time()
    answer = query_ollama(prompt)
    llm_latency = time.time() - start_gen
    print(f"[Latency] LLM Generation: {llm_latency:.4f} seconds")
    return answer


def evaluate_accuracy_with_RAG():
    print("\n🔍 Accuracy Evaluation (with RAG)...")
    df = pd.read_csv("dataset/train.csv").head(50)

    correct = 0
    total = len(df)

    for _, row in df.iterrows():
        query = row["prompt"]
        q_id = int(row["id"])
        true_label = row["answer"]

        options_text = (
            f"A: {row['A']}\nB: {row['B']}\nC: {row['C']}\nD: {row['D']}\nE: {row['E']}"
        )

        # Generate answer with RAG context + options
        generated_answer = run_rag_pipeline(query, options_text)

        # Parse the predicted letter directly
        pred = parse_answer_letter(generated_answer)

        is_correct = pred == true_label
        if is_correct:
            correct += 1

        print(
            f"Q ID: {q_id} | True: {true_label} | Pred: {pred} | Raw: {generated_answer[:50]!r} | Correct: {is_correct}"
        )

    accuracy = correct / total
    print(f"[with RAG] Accuracy Results on 50 samples: {accuracy:.2f}")
    return accuracy


def evaluate_accuracy_without_RAG():
    print("\n🔍 Accuracy Evaluation (without RAG)...")
    df = pd.read_csv("dataset/train.csv").head(50)

    correct = 0
    total = len(df)

    for _, row in df.iterrows():
        query = row["prompt"]
        q_id = int(row["id"])
        true_label = row["answer"]

        options_text = (
            f"A: {row['A']}\nB: {row['B']}\nC: {row['C']}\nD: {row['D']}\nE: {row['E']}"
        )

        prompt = f"""Answer the following multiple-choice science question.
Reply with ONLY the letter of the correct answer (A, B, C, D, or E). No explanation.

Question: {query}

{options_text}
"""
        generated_answer = query_ollama(prompt)

        pred = parse_answer_letter(generated_answer)

        is_correct = pred == true_label
        if is_correct:
            correct += 1

        print(
            f"Q ID: {q_id} | True: {true_label} | Pred: {pred} | Raw: {generated_answer[:50]!r} | Correct: {is_correct}"
        )

    accuracy = correct / total
    print(f"[without RAG] Accuracy Results on 50 samples: {accuracy:.2f}")
    return accuracy


if __name__ == "__main__":
    # --- Final Execution ---
    # q1 = "What is the equation for Newton's second law?"
    # answer1 = run_rag_pipeline(q1)
    # print(f"\nFinal Answer:\n{answer1}")

    # print("-" * 50)

    # q2 = "How do plants convert light?"
    # answer2 = run_rag_pipeline(q2)
    # print(f"\nFinal Answer:\n{answer2}")

    # print("-" * 50)

    evaluate_accuracy_with_RAG()
    evaluate_accuracy_without_RAG()
