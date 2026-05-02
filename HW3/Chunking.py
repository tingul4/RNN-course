# Terminal commands
# pip install langchain langchain-community langchain-huggingface chromadb sentence-transformers torch
# ollama pull llama3
#%%
import pandas as pd
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/train.csv")

def load_corpus(path=_DATA_PATH):
    """Load train.csv; each (question + one answer option) becomes one Document."""
    df = pd.read_csv(path)
    corpus = []
    for _, row in df.iterrows():
        for opt in ["A", "B", "C", "D", "E"]:
            text = f"{row['prompt'].strip()}\n\nOption {opt}: {str(row[opt]).strip()}"
            corpus.append(Document(
                page_content=text,
                metadata={"question_id": int(row["id"]), "option": opt,
                          "is_correct": str(row["answer"]).strip() == opt}
            ))
    return corpus

# --- TA Note: Use a small synthetic corpus for testing logic ---
# In the real assignment, students will load the Wikipedia parquet file.
raw_text_corpus = [
    "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic organisms. It is often called the powerhouse of the cell.",
    "Mitochondria generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy.",
    "Newton's laws of motion are three physical laws that, together, laid the foundation for classical mechanics.",
    "The first law states that an object remains at rest or in uniform motion unless acted upon by a force.",
    "The second law states that the rate of change of momentum of an object is directly proportional to the force applied, or F=ma.",
    "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.",
    "Large Language Models (LLMs) are AI systems capable of understanding and generating human language."
]

# Convert to LangChain Documents
# documents = [Document(page_content=text, metadata={"source": f"doc_{i}"}) for i, text in enumerate(raw_text_corpus)] # For test data
documents = load_corpus() # For real data

# --- Strategy A: Fixed-Size Chunking (Naive) ---
# Small chunks, strict cut-off
splitter_a = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=10
)
docs_a = splitter_a.split_documents(documents)
print(f"Strategy A (Fixed): Created {len(docs_a)} chunks.")

# --- Strategy B: Semantic/Larger Chunking ---
# Larger chunks to preserve context (Paragraph level)
splitter_b = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs_b = splitter_b.split_documents(documents)
print(f"Strategy B (Semantic): Created {len(docs_b)} chunks.")