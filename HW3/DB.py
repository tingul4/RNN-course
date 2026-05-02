#%%
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from Chunking import docs_a, docs_b

import os

# --- TA Note: Use a high-quality model compatible with RTX 4090 ---
EMBED_MODEL_NAME = "BAAI/bge-m3" # Or "sentence-transformers/all-MiniLM-L6-v2" for speed
print(f"Loading Embedding Model: {EMBED_MODEL_NAME} on CUDA...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={'device': 'cuda'}, # Utilizing RTX 4090
    encode_kwargs={'normalize_embeddings': True}
)


print("Handling ChromaDB Index... (fixed Chunking)")
if os.path.exists("./chroma_db_fixed") and os.listdir("./chroma_db_fixed"):
    print("Database already exists, loading from disk...") # Avoid generating repeated embeddings if already done
    vector_db = Chroma(
        collection_name="science_knowledge_base_A",
        embedding_function=embeddings,
        persist_directory="./chroma_db_fixed"
    )
else:
    print("Building fresh ChromaDB Index...")
    vector_db = Chroma.from_documents(
        documents=docs_a,
        embedding=embeddings,
        collection_name="science_knowledge_base_A",
        persist_directory="./chroma_db_fixed"
    )

print("Handling ChromaDB Index... (Semantic Chunking)")
if os.path.exists("./chroma_db_semantic") and os.listdir("./chroma_db_semantic"):
    print("Database already exists, loading from disk...") # Avoid generating repeated embeddings if already done
    vector_db = Chroma(
        collection_name="science_knowledge_base_B",
        embedding_function=embeddings,
        persist_directory="./chroma_db_semantic"
    )
else:
    print("Building fresh ChromaDB Index...")
    vector_db = Chroma.from_documents(
        documents=docs_b,
        embedding=embeddings,
        collection_name="science_knowledge_base_B",
        persist_directory="./chroma_db_semantic"
    )

print("Vector DB ready.")