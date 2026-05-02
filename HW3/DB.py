#%%
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from Chunking import docs_a, docs_b, docs_a_1000

# --- TA Note: Use a high-quality model compatible with RTX 4090 ---
EMBED_MODEL_NAME = "BAAI/bge-m3" # Or "sentence-transformers/all-MiniLM-L6-v2" for speed
print(f"Loading Embedding Model: {EMBED_MODEL_NAME} on CUDA...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={'device': 'cuda'}, # Utilizing RTX 4090
    encode_kwargs={'normalize_embeddings': True}
)

print("Handling ChromaDB Index... (fixed Chunking | chunk_size=500)")
vector_db_fixed_500 = Chroma(
    collection_name="fixed_chunk_size_500",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
if vector_db_fixed_500._collection.count() == 0:
    print("Building fresh ChromaDB Index...")
    vector_db_fixed_500.add_documents(documents=docs_a)
else:
    print("Database collection already exists, loading from disk...")

print("Handling ChromaDB Index... (fixed Chunking | chunk_size=1000)")
vector_db_fixed_1000 = Chroma(
    collection_name="fixed_chunk_size_1000",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
if vector_db_fixed_1000._collection.count() == 0:
    print("Building fresh ChromaDB Index...")
    vector_db_fixed_1000.add_documents(documents=docs_a_1000)
else:
    print("Database collection already exists, loading from disk...")

print("Handling ChromaDB Index... (Semantic Chunking)")
vector_db_semantic = Chroma(
    collection_name="semantic_chunk_size_1000",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
if vector_db_semantic._collection.count() == 0:
    print("Building fresh ChromaDB Index...")
    vector_db_semantic.add_documents(documents=docs_b)
else:
    print("Database collection already exists, loading from disk...")

print("Vector DB ready.")