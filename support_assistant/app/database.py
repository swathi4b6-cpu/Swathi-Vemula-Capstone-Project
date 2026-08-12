import os
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize lightweight local embeddings engine
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create isolated ephemeral/persistent vector container directory
CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Retrieve or rebuild a distinct vector storage partition
collection = chroma_client.get_or_create_collection(
    name="zepto_policy_corpus",
    metadata={"hnsw:space": "cosine"}
)

def populate_database_if_empty():
    """Reads raw local policy text files, generates features and indexes into ChromaDB."""
    if collection.count() > 0:
        return
        
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
    if not os.path.exists(docs_dir):
        print(f"[!] Warning: Directory '{docs_dir}' not detected. Skipping automatic initialization.")
        return

    print("[*] Empty collection detected. Running ingestion pipeline loop...")
    for file_name in sorted(os.listdir(docs_dir)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(docs_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            # Use strict structural reference IDs corresponding to filenames
            doc_id = os.path.splitext(file_name)[0]
            
            # Generate the embedding representation vector
            vector = embedding_model.encode(content).tolist()
            
            # Store atomic components directly inside storage index
            collection.add(
                documents=[content],
                embeddings=[vector],
                ids=[doc_id],
                metadatas=[{"source": file_name}]
            )
    print(f"[+] Ingestion complete. Indexed {collection.count()} reference policy fields.")

def query_vector_store(query_text: str, top_k: int = 3):
    """Encodes input runtime query strings and pulls nearest neighbor documents."""
    query_vector = embedding_model.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    
    # Restructure matrix entries into standardized functional formats
    output = []
    if results and results['documents'] and len(results['documents'][0]) > 0:
        for i in range(len(results['documents'][0])):
            output.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "score": float(results['distances'][0][i])
            })
    return output
