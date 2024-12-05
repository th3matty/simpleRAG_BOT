import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import chromadb
from app.services.embeddings import EmbeddingService
from app.config import settings

def load_test_documents():
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    
    # Create or get collection
    try:
        collection = client.create_collection("documents")
        print("collection", collection)
    except ValueError:
        collection = client.get_collection("documents")
    
    # Test documents
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "FastAPI is a modern web framework for building APIs with Python based on standard Python type hints.",
        "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation for more accurate responses.",
        "Geschäftsführer der Firma Malek ist Herr Rudolph Malek."
    ]
    
    # Generate embeddings
    embedding_service = EmbeddingService()
    embeddings = embedding_service.get_embeddings(documents)
    
    # Add documents to ChromaDB
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=[{"source": f"test_doc_{i}"} for i in range(len(documents))],
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    
    print("Test documents loaded successfully!")

if __name__ == "__main__":
    load_test_documents()