import sys
from pathlib import Path

# Add the parent directory to sys.path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

# Set up environment file path
import os

os.environ["ENV_FILE"] = str(backend_dir / ".env")

import chromadb
from app.services.embeddings import EmbeddingService
from app.config import settings


def load_test_documents():
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=settings.chroma_persist_directory)

    # Create or recreate collection
    collection_name = "documents"
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass

    collection = client.create_collection(collection_name)
    print(f"Created new collection: {collection_name}")

    # Read documents from markdown files
    documents_dir = Path(__file__).parent / "test_documents"
    documents = []
    filenames = []

    # Read all markdown files
    for file_path in sorted(documents_dir.glob("*.md")):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            documents.append(content)
            filenames.append(file_path.name)

    if not documents:
        print("No markdown files found in test_documents directory!")
        return

    # Generate embeddings
    embedding_service = EmbeddingService(model_name=settings.embedding_model)
    embeddings = embedding_service.get_embeddings(documents)

    # Add documents to ChromaDB
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=[{"source": filename} for filename in filenames],
        ids=[f"doc_{i}" for i in range(len(documents))],
    )

    print(f"Loaded {len(documents)} documents successfully!")
    for filename in filenames:
        print(f"- {filename}")


if __name__ == "__main__":
    load_test_documents()
