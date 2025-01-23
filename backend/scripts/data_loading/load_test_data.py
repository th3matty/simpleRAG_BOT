import sys
from pathlib import Path
import logging
import os

# Add both the backend directory and its parent to sys.path
script_dir = Path(__file__).parent
backend_dir = script_dir.parent.parent
project_root = backend_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_dir))


os.environ["ENV_FILE"] = str(backend_dir / ".env")

# Configure logging to show all details
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # Print to console
)

import chromadb
from app.services.embeddings import EmbeddingService
from app.core.config import settings
from app.services.chunker import DocumentProcessor

logger = logging.getLogger(__name__)


def load_test_documents():
    client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    embedding_service = EmbeddingService(model_name=settings.embedding_model)
    document_processor = DocumentProcessor(
        embedding_service=embedding_service, max_chunk_size=800, chunk_overlap=200
    )

    # Get or create collection
    collection_name = "documents"

    # Delete and recreate collection to clear all documents
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except:
        logger.info(f"No existing collection to delete")

    collection = client.create_collection(collection_name)
    logger.info(f"Created new collection: {collection_name}")

    # Process documents
    documents_dir = Path(backend_dir) / "data" / "test_documents"
    all_chunks = []

    # Read and process each markdown file
    for file_path in sorted(documents_dir.glob("*.md")):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

                # Process document with metadata
                metadata = {
                    "source": file_path.name,
                    "file_type": "markdown",
                    "original_filename": file_path.name,
                }

                processed_chunks = document_processor.process_document(
                    content=content, metadata=metadata
                )

                all_chunks.extend(processed_chunks)
                logger.info(
                    f"Processed {file_path.name} into {len(processed_chunks)} chunks"
                )

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            continue

    # Add all chunks to ChromaDB
    if all_chunks:
        try:
            collection.add(
                embeddings=[
                    chunk.embedding
                    for chunk in all_chunks
                    if chunk.embedding is not None
                ],
                documents=[chunk.content for chunk in all_chunks],
                metadatas=[chunk.metadata for chunk in all_chunks],
                ids=[
                    f"{chunk.metadata['parent_id']}_{chunk.metadata['chunk_index']}"
                    for chunk in all_chunks
                ],
            )
            logger.info(f"Successfully loaded {len(all_chunks)} chunks into ChromaDB")
        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {str(e)}")
            raise
    else:
        logger.warning("No chunks were processed successfully")


if __name__ == "__main__":
    load_test_documents()
