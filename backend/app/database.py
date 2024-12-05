import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any
from .config import settings, logger

class ChromaDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection("documents")
        return self._collection

    def query_documents(self, query_embedding: List[float], n_results: int = 3) -> Dict[str, Any]:
        """Query documents using embedding vector."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}", exc_info=True)
            raise

    def get_all_documents(self) -> Dict[str, Any]:
        """Retrieve all documents from the collection."""
        try:
            return self.collection.get()
        except Exception as e:
            logger.error(f"Error retrieving all documents: {str(e)}", exc_info=True)
            raise

    def add_documents(self, documents: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """Add documents to the collection."""
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadata
            )
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}", exc_info=True)
            raise

# Global database instance
db = ChromaDB()
