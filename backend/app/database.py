import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any
from .config import settings, logger


class ChromaDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False),
        )
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection("documents")
        return self._collection

    def query_documents(
        self, query_embedding: List[float], n_results: int = None
    ) -> Dict[str, Any]:
        """
        Query documents using embedding vector with similarity threshold filtering.

        Args:
            query_embedding: The query embedding vector
            n_results: Number of results to return (defaults to settings.top_k_results)

        Returns:
            Dictionary containing documents, metadata, and distances

        Raises:
            Exception: If query fails
        """
        try:
            n_results = n_results or settings.top_k_results
            logger.debug(
                f"Querying documents with threshold {settings.similarity_threshold}"
            )

            # Query with similarity threshold
            # ChromaDB uses distance (0 is most similar), so convert similarity to distance
            distance_threshold = 1 - settings.similarity_threshold

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"distance": {"$lte": distance_threshold}},
            )

            # If no results meet the threshold, fall back to top results without threshold
            if not results["documents"] or not results["documents"][0]:
                logger.debug(
                    "No results met similarity threshold, falling back to top results"
                )
                results = self.collection.query(
                    query_embeddings=[query_embedding], n_results=n_results
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

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: List[str],
    ):
        """Add documents to the collection."""
        try:
            self.collection.add(
                documents=documents, embeddings=embeddings, metadatas=metadata, ids=ids
            )
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}", exc_info=True)
            raise


# Global database instance
db = ChromaDB()
