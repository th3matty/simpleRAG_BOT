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

            # Query for top results and filter by similarity threshold later if needed
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )

            # Filter results by similarity threshold if we have any results
            if results["documents"] and results["documents"][0]:
                # Initialize lists for filtered results
                filtered_docs = []
                filtered_meta = []
                filtered_dist = []
                filtered_ids = []

                for i, distance in enumerate(results["distances"][0]):
                    # Convert distance to similarity (ChromaDB uses cosine distance)
                    similarity = (2 - distance) / 2
                    if similarity >= settings.similarity_threshold:
                        filtered_docs.append(results["documents"][0][i])
                        filtered_meta.append(results["metadatas"][0][i])
                        filtered_dist.append(distance)
                        filtered_ids.append(results["ids"][0][i])

                # If we have filtered results, update results with filtered data
                if filtered_docs:
                    results = {
                        "documents": [filtered_docs],
                        "metadatas": [filtered_meta],
                        "distances": [filtered_dist],
                        "ids": [filtered_ids],
                    }

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
