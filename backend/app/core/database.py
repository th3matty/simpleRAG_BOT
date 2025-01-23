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
        self,
        query_embedding: List[float],
        n_results: int = None,
        similarity_threshold: float = None,
    ) -> Dict[str, Any]:
        """
        Query documents using embedding vector with similarity threshold filtering.

        Args:
            query_embedding: The query embedding vector
            n_results: Number of results to return (defaults to settings.top_k_results)
            similarity_threshold: Minimum similarity score for results (defaults to settings.similarity_threshold)

        Returns:
            Dictionary containing documents, metadata, and distances

        Raises:
            Exception: If query fails
        """
        try:
            n_results = n_results or settings.top_k_results
            threshold = similarity_threshold or settings.similarity_threshold
            logger.debug(f"Querying documents with threshold {threshold}")

            # Get more initial results to ensure we don't miss relevant matches
            expanded_n = min(n_results * 5, 30)  # Get up to 5x results, max 30
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=expanded_n,
            )

            # Filter results by similarity threshold and limit to n_results
            if results["documents"] and results["documents"][0]:
                filtered_docs = []
                filtered_meta = []
                filtered_dist = []
                filtered_ids = []

                for i, distance in enumerate(results["distances"][0]):
                    # Convert distance to similarity score (0-1 range)
                    similarity = (2 - distance) / 2

                    # For factual queries, also consider relative threshold
                    # Keep results that are within 20% of the best match
                    best_similarity = (2 - results["distances"][0][0]) / 2
                    relative_threshold = max(threshold, best_similarity * 0.8)

                    if similarity >= relative_threshold:
                        filtered_docs.append(results["documents"][0][i])
                        filtered_meta.append(results["metadatas"][0][i])
                        filtered_dist.append(distance)
                        filtered_ids.append(results["ids"][0][i])

                        # Stop if we have enough results
                        if len(filtered_docs) >= n_results:
                            break

                # Update results with filtered data
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
