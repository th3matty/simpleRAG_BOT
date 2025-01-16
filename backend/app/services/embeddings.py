from sentence_transformers import SentenceTransformer
from typing import List
from ..core.exceptions import EmbeddingError
from ..core.config import settings
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    _instance = None
    _model_name = None

    def __new__(cls, model_name: str):
        if cls._instance is None or cls._model_name != model_name:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            try:
                logger.info(f"Initializing embedding model: {model_name}")
                cls._instance.model = SentenceTransformer(model_name)
                cls._model_name = model_name
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {str(e)}")
                raise EmbeddingError(f"Failed to initialize embedding model: {str(e)}")
        return cls._instance

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to generate embeddings for

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

    def get_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: String to generate embedding for

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            return self.get_embeddings([text])[0]
        except Exception as e:
            logger.error(f"Failed to generate single embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate single embedding: {str(e)}")
