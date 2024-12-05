class RAGException(Exception):
    """Base exception for RAG application."""
    pass

class DatabaseError(RAGException):
    """Raised when database operations fail."""
    pass

class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass

class LLMError(RAGException):
    """Raised when LLM operations fail."""
    pass

class ConfigurationError(RAGException):
    """Raised when configuration is invalid."""
    pass
