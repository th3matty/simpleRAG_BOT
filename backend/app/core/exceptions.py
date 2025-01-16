"""
Custom exceptions for the application.
"""


class RAGException(Exception):
    """Base exception for RAG-related errors."""

    pass


class ConfigurationError(RAGException):
    """Raised when there's an error in the configuration."""

    pass


class DatabaseError(RAGException):
    """Raised when there's an error with the database operations."""

    pass


class EmbeddingError(RAGException):
    """Raised when there's an error generating embeddings."""

    pass


class LLMError(RAGException):
    """Raised when there's an error with the LLM service."""

    pass


class CalculatorError(RAGException):
    """Raised when there's an error evaluating a mathematical expression."""

    pass
