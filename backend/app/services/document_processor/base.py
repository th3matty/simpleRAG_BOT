from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDocumentProcessor(ABC):
    """
    Abstract base class for document processors.
    All document type processors (PDF, DOCX, CSV) must implement these methods.
    """

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from the document.

        Args:
            file_path: Path to the document file

        Returns:
            Extracted text content
        """
        pass

    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from the document.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary containing document metadata
        """
        pass
