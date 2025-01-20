# app/services/document_processor/factory.py

from typing import Type
import os
from .base import BaseDocumentProcessor
import logging

logger = logging.getLogger(__name__)


class DocumentProcessorFactory:
    """
    Factory class to create appropriate document processor based on file type.
    """

    _processors = {}  # Will store our processor mappings

    @classmethod
    def register_processor(
        cls, file_extensions: list, processor_class: Type[BaseDocumentProcessor]
    ):
        """
        Register a processor class for specific file extensions.

        Args:
            file_extensions: List of file extensions (e.g., ['.pdf', '.PDF'])
            processor_class: The processor class to handle these extensions
        """
        logger.info(
            f"Registering processor {processor_class.__name__} for extensions: {file_extensions}"
        )
        for ext in file_extensions:
            cls._processors[ext.lower()] = processor_class
        logger.info(f"Current registered processors: {list(cls._processors.keys())}")

    @classmethod
    def get_processor(cls, file_ext: str) -> BaseDocumentProcessor:
        """
        Get appropriate processor for a file.

        Args:
            file_path: Path to the document file

        Returns:
            Instance of appropriate document processor

        Raises:
            ValueError: If no processor is registered for the file type
        """
        logger.info(f"Attempting to get processor for extension: {file_ext}")
        logger.info(f"Available processors: {list(cls._processors.keys())}")

        file_ext = file_ext.lower()
        processor_class = cls._processors.get(file_ext)

        if not processor_class:
            supported_formats = list(cls._processors.keys())
            raise ValueError(
                f"Unsupported file type: {file_ext}. Supported formats: {supported_formats}"
            )

        logger.info(f"Found processor {processor_class.__name__} for {file_ext}")
        return processor_class()
