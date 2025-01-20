# app/services/document_processor/pdf_processor.py

from .base import BaseDocumentProcessor
from typing import Dict, Any
import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFProcessor(BaseDocumentProcessor):
    """
    Processor for PDF documents using pypdf.
    """

    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content

        Raises:
            Exception: If PDF processing fails
        """
        try:
            logger.info(f"Starting text extraction from PDF: {file_path}")

            reader = PdfReader(file_path)
            text_content = []

            # Extract text from each page
            for page_num, page in enumerate(reader.pages):
                logger.debug(f"Processing page {page_num + 1}")
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    text_content.append(text)

            # Join all text with double newlines between pages
            full_text = "\n\n".join(text_content)

            logger.info(f"Successfully extracted {len(reader.pages)} pages")
            return full_text

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing PDF metadata

        Raises:
            Exception: If metadata extraction fails
        """
        try:
            logger.info(f"Extracting metadata from PDF: {file_path}")

            reader = PdfReader(file_path)
            metadata = {}

            # Extract basic metadata
            print(str(reader.metadata))
            if reader.metadata:
                metadata.update(
                    {
                        "title": reader.metadata.get("/Title", ""),
                        "author": reader.metadata.get("/Author", ""),
                        "subject": reader.metadata.get("/Subject", ""),
                        "creator": reader.metadata.get("/Creator", ""),
                        "producer": reader.metadata.get("/Producer", ""),
                    }
                )

            # Add additional information
            metadata.update({"page_count": len(reader.pages), "file_type": "pdf"})

            logger.info("Successfully extracted PDF metadata")
            logger.debug(f"Extracted metadata: {metadata}")

            return metadata

        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {str(e)}")
            raise
