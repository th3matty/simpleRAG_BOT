# tests/test_pdf_processor.py

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.document_processor.pdf_processor import PDFProcessor
from app.services.document_processor.factory import DocumentProcessorFactory


def test_pdf_processing():
    # Print current directory and backend directory for debugging
    logger.debug(f"Current working directory: {Path.cwd()}")
    logger.debug(f"Backend directory: {backend_dir}")

    # Construct correct path to PDF
    pdf_path = (
        backend_dir
        / "data"
        / "test_documents"
        / "test_formats"
        / "Lebenslauf_mitAzureCert_07_24.pdf"
    )
    logger.debug(f"Looking for PDF at: {pdf_path}")

    # Verify file exists
    if not pdf_path.exists():
        logger.error(f"PDF file not found at: {pdf_path}")
        return

    logger.info(f"Found PDF file at: {pdf_path}")

    # Register PDF processor
    DocumentProcessorFactory.register_processor([".pdf", ".PDF"], PDFProcessor)

    # Get processor through factory
    processor = DocumentProcessorFactory.get_processor(str(pdf_path))

    # Extract text and metadata
    text = processor.extract_text(str(pdf_path))
    metadata = processor.extract_metadata(str(pdf_path))

    # Print results
    print("\nExtracted Text Preview:")
    print(text[:500] + "...\n")
    print("Metadata:")
    print(metadata)


if __name__ == "__main__":
    test_pdf_processing()
