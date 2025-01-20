# tests/test_pdf_processor.py

import sys
from pathlib import Path
import logging

# Set up Python path for imports
script_dir = Path(__file__).parent
backend_dir = script_dir.parent
sys.path.insert(0, str(backend_dir))

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Now we can import our modules
from app.services.document_processor.pdf_processor import PDFProcessor
from app.services.document_processor.factory import DocumentProcessorFactory


def test_pdf_processing():
    try:
        # Register PDF processor
        DocumentProcessorFactory.register_processor([".pdf", ".PDF"], PDFProcessor)

        # Get absolute path to PDF
        pdf_path = (
            backend_dir
            / "data"
            / "test_documents"
            / "test_formats"
            / "Lebenslauf_mitAzureCert_07_24.pdf"
        )
        logger.debug(f"Testing PDF at path: {pdf_path}")

        if not pdf_path.exists():
            logger.error(f"PDF file not found at: {pdf_path}")
            return

        # Get processor through factory using file extension
        file_ext = pdf_path.suffix.lower()
        processor = DocumentProcessorFactory.get_processor(file_ext)

        # Extract text and metadata
        logger.info("Extracting text from PDF...")
        text = processor.extract_text(str(pdf_path))

        logger.info("Extracting metadata from PDF...")
        metadata = processor.extract_metadata(str(pdf_path))

        # Print results
        print("\nExtracted Text Preview:")
        print(text[:500] + "...\n")
        print("Metadata:")
        print(metadata)

    except Exception as e:
        logger.error(f"Error in test_pdf_processing: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    test_pdf_processing()
