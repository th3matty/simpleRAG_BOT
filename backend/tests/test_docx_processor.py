# tests/test_docx_processor.py

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.document_processor.docx_processor import DOCXProcessor
from app.services.document_processor.factory import DocumentProcessorFactory


def test_docx_processing():
    # Print current directory and backend directory for debugging
    logger.debug(f"Current working directory: {Path.cwd()}")
    logger.debug(f"Backend directory: {backend_dir}")

    # Construct path to DOCX
    docx_path = (
        backend_dir
        / "data"
        / "test_documents"
        / "test_formats"
        / "CV Matthäus Malek_12_24.docx"
    )
    logger.debug(f"Looking for DOCX at: {docx_path}")

    # Verify file exists
    if not docx_path.exists():
        logger.error(f"DOCX file not found at: {docx_path}")
        return

    logger.info(f"Found DOCX file at: {docx_path}")

    # Register DOCX processor
    DocumentProcessorFactory.register_processor([".docx", ".DOCX"], DOCXProcessor)

    # Get processor through factory
    processor = DocumentProcessorFactory.get_processor(str(docx_path))

    # Extract text and metadata
    text = processor.extract_text(str(docx_path))
    metadata = processor.extract_metadata(str(docx_path))

    # Print results
    print("\nExtracted Text Preview:")
    print(text[:500] + "...\n")
    print("Metadata:")
    print(metadata)


if __name__ == "__main__":
    test_docx_processing()
