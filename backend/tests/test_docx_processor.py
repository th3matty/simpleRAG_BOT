# tests/test_docx_processor.py

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
from app.services.document_processor.docx_processor import DOCXProcessor
from app.services.document_processor.factory import DocumentProcessorFactory


def test_docx_processing():
    try:
        # Register DOCX processor
        DocumentProcessorFactory.register_processor([".docx", ".DOCX"], DOCXProcessor)

        # Get absolute path to DOCX
        docx_path = (
            backend_dir
            / "data"
            / "test_documents"
            / "test_formats"
            / "CV Matthäus Malek_12_24.docx"
        )
        logger.debug(f"Testing DOCX at path: {docx_path}")

        if not docx_path.exists():
            logger.error(f"DOCX file not found at: {docx_path}")
            return

        # Get processor through factory using file extension
        file_ext = docx_path.suffix.lower()
        processor = DocumentProcessorFactory.get_processor(file_ext)

        # Extract text and metadata
        logger.info("Extracting text from DOCX...")
        text = processor.extract_text(str(docx_path))

        logger.info("Extracting metadata from DOCX...")
        metadata = processor.extract_metadata(str(docx_path))

        # Print results
        print("\nExtracted Text Preview:")
        print(text[:500] + "...\n")
        print("Metadata:")
        print(metadata)

    except Exception as e:
        logger.error(f"Error in test_docx_processing: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    test_docx_processing()
