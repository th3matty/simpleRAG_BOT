# tests/test_csv_processor.py

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
from app.services.document_processor.csv_processor import CSVProcessor
from app.services.document_processor.factory import DocumentProcessorFactory


def test_csv_processing():
    try:
        # Register CSV processor
        DocumentProcessorFactory.register_processor([".csv", ".CSV"], CSVProcessor)

        # Get absolute path to CSV
        csv_path = (
            backend_dir
            / "data"
            / "test_documents"
            / "test_formats"
            / "mock_customer_data.csv"
        )
        logger.debug(f"Testing CSV at path: {csv_path}")

        if not csv_path.exists():
            logger.error(f"CSV file not found at: {csv_path}")
            return

        # Get processor through factory using file extension
        file_ext = csv_path.suffix.lower()
        processor = DocumentProcessorFactory.get_processor(file_ext)

        # Extract text and metadata
        logger.info("Extracting text from CSV...")
        text = processor.extract_text(str(csv_path))

        logger.info("Extracting metadata from CSV...")
        metadata = processor.extract_metadata(str(csv_path))

        # Print results
        print("\nExtracted Text Preview:")
        print(text[:500] + "...\n")
        print("Metadata:")
        print(metadata)

    except Exception as e:
        logger.error(f"Error in test_csv_processing: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    test_csv_processing()
