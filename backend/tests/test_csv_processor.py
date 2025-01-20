# tests/test_csv_processor.py

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.document_processor.csv_processor import CSVProcessor
from app.services.document_processor.factory import DocumentProcessorFactory


def test_csv_processing():
    # Print current directory and backend directory for debugging
    logger.debug(f"Current working directory: {Path.cwd()}")
    logger.debug(f"Backend directory: {backend_dir}")

    # Construct path to CSV
    csv_path = (
        backend_dir
        / "data"
        / "test_documents"
        / "test_formats"
        / "mock_customer_data.csv"
    )
    logger.debug(f"Looking for CSV at: {csv_path}")

    # Verify file exists
    if not csv_path.exists():
        logger.error(f"CSV file not found at: {csv_path}")
        return

    logger.info(f"Found CSV file at: {csv_path}")

    # Register CSV processor
    DocumentProcessorFactory.register_processor([".csv", ".CSV"], CSVProcessor)

    # Get processor through factory
    processor = DocumentProcessorFactory.get_processor(str(csv_path))

    # Extract text and metadata
    text = processor.extract_text(str(csv_path))
    metadata = processor.extract_metadata(str(csv_path))

    # Print results
    print("\nExtracted Text Preview (first 500 chars):")
    print(text[:500] + "...\n")
    print("Metadata:")
    print(metadata)


if __name__ == "__main__":
    test_csv_processing()
