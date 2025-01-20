# app/services/document_processor/__init__.py
import logging
from .factory import DocumentProcessorFactory
from .pdf_processor import PDFProcessor
from .docx_processor import DOCXProcessor
from .csv_processor import CSVProcessor

logger = logging.getLogger(__name__)


def register_processors():
    """Register all document processors."""
    logger.info("Starting processor registration")

    processors_map = {
        PDFProcessor: [".pdf", ".PDF"],
        DOCXProcessor: [".docx", ".DOCX"],
        CSVProcessor: [".csv", ".CSV"],
    }

    for processor_class, extensions in processors_map.items():
        logger.info(
            f"Registering {processor_class.__name__} for extensions: {extensions}"
        )
        DocumentProcessorFactory.register_processor(extensions, processor_class)

    logger.info(
        f"Available processors: {list(DocumentProcessorFactory._processors.keys())}"
    )


# Optional: Register processors immediately for testing
register_processors()
