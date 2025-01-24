"""
Evaluate our semantic chunking strategy using the external evaluation framework.
This script compares our chunking approach with other methods using standardized metrics.
"""

from chunking_evaluation import BaseChunker, GeneralEvaluation
from chromadb.utils import embedding_functions
import re
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveChunker(BaseChunker):
    """
    Adapter class that implements our adaptive chunking strategy
    for the external evaluation framework.
    """

    def __init__(
        self,
        max_size: int = 1200,
        min_size: int = 400,
        overlap: int = 200,
    ):
        self.max_size = max_size
        self.min_size = min_size
        self.overlap = overlap
        logger.info(
            f"Initialized AdaptiveChunker with max_size={max_size}, "
            f"min_size={min_size}, overlap={overlap}"
        )

    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into sections by natural breaks."""
        sections = text.split("\n\n")  # Split by paragraph breaks
        return [s for s in sections if s.strip()]  # Remove empty sections

    def split_text(self, text: str) -> List[str]:
        """
        Implement the required interface method for the evaluation framework.
        Uses our adaptive chunking strategy.
        """
        import sys
        from pathlib import Path

        # Add the project root to Python path
        project_root = Path(__file__).parent.parent.parent
        sys.path.append(str(project_root))

        from app.services.chunker import DocumentProcessor
        from app.services.embeddings import EmbeddingService

        # Create a temporary embedding service for the chunker
        embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")

        # Initialize document processor with our settings
        processor = DocumentProcessor(
            embedding_service=embedding_service,
            max_chunk_size=self.max_size,
            chunk_overlap=self.overlap,
        )

        # Process the document and extract chunks
        processed_chunks = processor.process_document(text, {"source": "evaluation"})
        chunks = [chunk.content for chunk in processed_chunks]

        return chunks


def main():
    """Run the evaluation with different adaptive configurations."""
    # Test configurations
    configs = [
        {
            "max_size": 1200,
            "min_size": 400,
            "overlap": 200,
        },  # Default adaptive
        # {
        #     "max_size": 1500,
        #     "min_size": 500,
        #     "overlap": 250,
        # },  # Larger chunks
        # {
        #     "max_size": 900,
        #     "min_size": 300,
        #     "overlap": 150,
        # },  # Smaller chunks
    ]

    # Initialize evaluation and embedding function
    evaluation = GeneralEvaluation()
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Run evaluation for each configuration
    results = {}
    for config in configs:
        chunker = AdaptiveChunker(
            max_size=config["max_size"],
            min_size=config["min_size"],
            overlap=config["overlap"],
        )

        logger.info(
            f"\nEvaluating configuration: max_size={config['max_size']}, "
            f"min_size={config['min_size']}, overlap={config['overlap']}"
        )

        try:
            result = evaluation.run(chunker, embedding_function)
            config_name = (
                f"max_{config['max_size']}_min_{config['min_size']}_"
                f"overlap_{config['overlap']}"
            )
            results[config_name] = result

            logger.info("Results:")
            logger.info(f"IOU Mean: {result['iou_mean']:.4f}")
            logger.info(f"IOU Std: {result['iou_std']:.4f}")
            logger.info(f"Recall Mean: {result['recall_mean']:.4f}")
            logger.info(f"Recall Std: {result['recall_std']:.4f}")

        except Exception as e:
            logger.error(f"Error evaluating configuration: {str(e)}")

    # Print comparative results with more detail
    logger.info("\nComparative Results:")
    for config_name, result in results.items():
        logger.info(f"\n{config_name}:")
        logger.info(f"IOU Mean: {result['iou_mean']:.4f}")
        logger.info(f"IOU Std: {result['iou_std']:.4f}")
        logger.info(f"Recall Mean: {result['recall_mean']:.4f}")
        logger.info(f"Recall Std: {result['recall_std']:.4f}")


if __name__ == "__main__":
    main()
