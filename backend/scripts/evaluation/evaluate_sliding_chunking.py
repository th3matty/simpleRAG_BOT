import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent.parent
sys.path.append(str(backend_dir))

import os

os.environ["ENV_FILE"] = str(backend_dir / ".env")

from chromadb.utils import embedding_functions
from app.core.config import settings
from chunking_evaluation import BaseChunker, GeneralEvaluation
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlidingWindowChunker(BaseChunker):
    def __init__(self, chunk_size=1200, overlap_size=400):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        # Ensure overlap is not larger than chunk size
        if overlap_size >= chunk_size:
            raise ValueError("Overlap size must be smaller than chunk size")
        logger.info(
            f"Initialized SlidingWindowChunker with chunk_size={chunk_size}, overlap_size={overlap_size}"
        )

    def find_sentence_boundary(self, text, position, max_lookforward=200):
        """Find the nearest sentence boundary after the given position.

        Args:
            text: The input text
            position: Current position in text
            max_lookforward: Maximum characters to look ahead

        Returns:
            Position after the next sentence boundary or original position
        """
        # Handle case where position is already at the end
        if position >= len(text):
            return position

        # Look ahead for sentence boundary, but limit the search range
        search_text = text[position : position + max_lookforward]

        # More comprehensive sentence boundary detection
        # Handles common abbreviations and multiple punctuation marks
        sentence_end = re.search(
            r"(?<![A-Z][a-z]\.)(?<!\sw)(?<!\sdr)(?<!\smr)(?<!\sms)(?<!\smrs)(?<!\sphd)[.!?][\s\n]+",
            search_text.lower(),
        )

        if sentence_end:
            return position + sentence_end.end()

        # If no boundary found within max_lookforward, use chunk_size
        return position

    def split_text(self, text):
        logger.info(f"Splitting text of length {len(text)}")
        chunks = []
        start = 0

        while start < len(text):
            # Calculate the initial end position
            end = min(start + self.chunk_size, len(text))

            # Don't look for sentence boundary if we're at the end
            if end == len(text):
                chunks.append(text[start:])
                break

            # Find next sentence boundary
            adjusted_end = self.find_sentence_boundary(text, end)

            # Create chunk
            current_chunk = text[start:adjusted_end]
            chunks.append(current_chunk)

            # Calculate next start position
            # Ensure we move forward even if no sentence boundary found
            if adjusted_end == end:
                # No sentence boundary found, move by chunk_size - overlap
                start = end - self.overlap_size
            else:
                # Sentence boundary found, apply overlap from adjusted position
                start = adjusted_end - self.overlap_size

            # Ensure we always move forward
            start = max(start, end - self.chunk_size)

            logger.debug(
                f"Created chunk from {start} to {adjusted_end} "
                f"(length: {adjusted_end-start})"
            )

        logger.info(f"Created {len(chunks)} chunks")
        return chunks


def run_evaluation():
    try:
        logger.info("Starting evaluation process...")

        logger.info("Initializing evaluation...")
        evaluation = GeneralEvaluation()
        logger.info("Evaluation initialized")

        logger.info(
            f"Initializing embedding function with model {settings.embedding_model}..."
        )
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        logger.info("Embedding function initialized")

        logger.info("Initializing SlidingWindowChunker...")
        chunker = SlidingWindowChunker()
        logger.info("Chunker initialized")

        logger.info("Starting evaluation run...")
        try:
            results = evaluation.run(chunker, embedding_function)
            logger.info("Evaluation completed successfully")
        except Exception as eval_error:
            logger.error(f"Error during evaluation.run(): {str(eval_error)}")
            raise

        logger.info("\nEvaluation Results:")
        logger.info("-" * 50)
        logger.info(f"IOU Mean: {results['iou_mean']:.4f}")
        logger.info(f"IOU Std: {results['iou_std']:.4f}")
        logger.info(f"Recall Mean: {results['recall_mean']:.4f}")
        logger.info(f"Recall Std: {results['recall_std']:.4f}")

    except Exception as e:
        logger.error(f"Error during evaluation process: {str(e)}")
        import traceback

        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)
