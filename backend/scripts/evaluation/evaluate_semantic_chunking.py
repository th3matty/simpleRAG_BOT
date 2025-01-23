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


class SemanticChunker(BaseChunker):
    """
    Adapter class that implements our semantic chunking strategy
    for the external evaluation framework.
    """

    def __init__(self, max_chunk_size: int = 800, chunk_overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"Initialized SemanticChunker with max_chunk_size={max_chunk_size}, "
            f"overlap={chunk_overlap}"
        )

    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into sections by natural breaks."""
        sections = text.split("\n\n")  # Split by paragraph breaks
        return [s for s in sections if s.strip()]  # Remove empty sections

    def split_text(self, text: str) -> List[str]:
        """
        Implement the required interface method for the evaluation framework.
        Uses our semantic chunking strategy.
        """
        chunks = []
        sections = self._split_into_sections(text)

        for section_idx, section in enumerate(sections):
            logger.debug(f"Processing section {section_idx + 1} of {len(sections)}")

            paragraphs = section.split("\n\n")
            current_chunk = []
            current_length = 0

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                paragraph_length = len(paragraph)

                # If this paragraph would exceed the chunk size
                if current_length + paragraph_length > self.max_chunk_size:
                    if current_chunk:
                        # Save current chunk
                        chunk_text = "\n\n".join(current_chunk)
                        chunks.append(chunk_text)

                        # Keep overlap from the end of previous chunk
                        overlap_text = chunk_text[-self.chunk_overlap :]
                        current_chunk = [overlap_text]
                        current_length = len(overlap_text)

                    # Handle large paragraphs
                    if paragraph_length > self.max_chunk_size:
                        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                        sentence_chunk = []
                        sentence_length = 0

                        for sentence in sentences:
                            if sentence_length + len(sentence) > self.max_chunk_size:
                                if sentence_chunk:
                                    chunks.append(" ".join(sentence_chunk))

                                    # Keep last sentence for overlap
                                    sentence_chunk = [sentence_chunk[-1], sentence]
                                    sentence_length = sum(
                                        len(s) for s in sentence_chunk
                                    )
                            else:
                                sentence_chunk.append(sentence)
                                sentence_length += len(sentence) + 1

                        if sentence_chunk:
                            chunks.append(" ".join(sentence_chunk))
                    else:
                        current_chunk = [paragraph]
                        current_length = paragraph_length
                else:
                    current_chunk.append(paragraph)
                    current_length += paragraph_length + 2  # +2 for paragraph separator

            # Add remaining content
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(chunk_text)

        logger.info(f"Created {len(chunks)} chunks")
        return chunks


def main():
    """Run the evaluation with different chunk sizes and overlaps."""
    # Test configurations
    configs = [
        {"size": 800, "overlap": 200},  # Our default configuration
        {"size": 1000, "overlap": 200},  # Larger chunks
        {"size": 600, "overlap": 150},  # Smaller chunks
    ]

    # Initialize evaluation
    evaluation = GeneralEvaluation()

    # Use sentence-transformers for embeddings (matches our production setup)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Run evaluation for each configuration
    results = {}
    for config in configs:
        chunker = SemanticChunker(
            max_chunk_size=config["size"], chunk_overlap=config["overlap"]
        )

        logger.info(
            f"\nEvaluating configuration: chunk_size={config['size']}, "
            f"overlap={config['overlap']}"
        )

        try:
            result = evaluation.run(chunker, embedding_function)
            results[f"size_{config['size']}_overlap_{config['overlap']}"] = result

            logger.info("Results:")
            logger.info(f"IOU Mean: {result['iou_mean']:.4f}")
            logger.info(f"IOU Std: {result['iou_std']:.4f}")
            logger.info(f"Recall Mean: {result['recall_mean']:.4f}")
            logger.info(f"Recall Std: {result['recall_std']:.4f}")

        except Exception as e:
            logger.error(f"Error evaluating configuration: {str(e)}")

    # Print comparative results
    logger.info("\nComparative Results:")
    for config_name, result in results.items():
        logger.info(f"\n{config_name}:")
        logger.info(f"IOU Mean: {result['iou_mean']:.4f}")
        logger.info(f"Recall Mean: {result['recall_mean']:.4f}")


if __name__ == "__main__":
    main()
