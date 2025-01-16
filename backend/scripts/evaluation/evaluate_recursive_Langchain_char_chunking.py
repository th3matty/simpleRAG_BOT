import sys
from pathlib import Path
from typing import List

# Add the backend directory to sys.path
script_dir = Path(__file__).parent
backend_dir = script_dir.parent.parent
project_root = backend_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_dir))

import os

os.environ["ENV_FILE"] = str(backend_dir / ".env")

from chromadb.utils import embedding_functions
from app.core.config import settings
from chunking_evaluation import BaseChunker, GeneralEvaluation
import logging
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logging with a clear format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LangChainRecursiveChunker(BaseChunker):
    """
    A wrapper class that adapts LangChain's RecursiveCharacterTextSplitter
    to work with our evaluation framework. This chunker splits text recursively
    using a hierarchy of separators, similar to how humans would naturally
    break down text.
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        separators: List[str] = None,
    ):
        """
        Initialize the LangChain recursive chunker with customizable parameters.

        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: Custom separators to use for splitting (optional)
        """
        super().__init__()

        # If no separators provided, use LangChain's default hierarchy
        # but modified to better handle sentence structures
        if separators is None:
            separators = [
                "\n\n",  # Paragraph breaks
                "\n",  # Line breaks
                ".",  # End of sentences
                "?",  # Question marks
                "!",  # Exclamation marks
                " ",  # Word boundaries
                "",  # Character level (last resort)
            ]

        # Initialize the LangChain text splitter with our parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            keep_separator=True,
            is_separator_regex=False,
        )

        logger.info(
            f"Initialized LangChain Recursive Chunker with "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split input text into chunks using LangChain's recursive splitter.

        Args:
            text: The input text to be split

        Returns:
            List of text chunks
        """
        logger.info(f"Splitting text of length {len(text)}")

        # Use LangChain's create_documents method and extract the page content
        documents = self.text_splitter.create_documents([text])
        chunks = [doc.page_content for doc in documents]

        logger.info(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i+1} length: {len(chunk)}")

        return chunks


def run_evaluation():
    """
    Run the evaluation process using our LangChain-based chunker.
    This function sets up the chunker, runs the evaluation, and
    reports the results with proper error handling.
    """
    try:
        logger.info("Starting evaluation process...")

        # Initialize our components
        chunker = LangChainRecursiveChunker(chunk_size=1200, chunk_overlap=200)
        evaluation = GeneralEvaluation()

        # Set up the embedding function
        logger.info(f"Initializing embedding function with {settings.embedding_model}")
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )

        # Run the evaluation
        logger.info("Running evaluation...")
        try:
            results = evaluation.run(chunker, embedding_function)
            logger.info("Evaluation completed successfully")
        except Exception as eval_error:
            logger.error("Evaluation failed with error:", str(eval_error))
            raise

        # Report results
        logger.info("\nEvaluation Results:")
        logger.info("-" * 50)
        logger.info(f"IOU Mean: {results['iou_mean']:.4f}")
        logger.info(f"IOU Std: {results['iou_std']:.4f}")
        logger.info(f"Recall Mean: {results['recall_mean']:.4f}")
        logger.info(f"Recall Std: {results['recall_std']:.4f}")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)
