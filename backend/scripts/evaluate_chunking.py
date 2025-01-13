import sys
from pathlib import Path

# Add the parent directory to sys.path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

# Set up environment file path
import os

os.environ["ENV_FILE"] = str(backend_dir / ".env")

import chromadb
from chromadb.utils import embedding_functions
from app.config import settings
from chunking_evaluation import BaseChunker, GeneralEvaluation
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChunker(BaseChunker):
    def __init__(self, chunk_size=1200):
        self.chunk_size = chunk_size
        logger.info(f"Initialized RAGChunker with chunk_size={chunk_size}")

    def split_text(self, text):
        logger.info(f"Splitting text of length {len(text)}")
        # Simple fixed-size chunking strategy
        chunks = [
            text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
        ]
        logger.info(f"Created {len(chunks)} chunks")
        return chunks


def run_evaluation():
    try:
        logger.info("Starting evaluation process...")

        logger.info("Initializing ChromaDB client...")
        client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        logger.info(
            f"ChromaDB client initialized at {settings.chroma_persist_directory}"
        )

        logger.info("Getting collection...")
        collection = client.get_collection("documents")
        logger.info("Collection retrieved successfully")

        logger.info("Retrieving documents from collection...")
        result = collection.get()
        documents = result["documents"]
        logger.info(f"Retrieved {len(documents)} documents")

        for i, doc in enumerate(documents):
            logger.info(f"Document {i+1} length: {len(doc)} characters")
            logger.debug(f"Document {i+1} preview: {doc[:100]}...")

        if not documents:
            logger.error("No documents found in the collection!")
            return

        logger.info("Initializing chunker and evaluation...")
        chunker = RAGChunker()
        evaluation = GeneralEvaluation()
        logger.info("Chunker and evaluation initialized")

        logger.info(
            f"Initializing embedding function with model {settings.embedding_model}..."
        )
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        logger.info("Embedding function initialized")

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
