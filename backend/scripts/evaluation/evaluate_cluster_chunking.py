import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent.parent
sys.path.append(str(backend_dir))

import os

os.environ["ENV_FILE"] = str(backend_dir / ".env")

from chromadb.utils import embedding_functions
from app.core.config import settings
from chunking_evaluation import GeneralEvaluation
from chunking_evaluation import ClusterSemanticChunker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        logger.info("Initializing ClusterSemanticChunker...")
        chunker = ClusterSemanticChunker(embedding_function, max_chunk_size=400)
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
