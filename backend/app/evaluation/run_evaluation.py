import os
import sys
from pathlib import Path
import asyncio
import logging

# Add the backend directory to sys.path
current_file = Path(__file__).resolve()
backend_dir = current_file.parent.parent.parent

# Set up environment file path
os.environ["ENV_FILE"] = str(backend_dir / ".env")

sys.path.insert(0, str(backend_dir))

from app.evaluation.runner import EvaluationRunner
from app.evaluation.generate_test_cases import generate_test_cases_from_article1
from app.services.embeddings import EmbeddingService
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    try:
        # Initialize services
        embedding_service = EmbeddingService(settings.embedding_model)
        runner = EvaluationRunner(embedding_service)

        # Generate test cases
        test_cases = generate_test_cases_from_article1()
        logger.info(f"Generated {len(test_cases)} test cases")

        # Run evaluation
        results = await runner.evaluate_test_cases(test_cases)

        # Print results
        print("\nEvaluation Results:")
        print("=" * 50)
        print(results)

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
