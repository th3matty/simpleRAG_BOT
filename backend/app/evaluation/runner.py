from typing import List, Dict, Any
import logging
from app.evaluation.metrics import RetrievalMetrics, EvaluationResults
from app.evaluation.generate_test_cases import TestCase
from app.core.database import db
from app.core.config import settings
from app.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Runs test cases through our retrieval system and evaluates performance.
    This class connects our test cases with our metrics calculation.
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def _calculate_mrr(
        self, retrieved_docs: List[Dict[str, Any]], expected_docs: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        Returns 1/rank of first relevant document, or 0 if none found.
        """
        for i, retrieved in enumerate(retrieved_docs, 1):
            for expected in expected_docs:
                if self._texts_match(retrieved["content"], expected["content"]):
                    return 1.0 / i
        return 0.0

    def _texts_match(self, text1: str, text2: str, threshold: float = 0.3) -> bool:
        """Determine if two texts match based on word overlap using Jaccard similarity."""

        def normalize_text(text: str) -> str:
            return " ".join(text.lower().strip().split())

        text1, text2 = normalize_text(text1), normalize_text(text2)
        words1 = set(text1.split())
        words2 = set(text2.split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold

    def _calculate_metrics_for_query(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        expected_docs: List[Dict[str, Any]],
        query_type: str,
    ) -> RetrievalMetrics:
        """
        Calculate metrics for a single query by comparing retrieved and expected documents.
        Uses text matching to determine relevance.
        """
        retrieved_contents = [doc["content"] for doc in retrieved_docs]
        expected_contents = [doc["content"] for doc in expected_docs]

        matches = sum(
            1
            for r in retrieved_contents
            for e in expected_contents
            if self._texts_match(r, e)
        )

        precision = matches / len(retrieved_contents) if retrieved_contents else 0
        recall = matches / len(expected_contents) if expected_contents else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return RetrievalMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mrr=self._calculate_mrr(retrieved_docs, expected_docs),
            relevance_scores=[doc.get("score", 0) for doc in retrieved_docs],
            query_type=query_type,
        )

    async def evaluate_test_cases(
        self, test_cases: List[TestCase]
    ) -> EvaluationResults:
        """
        Run all test cases through the retrieval system and evaluate performance.
        This is the main method that orchestrates the evaluation process.
        """
        metrics_by_query = {}

        for test_case in test_cases:
            logger.info(f"\nProcessing test case: {test_case.query}")
            logger.info(
                f"Expected content: {test_case.expected_docs[0]['content'][:100]}..."
            )

            query_embedding = self.embedding_service.get_single_embedding(
                test_case.query
            )

            results = db.query_documents(
                query_embedding=query_embedding, n_results=settings.top_k_results
            )

            logger.info(f"Retrieved {len(results['documents'][0])} documents")
            for doc, score in zip(results["documents"][0], results["distances"][0]):
                logger.info(f"Retrieved doc (score={score}): {doc[:100]}...")

            retrieved_docs = [
                {
                    "content": doc,
                    "metadata": meta,
                    "score": (2 - distance) / 2,  # Convert distance to similarity
                }
                for doc, meta, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]

            metrics = self._calculate_metrics_for_query(
                query=test_case.query,
                retrieved_docs=retrieved_docs,
                expected_docs=test_case.expected_docs,
                query_type=test_case.query_type,
            )

            metrics_by_query[test_case.query] = metrics

        all_metrics = list(metrics_by_query.values())

        return EvaluationResults(
            metrics_by_query=metrics_by_query,
            avg_precision=sum(m.precision for m in all_metrics) / len(all_metrics),
            avg_recall=sum(m.recall for m in all_metrics) / len(all_metrics),
            avg_f1=sum(m.f1_score for m in all_metrics) / len(all_metrics),
            avg_mrr=sum(m.mrr for m in all_metrics) / len(all_metrics),
        )
