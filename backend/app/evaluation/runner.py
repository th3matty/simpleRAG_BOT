from typing import List, Dict, Any
import logging
from app.evaluation.metrics import RetrievalMetrics, EvaluationResults
from app.evaluation.generate_test_cases import TestCase
from app.core.database import db
from app.core.config import settings
from app.services.embeddings import EmbeddingService
from app.services.query_classifier import QueryClassifier

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Runs test cases through our retrieval system and evaluates performance.
    This class connects our test cases with our metrics calculation.
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.query_classifier = QueryClassifier()

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
        """
        Determine if two texts match using a combination of:
        1. Key term matching for factual information
        2. Word overlap for general similarity
        """

        def normalize_text(text: str) -> str:
            return " ".join(text.lower().strip().split())

        text1, text2 = normalize_text(text1), normalize_text(text2)

        # Extract key terms (dates, proper nouns, numbers)
        import re

        def extract_key_terms(text: str) -> set:
            # Match dates (e.g., 1990er, 2017)
            dates = set(re.findall(r"\d{4}(?:er)?", text))
            # Match capitalized terms (potential proper nouns)
            proper_nouns = set(
                word for word in text.split() if any(c.isupper() for c in word)
            )
            # Match specific terms we care about
            important_terms = {
                "biodeutsch",
                "bio-deutsch",
                "duden",
                "npd",
                "kanak",
                "attak",
            }
            return dates | proper_nouns | (important_terms & set(text.split()))

        # Get key terms from both texts
        terms1 = extract_key_terms(text1)
        terms2 = extract_key_terms(text2)

        # If we have key terms, they should match
        if terms1 and terms2:
            term_overlap = len(terms1 & terms2) / len(terms1 | terms2)
            if term_overlap < 0.2:  # At least 20% of key terms should match
                return False

        # Calculate word overlap using sliding window for longer texts
        words1 = text1.split()
        words2 = text2.split()

        # Use smaller window size for shorter text
        window_size = min(len(words1), len(words2), 10)

        max_overlap = 0
        for i in range(len(words1) - window_size + 1):
            window1 = set(words1[i : i + window_size])
            for j in range(len(words2) - window_size + 1):
                window2 = set(words2[j : j + window_size])
                overlap = len(window1 & window2) / len(window1 | window2)
                max_overlap = max(max_overlap, overlap)

        return max_overlap >= threshold

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

            # Get query-specific threshold
            query_type, confidence = self.query_classifier.classify(test_case.query)
            threshold = self.query_classifier.get_recommended_threshold(query_type)
            logger.info(
                f"Query classified as {query_type.value} (confidence: {confidence:.2f})"
            )
            logger.info(f"Using similarity threshold: {threshold}")

            results = db.query_documents(
                query_embedding=query_embedding,
                n_results=settings.top_k_results,
                similarity_threshold=threshold,
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
