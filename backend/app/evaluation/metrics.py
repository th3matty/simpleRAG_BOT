from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """
    Stores metrics for evaluating retrieval performance.
    Each metric tells us something different about the quality of results.
    """

    precision: float  # How many retrieved docs were relevant
    recall: float  # How many relevant docs were retrieved
    f1_score: float  # Balance between precision and recall
    mrr: float  # Mean Reciprocal Rank (position of first relevant doc)
    relevance_scores: List[float]  # Raw similarity scores
    query_type: str  # Type of query being evaluated

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for easy logging/storage."""
        return {
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1_score": round(self.f1_score, 3),
            "mrr": round(self.mrr, 3),
            "avg_relevance": round(np.mean(self.relevance_scores), 3),
            "query_type": self.query_type,
        }

    def __str__(self) -> str:
        """String representation for logging."""
        metrics = self.to_dict()
        return (
            f"Query Type: {metrics['query_type']}\n"
            f"Precision: {metrics['precision']}\n"
            f"Recall: {metrics['recall']}\n"
            f"F1 Score: {metrics['f1_score']}\n"
            f"MRR: {metrics['mrr']}\n"
            f"Avg Relevance: {metrics['avg_relevance']}"
        )


@dataclass
class EvaluationResults:
    """
    Aggregates metrics across multiple test cases.
    Helps track overall system performance.
    """

    metrics_by_query: Dict[str, RetrievalMetrics]  # Results for each query
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_mrr: float

    def get_metrics_by_type(self) -> Dict[str, Dict[str, float]]:
        """
        Group metrics by query type.
        Helps understand performance across different query categories.
        """
        # Group metrics by query type
        metrics_by_type = {}
        for query, metrics in self.metrics_by_query.items():
            query_type = metrics.query_type
            if query_type not in metrics_by_type:
                metrics_by_type[query_type] = {
                    "precision": [],
                    "recall": [],
                    "f1_score": [],
                    "mrr": [],
                }

            metrics_dict = metrics.to_dict()
            for key in ["precision", "recall", "f1_score", "mrr"]:
                metrics_by_type[query_type][key].append(metrics_dict[key])

        # Calculate averages for each type, handling empty lists
        for query_type in metrics_by_type:
            for key in metrics_by_type[query_type]:
                values = metrics_by_type[query_type][key]
                if values:  # Only calculate mean if we have values
                    metrics_by_type[query_type][key] = round(float(np.mean(values)), 3)
                else:
                    metrics_by_type[query_type][
                        key
                    ] = 0.0  # Default to 0 for empty lists

        return metrics_by_type

    def __str__(self) -> str:
        """Detailed string representation of results."""
        output = [
            "Overall Metrics:",
            f"Average Precision: {round(self.avg_precision, 3)}",
            f"Average Recall: {round(self.avg_recall, 3)}",
            f"Average F1: {round(self.avg_f1, 3)}",
            f"Average MRR: {round(self.avg_mrr, 3)}",
            "\nMetrics by Query Type:",
        ]

        for query_type, metrics in self.get_metrics_by_type().items():
            output.extend(
                [
                    f"\n{query_type}:",
                    f"  Precision: {metrics['precision']}",
                    f"  Recall: {metrics['recall']}",
                    f"  F1 Score: {metrics['f1_score']}",
                    f"  MRR: {metrics['mrr']}",
                ]
            )

        return "\n".join(output)
