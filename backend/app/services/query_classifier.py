"""
Service for classifying queries into different types based on patterns and rules.
This helps optimize retrieval by applying appropriate strategies per query type.
"""

import re
from enum import Enum
from typing import Tuple


class QueryType(Enum):
    """Enumeration of possible query types."""

    FACTUAL = "factual"  # When, Who, Where questions seeking specific facts
    DEFINITION = "definition"  # What is, Define, Explain questions
    CONTEXT = "context"  # Why, How questions seeking broader context


class QueryClassifier:
    """
    Classifies queries into different types based on patterns and rules.
    This helps optimize retrieval strategies based on query intent.
    """

    # Patterns for different query types
    FACTUAL_PATTERNS = [
        # Time-based patterns
        r"^wann\s",  # When questions
        r"seit\swann",  # Since when
        r"ab\swann",  # From when
        r"\d{4}",  # Years
        # Entity-based patterns
        r"^wo\s",  # Where questions
        r"^wer\s",  # Who questions
        r"^welche[rs]?\s",  # Which questions
        r"^von\swem",  # By whom
        # Action patterns
        r"wurde",  # Past tense indicators
        r"hat.*verwendet",  # Usage questions
        r"nutzte",  # Usage in past
        r"veröffentlichte",  # Publication
        r"erschien",  # Appearance/publication
        # Event patterns
        r"erstmals",  # First occurrence
        r"zuerst",  # First time
        r"zum\sersten\smal",  # For the first time
    ]

    DEFINITION_PATTERNS = [
        r"^was (ist|sind|bedeutet)",  # What is/are questions
        r"^definiere",  # Define questions
        r"^erkläre",  # Explain questions
        r"^beschreibe",  # Describe questions
        r"bedeutung",  # Meaning questions
        r"definition",  # Definition questions
    ]

    CONTEXT_PATTERNS = [
        r"^warum",  # Why questions
        r"^wie\s",  # How questions
        r"^inwiefern",  # In what way questions
        r"unterschied",  # Difference/comparison indicators
        r"zusammenhang",  # Relationship indicators
        r"kontext",  # Context questions
        r"hintergrund",  # Background questions
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self.factual_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.FACTUAL_PATTERNS
        ]
        self.definition_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DEFINITION_PATTERNS
        ]
        self.context_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CONTEXT_PATTERNS
        ]

    def classify(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify a query into a specific type with confidence score.

        Args:
            query: The query string to classify

        Returns:
            Tuple of (QueryType, confidence_score)
            Confidence score ranges from 0.0 to 1.0
        """
        query = query.strip().lower()

        # Check patterns for each type with weighted scoring
        factual_matches = sum(
            2 if p.match(query) else 1 for p in self.factual_patterns if p.search(query)
        )
        definition_matches = sum(
            2 if p.match(query) else 1
            for p in self.definition_patterns
            if p.search(query)
        )
        context_matches = sum(
            2 if p.match(query) else 1 for p in self.context_patterns if p.search(query)
        )

        # Calculate confidence scores
        total_matches = factual_matches + definition_matches + context_matches
        if total_matches == 0:
            # Default to FACTUAL with low confidence if no patterns match
            return QueryType.FACTUAL, 0.3

        # Determine the dominant type
        scores = [
            (QueryType.FACTUAL, factual_matches),
            (QueryType.DEFINITION, definition_matches),
            (QueryType.CONTEXT, context_matches),
        ]

        # Sort by number of matches, highest first
        scores.sort(key=lambda x: x[1], reverse=True)

        # Calculate confidence score with higher weight for exact matches
        confidence = scores[0][1] / (total_matches * 1.5) if total_matches > 0 else 0.3

        return scores[0][0], min(1.0, confidence)

    def get_recommended_threshold(self, query_type: QueryType) -> float:
        """
        Get the recommended similarity threshold for a query type.

        Args:
            query_type: The type of query

        Returns:
            Recommended similarity threshold value
        """
        thresholds = {
            QueryType.FACTUAL: 0.45,  # Slightly more permissive for facts
            QueryType.DEFINITION: 0.4,  # Medium threshold for definitions
            QueryType.CONTEXT: 0.3,  # Lower threshold for contextual queries
        }
        return thresholds.get(query_type, 0.5)  # Default to 0.5 if type unknown
