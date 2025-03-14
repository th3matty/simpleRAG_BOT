"""
Tools configuration for the LLM service.
This module defines the available tools and their schemas.
"""

import json
from typing import Dict, Any, List, Optional
import logging
from .config import settings
from ..services.calculator import Calculator
from .database import db
from ..services.embeddings import EmbeddingService
from ..services.query_classifier import QueryClassifier, QueryType

logger = logging.getLogger(__name__)


TOOLS = [
    {
        "name": "search_documents",
        "description": "Search the document database for relevant information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "A simple calculator that performs basic arithmetic operations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4')",
                }
            },
            "required": ["expression"],
        },
    },
]

# Tool names for easy reference
TOOL_NAMES = {"SEARCH": "search_documents", "CALCULATOR": "calculator"}


class ToolExecutor:
    """Handles the execution of different tools."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.calculator = Calculator()
        self.query_classifier = QueryClassifier()

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute a tool with the given input.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            The result of the tool execution as a string

        Raises:
            ValueError: If the tool name is unknown
        """
        if tool_name == TOOL_NAMES["SEARCH"]:
            return self._execute_search(tool_input["query"])
        elif tool_name == TOOL_NAMES["CALCULATOR"]:
            return self._execute_calculator(tool_input["expression"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _execute_search(self, query: str) -> str:
        """
        Execute the search documents tool with enhanced result formatting.

        Args:
            query: The search query string

        Returns:
            Formatted string containing search results with relevance scores
        """
        logger.info(f"Executing search with query: {query}")

        try:
            # Generate embedding for the query
            query_embedding = self.embedding_service.get_single_embedding(query)

            # Classify query and get appropriate threshold
            query_type, confidence = self.query_classifier.classify(query)
            threshold = self.query_classifier.get_recommended_threshold(query_type)
            logger.info(
                f"Query classified as {query_type.value} (confidence: {confidence:.2f})"
            )
            logger.info(f"Using similarity threshold: {threshold}")

            # Search for documents with query-specific threshold
            results = db.query_documents(
                query_embedding=query_embedding, similarity_threshold=threshold
            )

            # Check if we have any results
            if not results["documents"] or not results["documents"][0]:
                logger.info("No documents found matching the query")
                return "No relevant documents found."

            # Format results as a string with similarity scores
            formatted_results = []
            for doc, metadata, distance, doc_id in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],  # Get IDs directly from query results
            ):
                # Convert distance to similarity score
                # ChromaDB uses cosine distance where:
                # - 0 means identical (cos(0°) = 1)
                # - 2 means opposite (cos(180°) = -1)
                # Convert to a 0-1 similarity score where 1 is most similar
                similarity = (2 - distance) / 2

                # Determine relevance category based on similarity threshold
                if similarity >= settings.high_similarity_threshold:
                    relevance = "High"
                elif similarity >= settings.moderate_similarity_threshold:
                    relevance = "Moderate"
                else:
                    relevance = "Low"

                # Format must match the parsing in chat.py
                formatted_results.append(
                    f"Document (ID: {doc_id}) "
                    f"Relevance: {relevance} (Score: {similarity:.3f}) "
                    f"{doc}"
                )

            logger.info(
                f"Found {len(formatted_results)} documents "
                f"(Threshold: {settings.similarity_threshold})"
            )
            return "\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return f"Error searching documents: {str(e)}"

    def _execute_calculator(self, expression: str) -> str:
        """Execute the calculator tool."""
        logger.info(f"Executing calculator with expression: {expression}")
        try:
            result = self.calculator.evaluate(expression)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating result: {str(e)}"
