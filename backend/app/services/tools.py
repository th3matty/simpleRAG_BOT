"""
Tools configuration for the LLM service.
This module defines the available tools and their schemas.
"""

from typing import Dict, Any
from .calculator import Calculator
from ..database import db
from ..services.embeddings import EmbeddingService
from ..config import logger

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
        """Execute the search documents tool."""
        logger.info(f"Executing search with query: {query}")

        # Generate embedding for the query
        query_embedding = self.embedding_service.get_single_embedding(query)

        # Search for documents
        results = db.query_documents(
            query_embedding=query_embedding,
            n_results=3,  # You might want to make this configurable
        )

        if not results["documents"][0]:
            return "No relevant documents found."

        # Format results as a string
        formatted_results = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            formatted_results.append(
                f"Document (ID: {meta.get('id', 'unknown')}): {doc}"
            )

        return "\n\n".join(formatted_results)

    def _execute_calculator(self, expression: str) -> str:
        """Execute the calculator tool."""
        logger.info(f"Executing calculator with expression: {expression}")
        try:
            result = self.calculator.evaluate(expression)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating result: {str(e)}"
