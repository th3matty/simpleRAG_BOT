# app/services/document_processor/csv_processor.py

from .base import BaseDocumentProcessor
from typing import Dict, Any
import logging
import pandas as pd
import os
from io import StringIO

logger = logging.getLogger(__name__)


class CSVProcessor(BaseDocumentProcessor):
    """
    Processor for CSV files using pandas.
    """

    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from a CSV file.
        Converts CSV data into a readable text format.

        Args:
            file_path: Path to the CSV file

        Returns:
            CSV content in a readable text format

        Raises:
            Exception: If CSV processing fails
        """
        try:
            logger.info(f"Starting text extraction from CSV: {file_path}")

            # Read CSV file
            df = pd.read_csv(file_path)

            # Convert DataFrame to string representation
            buffer = StringIO()
            df.to_string(buffer, index=False)
            text_content = buffer.getvalue()

            logger.info(f"Successfully extracted CSV content with {len(df)} rows")
            return text_content

        except Exception as e:
            logger.error(f"Error extracting text from CSV: {str(e)}")
            raise

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary containing CSV metadata

        Raises:
            Exception: If metadata extraction fails
        """
        try:
            logger.info(f"Extracting metadata from CSV: {file_path}")

            df = pd.read_csv(file_path)

            metadata = {
                "file_type": "csv",
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": df.columns.tolist(),
                "file_size": os.path.getsize(file_path),
                "empty_cells": df.isna().sum().sum(),
                "total_cells": df.size,
            }

            # Add some basic statistics if numerical columns exist
            numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
            if not numeric_columns.empty:
                metadata["numeric_columns"] = numeric_columns.tolist()
                metadata["has_numeric_data"] = True
            else:
                metadata["has_numeric_data"] = False

            logger.info("Successfully extracted CSV metadata")
            logger.debug(f"Extracted metadata: {metadata}")

            return metadata

        except Exception as e:
            logger.error(f"Error extracting CSV metadata: {str(e)}")
            raise
