# app/services/file_handler.py

import tempfile
from fastapi import UploadFile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FileHandler:
    @staticmethod
    async def save_upload_file_temporarily(upload_file: UploadFile) -> str:
        """
        Save an uploaded file temporarily and return its path.

        Args:
            upload_file: The uploaded file from FastAPI

        Returns:
            str: Path to the temporary file

        Raises:
            Exception: If file saving fails
        """
        try:
            suffix = Path(upload_file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                contents = await upload_file.read()
                temp_file.write(contents)
                return temp_file.name
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise
