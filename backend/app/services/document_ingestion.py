# app/services/document_ingestion.py

import datetime
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import HTTPException

from .file_handler import FileHandler
from .chunker import DocumentProcessor
from .document_processor.factory import DocumentProcessorFactory
from ..core.database import db
from ..core.exceptions import RAGException

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.document_processor = DocumentProcessor(embedding_service=embedding_service)
        self.file_handler = FileHandler()

    def _validate_file(self, filename: str) -> str:
        """Validate file and return file extension."""
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        file_ext = Path(filename).suffix.lower()
        logger.info(f"Extracted file extension: '{file_ext}'")

        if not file_ext:
            raise HTTPException(
                status_code=400,
                detail="File must have a valid extension (.pdf, .docx, or .csv)",
            )
        return file_ext

    def _prepare_metadata(
        self,
        filename: str,
        file_ext: str,
        title: Optional[str],
        source: str,
        tags: Optional[str],
        doc_metadata: Dict,
    ) -> Dict[str, Any]:
        """Prepare metadata dictionary."""
        metadata = {
            "source": source,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "original_filename": filename,
            "file_type": file_ext.replace(".", ""),
        }

        if title:
            metadata["title"] = title
        if tags:
            metadata["tags"] = ",".join(tags.split(","))

        # Update with extracted metadata from file
        metadata.update(doc_metadata)
        return metadata

    async def process_file(self, file, title=None, source="file-upload", tags=None):
        """Process a file and return processed chunks."""
        try:
            file_ext = self._validate_file(file.filename)

            # Get appropriate processor
            processor = DocumentProcessorFactory.get_processor(file_ext)
            logger.info(f"Successfully got processor for {file_ext}")

            # Save file temporarily
            temp_file_path = await self.file_handler.save_upload_file_temporarily(file)
            logger.info(f"File saved temporarily at: {temp_file_path}")

            try:
                # Extract content and metadata
                content = processor.extract_text(temp_file_path)
                doc_metadata = processor.extract_metadata(temp_file_path)

                # Prepare metadata
                metadata_dict = self._prepare_metadata(
                    file.filename, file_ext, title, source, tags, doc_metadata
                )

                # Process document
                processed_chunks = self.document_processor.process_document(
                    content=content, metadata=metadata_dict
                )

                if not processed_chunks:
                    raise RAGException("No chunks were successfully processed")

                return processed_chunks

            finally:
                # Cleanup temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

    def save_to_database(self, processed_chunks):
        """Save processed chunks to database."""
        try:
            processed_metadata = [
                {k: convert_metadata_value(v) for k, v in chunk.metadata.items()}
                for chunk in processed_chunks
            ]

            db.add_documents(
                documents=[chunk.content for chunk in processed_chunks],
                embeddings=[chunk.embedding for chunk in processed_chunks],
                metadata=processed_metadata,
                ids=[
                    f"{chunk.metadata['parent_id']}_{chunk.metadata['chunk_index']}"
                    for chunk in processed_chunks
                ],
            )

            return {
                "document_ids": [processed_chunks[0].metadata["parent_id"]],
                "metadata": {
                    "file_type": processed_chunks[0].metadata.get("file_type"),
                    "chunk_count": len(processed_chunks),
                    "timestamp": processed_chunks[0].metadata.get("timestamp"),
                    "original_filename": processed_chunks[0].metadata.get(
                        "original_filename"
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            raise

    async def process_and_save(self, file, title=None, source="file-upload", tags=None):
        """Complete pipeline for processing and saving a file."""
        processed_chunks = await self.process_file(file, title, source, tags)
        return self.save_to_database(processed_chunks)


def convert_metadata_value(value):
    """Convert metadata values to ChromaDB-compatible format."""
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
