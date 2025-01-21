import json
import os
from pathlib import Path
from fastapi import APIRouter, Form, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import datetime
import logging

from ..services.chunker import DocumentProcessor
from ..services.document_processor.factory import DocumentProcessorFactory
from ..services.file_handler import FileHandler

from ..services.llm import LLMService
from ..services.embeddings import EmbeddingService
from ..core.tools import ToolExecutor
from ..core.database import db
from ..core.config import settings
from ..core.exceptions import DatabaseError, RAGException

from collections import defaultdict

logger = logging.getLogger(__name__)

router = APIRouter()


# Chat-related models
class ChatRequest(BaseModel):
    query: str = Field(..., description="User query string", min_length=1)


class DocumentSource(BaseModel):
    content: str = Field(..., description="Source document content")
    metadata: Dict[str, Any] = Field(..., description="Source document metadata")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated response from LLM")
    sources: List[DocumentSource] = Field(
        default=[], description="Source documents used for response"
    )
    metadata: Dict[str, Any] = Field(
        ..., description="Response metadata including token usage"
    )
    tool_used: Optional[str] = Field(None, description="Name of the tool that was used")
    tool_input: Optional[Dict[str, Any]] = Field(
        None, description="Input provided to the tool"
    )
    tool_result: Optional[str] = Field(None, description="Result returned by the tool")


# Document-related models


class DocumentMetadata(BaseModel):
    title: Optional[str] = Field(None, description="Document title")
    source: str = Field(default="api-upload", description="Source of the document")
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorizing the document"
    )


class DocumentInput(BaseModel):
    content: str = Field(..., description="Document content")
    metadata: Optional[DocumentMetadata] = None


class DocumentChunk(BaseModel):
    """Represents a single chunk of a document with its metadata"""

    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Position of chunk in original document")
    metadata: Dict[str, Any] = Field(..., description="Chunk-specific metadata")
    embedding: Optional[List[float]] = Field(
        None, description="Vector embedding of the chunk"
    )


class DocumentComplete(BaseModel):
    """Represents a complete document with all its chunks"""

    document_id: str = Field(..., description="Unique identifier for the document")
    title: Optional[str] = Field(None, description="Document title")
    source: str = Field(..., description="Document source")
    chunks: List[DocumentChunk] = Field(..., description="List of document chunks")
    metadata: Dict[str, Any] = Field(..., description="Document-level metadata")


# Response models
class DocumentListResponse(BaseModel):
    """Response model for document retrieval"""

    count: int = Field(..., description="Total number of documents")
    documents: List[DocumentComplete] = Field(
        ..., description="List of documents with their chunks"
    )


class DocumentUploadResponse(BaseModel):
    message: str
    document_ids: List[str]
    metadata: Dict[str, Any] = Field(
        ..., description="Response metadata including token usage"
    )


# Request Model
class DocumentUploadRequest(BaseModel):
    documents: List[DocumentInput] = Field(
        ..., description="List of documents with their metadata"
    )


class FileUploadMetadata(BaseModel):
    title: Optional[str] = None
    source: str = Field(default="api-upload")
    tags: List[str] = Field(default_factory=list)


class FileUploadRequest(BaseModel):
    metadata: Optional[FileUploadMetadata] = None


def get_llm_service() -> LLMService:
    """Dependency for LLM service."""
    return LLMService(settings.anthropic_api_key)


def get_embedding_service() -> EmbeddingService:
    """Dependency for embedding service."""
    return EmbeddingService(settings.embedding_model)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """
    Process a chat request using tools like search and calculator.

    Args:
        request: ChatRequest containing the user query
        llm_service: Injected LLM service
        embedding_service: Injected embedding service

    Returns:
        ChatResponse containing the generated response and any tool usage details

    Raises:
        HTTPException: If any step of the process fails
    """
    try:
        logger.info(f"Processing chat request: {request.query}")

        # Initialize tool executor
        tool_executor = ToolExecutor(embedding_service)

        # Process query with tool support
        response = llm_service.process_query(
            query=request.query, tool_executor=tool_executor
        )
        logger.info(f"Response generated: {response}")
        logger.info(
            f"Tool used: {response['tool_used'] if 'tool_used' in response else None}"
        )

        # Prepare response
        sources = []
        if response["tool_used"] == "search_documents":
            # If search was used, extract sources from the tool result
            results = response["tool_result"].split("\n\n")
            logger.info(f"results generated: {results}")

            for result in results:
                if result.startswith("Document (ID: "):
                    logger.info(f"result with documents id: {result}")
                    try:
                        # Extract document ID
                        doc_id = result[result.find("ID: ") + 4 : result.find(")")]

                        # Extract relevance info
                        relevance_start = result.find("Relevance: ") + 10
                        relevance_end = result.find(" (Score:")
                        relevance_category = result[relevance_start:relevance_end]

                        # Extract score
                        score_start = result.find("Score: ") + 7
                        score_end = result.find(")", score_start)
                        relevance_score = float(result[score_start:score_end])

                        # Extract content (everything after the score parenthesis)
                        content = result[result.find(")", score_end) + 2 :].strip()

                        logger.debug(
                            f"Parsed document - ID: {doc_id}, "
                            f"Relevance: {relevance_category}, "
                            f"Score: {relevance_score}"
                        )

                        # Create source with enhanced metadata
                        sources.append(
                            DocumentSource(
                                content=content,
                                metadata={
                                    "id": doc_id,
                                    "relevance_category": relevance_category,
                                    "relevance_score": relevance_score,
                                },
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error parsing document result: {str(e)}")
                        continue

        return ChatResponse(
            response=response["text"],
            sources=sources,
            metadata={
                "model": response["model"],
                "finish_reason": response["finish_reason"],
                "usage": response["usage"],
            },
            tool_used=response.get("tool_used"),
            tool_input=response.get("tool_input"),
            tool_result=response.get("tool_result"),
        )

    except RAGException as e:
        logger.error(f"RAG error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents():
    """
    Retrieve all documents from the database, organized by their original structure.
    This endpoint reassembles chunked documents into their complete form while
    maintaining chunk relationships and metadata.

    Returns:
        DocumentListResponse containing organized documents and their chunks
    """
    try:
        logger.info("Retrieving all documents from database")
        logger.debug(f"Using ChromaDB directory: {settings.chroma_persist_directory}")

        # Get all documents from the database
        results = db.collection.get(include=["documents", "metadatas", "embeddings"])
        logger.debug(f"Raw database results: {results}")

        if not results or not results["documents"]:
            logger.info("No documents found in database")
            return DocumentListResponse(count=0, documents=[])

        # Group chunks by their parent document ID
        document_chunks = defaultdict(list)

        # Process each chunk and organize by parent document
        for doc, meta, embedding, doc_id in zip(
            results["documents"],
            results["metadatas"],
            results["embeddings"],
            results["ids"],
        ):
            # Extract parent_id from metadata
            parent_id = meta.get("parent_id")
            if not parent_id:
                logger.warning(f"Chunk {doc_id} has no parent_id, skipping")
                continue

            # Create chunk info
            chunk = DocumentChunk(
                content=doc,
                chunk_index=meta.get("chunk_index", 0),
                metadata={
                    k: v
                    for k, v in meta.items()
                    if k not in ["parent_id", "chunk_index"]
                },
                embedding=embedding.tolist() if embedding is not None else None,
            )

            document_chunks[parent_id].append(chunk)

        # Organize chunks into complete documents
        documents = []
        for parent_id, chunks in document_chunks.items():
            # Sort chunks by their index
            sorted_chunks = sorted(chunks, key=lambda x: x.chunk_index)

            # Get document-level metadata from first chunk
            first_chunk = sorted_chunks[0]
            doc_metadata = first_chunk.metadata.copy()

            # Create document with its chunks
            document = DocumentComplete(
                document_id=parent_id,
                title=doc_metadata.get("title"),
                source=doc_metadata.get("source", "unknown"),
                chunks=sorted_chunks,
                metadata={
                    "timestamp": doc_metadata.get("timestamp"),
                    "tags": doc_metadata.get("tags", []),
                    "total_chunks": len(sorted_chunks),
                    "file_type": doc_metadata.get("file_type"),
                },
            )
            documents.append(document)

        logger.info(f"Retrieved {len(documents)} documents with their chunks")

        return DocumentListResponse(count=len(documents), documents=documents)

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@router.post("/documents/upload/file", response_model=DocumentUploadResponse)
async def upload_file_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    source: Optional[str] = Form("file-upload"),
    tags: Optional[str] = Form(None),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    try:
        logger.info("Starting file upload process")
        logger.info(f"Received file: {file.filename}")

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        file_ext = Path(file.filename).suffix.lower()
        logger.info(f"Extracted file extension: '{file_ext}'")

        if not file_ext:
            logger.error("No file extension found")
            raise HTTPException(
                status_code=400,
                detail="File must have a valid extension (.pdf, .docx, or .csv)",
            )

        # Initialize processors
        logger.info("Initializing processors")
        document_processor = DocumentProcessor(embedding_service=embedding_service)
        file_handler = FileHandler()

        # Get the appropriate processor
        try:
            logger.info(f"Getting processor for extension: {file_ext}")
            processor = DocumentProcessorFactory.get_processor(file_ext)
            logger.info(f"Successfully got processor for {file_ext}")
        except ValueError as e:
            logger.error(f"Failed to get processor: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported formats: ['.pdf', '.docx', '.csv']",
            )

        # Save file temporarily
        logger.info("Saving file temporarily")
        temp_file_path = await file_handler.save_upload_file_temporarily(file)
        logger.info(f"File saved temporarily at: {temp_file_path}")

        try:
            # Extract content and metadata
            content = processor.extract_text(temp_file_path)
            doc_metadata = processor.extract_metadata(temp_file_path)

            # Prepare metadata combining all sources
            metadata_dict = {
                "source": source,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "original_filename": file.filename,
                "file_type": file_ext.replace(".", ""),
            }

            # Add optional metadata
            if title:
                metadata_dict["title"] = title
            if tags:
                metadata_dict["tags"] = ",".join(tags.split(","))

            # Update with extracted metadata from file
            metadata_dict.update(doc_metadata)

            # Process document
            processed_chunks = document_processor.process_document(
                content=content, metadata=metadata_dict
            )

            if not processed_chunks:
                raise RAGException("No chunks were successfully processed")

            processed_metadata = [
                {k: convert_metadata_value(v) for k, v in chunk.metadata.items()}
                for chunk in processed_chunks
            ]

            # Add to database
            db.add_documents(
                documents=[chunk.content for chunk in processed_chunks],
                embeddings=[chunk.embedding for chunk in processed_chunks],
                metadata=processed_metadata,
                ids=[
                    f"{chunk.metadata['parent_id']}_{chunk.metadata['chunk_index']}"
                    for chunk in processed_chunks
                ],
            )

            return DocumentUploadResponse(
                message=f"Successfully processed file: {file.filename}",
                document_ids=[processed_chunks[0].metadata["parent_id"]],
                metadata={
                    "file_type": metadata_dict["file_type"],
                    "chunk_count": len(processed_chunks),
                    "timestamp": metadata_dict["timestamp"],
                    "original_filename": file.filename,
                },
            )

        finally:
            # Cleanup temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


def convert_metadata_value(value):
    """Convert metadata values to ChromaDB-compatible format."""
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
