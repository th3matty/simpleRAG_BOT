from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import datetime
import time
import logging

from app.services.chunker import DocumentProcessor

from ..services.llm import LLMService
from ..services.embeddings import EmbeddingService
from ..core.tools import ToolExecutor
from ..core.database import db
from ..core.config import settings
from ..core.exceptions import DatabaseError, EmbeddingError, RAGException

from collections import defaultdict
from itertools import groupby
from operator import itemgetter

logger = logging.getLogger(__name__)

router = APIRouter()


# Chat-related models
class ChatRequest(BaseModel):
    query: str = Field(..., description="User query string", min_length=1)


class Source(BaseModel):
    content: str = Field(..., description="Source document content")
    metadata: Dict[str, Any] = Field(..., description="Source document metadata")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated response from LLM")
    sources: List[Source] = Field(
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


class Document(BaseModel):
    content: str = Field(..., description="Document content")
    metadata: Optional[DocumentMetadata] = None


class ChunkInfo(BaseModel):
    """Represents a single chunk of a document with its metadata"""

    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Position of chunk in original document")
    metadata: Dict[str, Any] = Field(..., description="Chunk-specific metadata")


class DocumentWithChunks(BaseModel):
    """Represents a complete document with all its chunks"""

    document_id: str = Field(..., description="Unique identifier for the document")
    title: Optional[str] = Field(None, description="Document title")
    source: str = Field(..., description="Document source")
    chunks: List[ChunkInfo] = Field(..., description="List of document chunks")
    metadata: Dict[str, Any] = Field(..., description="Document-level metadata")


# Response models
class GetDocumentsResponse(BaseModel):
    """Response model for document retrieval"""

    count: int = Field(..., description="Total number of documents")
    documents: List[DocumentWithChunks] = Field(
        ..., description="List of documents with their chunks"
    )


class UploadResponse(BaseModel):
    message: str
    document_ids: List[str]
    metadata: Dict[str, Any] = Field(
        ..., description="Response metadata including token usage"
    )


# Request Model
class AddDocumentsRequest(BaseModel):
    documents: List[Document] = Field(
        ..., description="List of documents with their metadata"
    )


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
                            Source(
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


@router.get("/documents", response_model=GetDocumentsResponse)
async def get_documents():
    """
    Retrieve all documents from the database, organized by their original structure.
    This endpoint reassembles chunked documents into their complete form while
    maintaining chunk relationships and metadata.

    Returns:
        GetDocumentsResponse containing organized documents and their chunks
    """
    try:
        logger.info("Retrieving all documents from database")

        # Add debug logging
        logger.debug(f"Using ChromaDB directory: {settings.chroma_persist_directory}")

        # Get all documents from the database
        results = db.get_all_documents()
        # Add debug logging for results
        logger.debug(f"Raw database results: {results}")

        if not results or not results["documents"]:
            logger.info("No documents found in database")
            return GetDocumentsResponse(count=0, documents=[])

        # Group chunks by their parent document ID
        document_chunks = defaultdict(list)

        # Process each chunk and organize by parent document
        for doc, meta, doc_id in zip(
            results["documents"], results["metadatas"], results["ids"]
        ):
            # Extract parent_id from metadata
            parent_id = meta.get("parent_id")
            if not parent_id:
                logger.warning(f"Chunk {doc_id} has no parent_id, skipping")
                continue

            # Create chunk info
            chunk = ChunkInfo(
                content=doc,
                chunk_index=meta.get("chunk_index", 0),
                metadata={
                    k: v
                    for k, v in meta.items()
                    if k not in ["parent_id", "chunk_index"]
                },
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
            document = DocumentWithChunks(
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

        return GetDocumentsResponse(count=len(documents), documents=documents)

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(
    request: AddDocumentsRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """
    Upload and process documents using our semantic chunking system.
    Each document is split into meaningful chunks while preserving context and structure.

    Args:
        request: AddDocumentsRequest containing list of documents
        embedding_service: Injected embedding service

    Returns:
        UploadResponse containing success message and document IDs
    """
    try:
        logger.info(
            f"Processing upload request with {len(request.documents)} documents"
        )

        # Initialize our document processor
        document_processor = DocumentProcessor(embedding_service=embedding_service)

        all_chunks = []
        parent_doc_ids = []

        # Process each document
        for doc in request.documents:
            try:
                # Prepare metadata with defaults if none provided
                metadata = {
                    "source": "api-upload",
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                }

                if doc.metadata:
                    metadata.update(
                        {
                            "source": doc.metadata.source,
                            "tags": doc.metadata.tags,
                        }
                    )
                    if doc.metadata.title:
                        metadata["title"] = doc.metadata.title

                # Process document into chunks
                processed_chunks = document_processor.process_document(
                    content=doc.content, metadata=metadata
                )

                if processed_chunks:
                    all_chunks.extend(processed_chunks)
                    parent_doc_ids.append(processed_chunks[0].metadata["parent_id"])
                    logger.info(
                        f"Document processed into {len(processed_chunks)} chunks"
                    )

            except Exception as doc_error:
                logger.error(f"Error processing document: {str(doc_error)}")
                continue

        # Add all chunks to the database
        if all_chunks:
            try:
                db.add_documents(
                    documents=[chunk.content for chunk in all_chunks],
                    embeddings=[chunk.embedding for chunk in all_chunks],
                    metadata=[chunk.metadata for chunk in all_chunks],
                    ids=[
                        f"{chunk.metadata['parent_id']}_{chunk.metadata['chunk_index']}"
                        for chunk in all_chunks
                    ],
                )

                logger.info(
                    f"Successfully uploaded {len(all_chunks)} chunks from {len(parent_doc_ids)} documents"
                )

                return UploadResponse(
                    message=f"Successfully processed {len(request.documents)} documents into {len(all_chunks)} chunks",
                    document_ids=parent_doc_ids,
                    metadata={
                        "document_count": len(request.documents),
                        "chunk_count": len(all_chunks),
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                    },
                )

            except Exception as db_error:
                logger.error(f"Database error: {str(db_error)}")
                raise DatabaseError(f"Failed to store documents: {str(db_error)}")
        else:
            raise RAGException("No chunks were successfully processed")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
