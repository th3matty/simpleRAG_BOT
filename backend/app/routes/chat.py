from fastapi import APIRouter, Form, HTTPException, Depends, UploadFile, File
from typing import Optional
import datetime
import logging

from ..services.document_ingestion import DocumentIngestionService


from ..services.llm import LLMService
from ..services.embeddings import EmbeddingService
from ..core.tools import ToolExecutor
from ..core.database import db
from ..core.config import settings
from ..core.exceptions import RAGException
from ..models import (
    ChatRequest,
    DocumentSource,
    ChatResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    DocumentChunk,
    DocumentComplete,
    DocumentDeleteResponse,
)

from collections import defaultdict

logger = logging.getLogger(__name__)

router = APIRouter()


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


@router.get("/documents/source/{source}", response_model=DocumentListResponse)
async def get_documents_by_source(source: str):
    """
    Retrieve all documents from a specific source in the database.
    This endpoint reassembles chunked documents into their complete form while
    maintaining chunk relationships and metadata.

    Args:
        source: Source/filename to filter documents by (e.g., 'article1.md')

    Returns:
        DocumentListResponse containing organized documents and their chunks
    """
    try:
        logger.info(f"Retrieving documents for source: {source}")
        logger.debug(f"Using ChromaDB directory: {settings.chroma_persist_directory}")

        # Get documents from the default collection with source filter
        try:
            results = db.collection.get(
                include=["documents", "metadatas", "embeddings"],
                where={"source": source},
            )
            logger.debug(f"Raw database results: {results}")
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error retrieving documents: {str(e)}"
            )

        if not results or not results["documents"]:
            logger.info(f"No documents found in collection: {source}")
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

        logger.info(f"Retrieved {len(documents)} documents for source: {source}")

        return DocumentListResponse(count=len(documents), documents=documents)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error retrieving documents from collection: {str(e)}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


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
    """Upload and process a document file."""
    try:
        ingestion_service = DocumentIngestionService(embedding_service)
        result = await ingestion_service.process_and_save(file, title, source, tags)

        return DocumentUploadResponse(
            message=f"Successfully processed file: {file.filename}",
            document_ids=result["document_ids"],
            metadata=result["metadata"],
        )

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/documents/update/file", response_model=DocumentUploadResponse)
async def update_file_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    source: Optional[str] = Form("file-upload"),
    tags: Optional[str] = Form(None),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """
    Update an existing document file or create if it doesn't exist.
    Deletes the old version if it exists and processes the new version.

    Args:
        file: File to update
        title: Optional title for the document
        source: Source of the document (default: file-upload)
        tags: Optional tags for categorization
        embedding_service: Injected embedding service

    Returns:
        DocumentUploadResponse with update status and metadata
    """
    try:
        # First find if document exists
        old_results = db.collection.get(where={"original_filename": file.filename})

        # Delete old version if exists
        if old_results["ids"]:
            logger.info(
                f"Found existing document with {len(old_results['ids'])} chunks"
            )
            db.collection.delete(ids=old_results["ids"])
            logger.info("Deleted old version")

        # Use ingestion service to process and save new version
        ingestion_service = DocumentIngestionService(embedding_service)
        result = await ingestion_service.process_and_save(file, title, source, tags)

        # Prepare enhanced metadata with update information
        enhanced_metadata = {
            **result["metadata"],
            "updated": True,
            "previous_version": {
                "chunk_count": len(old_results["ids"]) if old_results["ids"] else 0,
                "timestamp": (
                    old_results["metadatas"][0].get("timestamp")
                    if old_results["ids"]
                    else None
                ),
            },
            "changes": {
                "update_timestamp": datetime.datetime.utcnow().isoformat(),
            },
        }

        # Prepare response message
        message = (
            (
                f"Successfully updated file: {file.filename}. "
                f"Previous version had {len(old_results['ids'])} chunks, "
                f"new version has {len(result['document_ids'])} chunks."
            )
            if old_results["ids"]
            else f"Created new document: {file.filename}"
        )

        return DocumentUploadResponse(
            message=message,
            document_ids=result["document_ids"],
            metadata=enhanced_metadata,
        )

    except Exception as e:
        logger.error(f"Error updating document: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500, detail=f"Failed to update document: {str(e)}"
        )


@router.delete("/documents/source/{source}", response_model=DocumentDeleteResponse)
async def delete_documents_by_source(source: str):
    """
    Delete all documents from a specific source in the database.

    Args:
        source: Source/filename to delete documents from (e.g., 'article1.md')

    Returns:
        DocumentDeleteResponse containing deletion details
    """
    try:
        logger.info(f"Deleting documents for source: {source}")

        # Get all documents with the specified source
        results = db.collection.get(include=["metadatas"], where={"source": source})

        if not results or not results["ids"]:
            logger.info(f"No documents found for source: {source}")
            return DocumentDeleteResponse(
                message=f"No documents found for source: {source}",
                deleted_count=0,
                source=source,
            )

        # Delete all documents with the specified source
        db.collection.delete(ids=results["ids"])

        deletion_count = len(results["ids"])
        logger.info(
            f"Successfully deleted {deletion_count} documents from source: {source}"
        )

        return DocumentDeleteResponse(
            message=f"Successfully deleted {deletion_count} documents from source: {source}",
            deleted_count=deletion_count,
            source=source,
            metadata={
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "deleted_ids": results["ids"],
            },
        )

    except Exception as e:
        logger.error(f"Error deleting documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete documents from source {source}: {str(e)}",
        )


@router.delete("/documents/collection", response_model=DocumentDeleteResponse)
async def delete_collection():
    """
    Delete all documents from the collection.
    This is a destructive operation and cannot be undone.

    Returns:
        DocumentDeleteResponse containing deletion details
    """
    try:
        logger.info("Deleting all documents from collection")

        # Get all document IDs first to include in response
        results = db.collection.get(include=["metadatas"])

        if not results or not results["ids"]:
            logger.info("No documents found in collection")
            return DocumentDeleteResponse(
                message="No documents found in collection",
                deleted_count=0,
                source="all",
            )

        # Delete all documents by matching any source
        db.collection.delete(where={"source": {"$ne": ""}})

        deletion_count = len(results["ids"])
        logger.info(f"Successfully deleted {deletion_count} documents from collection")

        return DocumentDeleteResponse(
            message=f"Successfully deleted {deletion_count} documents from collection",
            deleted_count=deletion_count,
            source="all",
            metadata={
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "deleted_ids": results["ids"],
            },
        )

    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection: {str(e)}",
        )
