import datetime
import time
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from ..services.llm import LLMService
from ..services.embeddings import EmbeddingService
from ..services.tools import ToolExecutor
from ..database import db
from ..config import settings, logger
from ..exceptions import DatabaseError, EmbeddingError, RAGException

router = APIRouter()


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


class DebugDocument(BaseModel):
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")


class DebugResponse(BaseModel):
    count: int = Field(..., description="Total number of documents")
    documents: List[DebugDocument] = Field(..., description="List of all documents")


class AddDocumentsRequest(BaseModel):
    documents: List[str]


class UploadResponse(BaseModel):
    message: str
    document_ids: List[str]
    metadata: Dict[str, Any] = Field(
        ..., description="Response metadata including token usage"
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

        # Prepare response
        sources = []
        if response.get("tool_used") == "search_documents":
            # If search was used, extract sources from the tool result
            results = response["tool_result"].split("\n\n")
            for result in results:
                if result.startswith("Document (ID: "):
                    # Parse document content and metadata from the formatted result
                    doc_id = result[result.find("ID: ") + 4 : result.find(")")]
                    content = result[result.find("): ") + 3 :]
                    sources.append(Source(content=content, metadata={"id": doc_id}))

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


@router.get("/debug/documents", response_model=DebugResponse)
async def get_documents():
    """
    Debug endpoint to inspect all documents in the database.

    Returns:
        DebugResponse containing all documents and their count

    Raises:
        HTTPException: If database query fails
    """
    try:
        results = db.get_all_documents()

        documents = [
            DebugDocument(content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

        return DebugResponse(count=len(documents), documents=documents)

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(
    request: AddDocumentsRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """
    Upload documents to the RAG system.

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
        document_ids = []
        metadata = []

        # Generate embeddings for all documents at once
        embeddings = embedding_service.get_embeddings(request.documents)

        # Create metadata and IDs for each document
        for i in range(len(request.documents)):
            doc_id = f"doc_{i}_{int(time.time())}"
            document_ids.append(doc_id)

            metadata_item = {
                "id": doc_id,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "source": "api-upload",
            }
            metadata.append(metadata_item)

        # Add documents to the database with their IDs
        db.add_documents(
            documents=request.documents,
            embeddings=embeddings,
            metadata=metadata,
            ids=document_ids,
        )

        logger.info(f"Successfully uploaded {len(request.documents)} documents")

        return UploadResponse(
            message=f"Successfully uploaded {len(request.documents)} documents",
            document_ids=document_ids,
            metadata={
                "document_count": len(request.documents),
                "timestamp": datetime.datetime.utcnow().isoformat(),
            },
        )

    except EmbeddingError as e:
        logger.error(f"Embedding error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
