# models/responses.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from .model_document import DocumentComplete


class DocumentListResponse(BaseModel):
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


class DocumentDeleteResponse(BaseModel):
    message: str = Field(..., description="Status message about the deletion operation")
    deleted_count: int = Field(..., description="Number of documents deleted")
    source: str = Field(..., description="Source from which documents were deleted")
    metadata: Dict[str, Any] = Field(
        default={}, description="Optional metadata about the deletion operation"
    )
