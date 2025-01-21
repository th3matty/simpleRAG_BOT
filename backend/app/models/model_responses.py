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
