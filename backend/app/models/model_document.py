# models/document.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class DocumentType(Enum):
    PDF = "pdf"
    DOCX = "docx"
    CSV = "csv"
    TEXT = "text"


class DocumentMetadata(BaseModel):
    title: Optional[str] = Field(None, description="Document title")
    source: str = Field(default="api-upload", description="Source of the document")
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorizing the document"
    )


class DocumentSource(BaseModel):
    content: str = Field(..., description="Source document content")
    metadata: Dict[str, Any] = Field(..., description="Source document metadata")


class DocumentChunk(BaseModel):
    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Position of chunk in original document")
    metadata: Dict[str, Any] = Field(..., description="Chunk-specific metadata")
    embedding: Optional[List[float]] = Field(
        None, description="Vector embedding of the chunk"
    )


class DocumentComplete(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    title: Optional[str] = Field(None, description="Document title")
    source: str = Field(..., description="Document source")
    chunks: List[DocumentChunk] = Field(..., description="List of document chunks")
    metadata: Dict[str, Any] = Field(..., description="Document-level metadata")
