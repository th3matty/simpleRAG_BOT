# models/requests.py
from pydantic import BaseModel, Field
from typing import List, Optional
from .model_document import DocumentInput


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
