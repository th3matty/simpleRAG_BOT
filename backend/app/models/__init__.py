# Import your actual models here
__all__ = []

from enum import Enum
from typing import Optional
from pydantic import BaseModel
from fastapi import UploadFile

from routes.chat import DocumentMetadata


class DocumentType(Enum):
    PDF = "pdf"
    DOCX = "docx"
    CSV = "csv"
    TEXT = "text"  # for direct text input


class FileUploadRequest(BaseModel):
    metadata: Optional[DocumentMetadata] = None
