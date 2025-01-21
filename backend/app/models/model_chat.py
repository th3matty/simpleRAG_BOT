from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from .model_document import DocumentSource


class ChatRequest(BaseModel):
    query: str = Field(..., description="User query string", min_length=1)


class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated response from LLM")
    sources: List["DocumentSource"] = Field(
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
