"""
Custom Document class to replace LangChain's Document schema.
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Represents a document chunk with content and metadata.
    Replacement for langchain.schema.Document.
    """
    page_content: str = Field(..., description="The text content of the document chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the document")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __repr__(self) -> str:
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"
    
    def __str__(self) -> str:
        return self.page_content
