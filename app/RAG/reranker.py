from app.RAG.document import Document
from app.RAG.voyage_embeddings import VoyageReranker
from typing import List, Any


def format_page_number(page_number):
    """
    Format the page_number value as follows:
    
    - If page_number is an integer, return "Page <number>".
    - If page_number is a string that contains "Sheet":
         Remove a leading "Page:" (if present), split by " - ", and return the parts on separate lines.
    - Otherwise, if page_number is a string:
         If it consists solely of digits, return "Page <number>".
         Otherwise, if it does not already start with "Page", prefix it with "Page ".
    - In all other cases, return page_number unchanged.
    """
    # Handle integer input.
    if isinstance(page_number, int):
        return f"Page {page_number}"
    
    # Handle string input.
    if isinstance(page_number, str):
        # If "Sheet" is in the string, apply special formatting.
        if "Sheet" in page_number:
            if page_number.startswith("Page:"):
                page_number = page_number[len("Page:"):]
            parts = page_number.split(" - ")
            if len(parts) == 2:
                return f"\n\t\t{parts[0].strip()}\n\t\t{parts[1].strip()}"
        else:
            # If the string is composed of digits, prefix with "Page "
            if page_number.isdigit():
                return f"Page {page_number}"
            # Otherwise, if it doesn't start with "Page", attach "Page " at the beginning.
            if not page_number.startswith("Page"):
                return f"Page {page_number}"
    return page_number


class CustomContextualCompressionRetriever:
    """
    Custom compression retriever to replace LangChain's ContextualCompressionRetriever.
    Uses Voyage AI reranker directly.
    """
    
    def __init__(self, base_compressor: VoyageReranker, base_retriever: Any):
        self.base_compressor = base_compressor
        self.base_retriever = base_retriever
        self.total_tokens = 0
    
    def invoke(self, query: str, **kwargs) -> List[Document]:
        """
        Retrieve and rerank documents.
        
        Args:
            query: Query string
            **kwargs: Additional arguments
            
        Returns:
            List of reranked documents
        """
        # Get documents from base retriever
        docs = self.base_retriever.invoke(query, **kwargs)
        
        if not docs:
            return []
        
        # Extract document contents
        doc_contents = [doc.page_content if isinstance(doc, Document) else doc for doc in docs]
        
        # Rerank using Voyage AI
        rerank_result = self.base_compressor.rerank(query=query, documents=doc_contents)
        self.total_tokens = rerank_result.total_tokens
        
        # Create compressed docs from rerank result
        compressed = []
        for res in rerank_result.results:
            doc = docs[res.index]
            # Create a copy of the document with updated metadata
            doc_copy = Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy() if hasattr(doc, 'metadata') else {}
            )
            
            # Format page number
            if "page_number" in doc_copy.metadata:
                doc_copy.metadata["page_number"] = format_page_number(doc_copy.metadata["page_number"])
            
            # Add relevance score and index
            doc_copy.metadata["relevance_score"] = res.relevance_score
            doc_copy.metadata["index"] = len(compressed)
            compressed.append(doc_copy)
        
        return compressed
