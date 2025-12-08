"""
Custom Voyage AI embeddings wrapper without LangChain.
Direct API integration using voyageai package.
"""
import os
from typing import List, Optional
from loguru import logger
import voyageai


class VoyageEmbeddings:
    """
    Custom Voyage AI embeddings class to replace LangChain's VoyageAIEmbeddings.
    Provides direct API access with token tracking.
    """
    
    def __init__(
        self,
        model: str = "voyage-3-large",
        api_key: Optional[str] = None,
        batch_size: int = 128,
        truncation: bool = True,
        show_progress_bar: bool = False
    ):
        self.model = model
        self.batch_size = batch_size
        self.truncation = truncation
        self.show_progress_bar = show_progress_bar
        
        # Initialize Voyage AI client
        api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is required")
        
        self.client = voyageai.Client(api_key=api_key)
        
        # Token tracking
        self._total_tokens = 0
        self._query_tokens = 0
        self._document_tokens = 0
    
    def get_total_tokens(self) -> int:
        """Get the total number of tokens processed."""
        return self._query_tokens + self._document_tokens
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings: List[List[float]] = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                result = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type="document",
                    truncation=self.truncation
                )
                
                embeddings.extend(result.embeddings)
                self._document_tokens += result.total_tokens
                
                if self.show_progress_bar:
                    logger.info(f"Embedded batch {i//self.batch_size + 1}, tokens: {result.total_tokens}")
                    
            except Exception as e:
                logger.error(f"Error embedding batch {i//self.batch_size + 1}: {e}")
                raise
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            result = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="query",
                truncation=self.truncation
            )
            
            self._query_tokens += result.total_tokens
            return result.embeddings[0]
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise


class VoyageReranker:
    """
    Custom Voyage AI reranker to replace LangChain's VoyageAIRerank.
    """
    
    def __init__(
        self,
        model: str = "rerank-2",
        top_k: int = 12,
        api_key: Optional[str] = None,
        truncation: bool = True
    ):
        self.model = model
        self.top_k = top_k
        self.truncation = truncation
        
        # Initialize Voyage AI client
        api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is required")
        
        self.client = voyageai.Client(api_key=api_key)
        self.total_tokens = 0
    
    def rerank(self, query: str, documents: List[str]) -> 'RerankResult':
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Query text
            documents: List of document texts to rerank
            
        Returns:
            Rerank result with sorted documents and scores
        """
        try:
            result = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_k=self.top_k,
                truncation=self.truncation
            )
            
            self.total_tokens = result.total_tokens
            return result
            
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            raise
    
    def get_total_tokens(self) -> int:
        """
        Get total tokens used for reranking.
        
        Returns:
            Total token count
        """
        return self.total_tokens


class RerankResult:
    """Simple container for rerank results."""
    
    def __init__(self, results: List, total_tokens: int):
        self.results = results
        self.total_tokens = total_tokens
