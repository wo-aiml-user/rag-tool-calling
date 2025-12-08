"""
Retrieval tool implementation for RAG pipeline.
Defines retrieval as a tool for DeepSeek function calling.
"""
from typing import List, Dict, Any, Optional
from pymilvus import Collection, connections, utility
from app.RAG.voyage_embeddings import VoyageEmbeddings, VoyageReranker
from app.RAG.document import Document
from app.config import settings
from loguru import logger


class RetrievalTool:
    """
    Retrieval tool for querying documents from Milvus.
    Can be used as a function calling tool with DeepSeek.
    """
    
    def __init__(
        self,
        collection_name: str,
        embedding: VoyageEmbeddings,
        reranker: Optional[VoyageReranker] = None,
        top_k: int = 10,
        rerank_top_k: int = 6
    ):
        """
        Initialize retrieval tool.
        
        Args:
            collection_name: Milvus collection name
            embedding: Embedding model instance
            reranker: Optional reranker instance
            top_k: Number of documents to retrieve
            rerank_top_k: Number of documents after reranking
        """
        self.collection_name = collection_name
        self.embedding = embedding
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
    
    def retrieve(
        self,
        query: str,
        file_ids: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            file_ids: Optional list of file IDs to filter by
            
        Returns:
            List of relevant documents
        """
        try:
            # Connect to Milvus
            connections.connect(
                uri=settings.MILVUS_URI,
                db_name=settings.MILVUS_DB_NAME,
                token=settings.MILVUS_TOKEN
            )
            
            # Check if collection exists
            if not utility.has_collection(self.collection_name):
                logger.warning(f"Collection {self.collection_name} does not exist")
                return []
            
            # Load collection
            collection = Collection(name=self.collection_name)
            collection.load()
            
            # Generate query embedding
            query_vector = self.embedding.embed_query(query)
            
            # Build search expression for file filtering
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
            
            expr = None
            if file_ids:
                expr = " or ".join([f'file_id == "{fid}"' for fid in file_ids])
            
            # Search
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=self.top_k,
                expr=expr,
                output_fields=["text", "file_id", "file_name", "file_path", "page_number", "chunk_number"]
            )
            
            # Convert results to documents
            documents = []
            if results and len(results) > 0:
                for hit in results[0]:
                    doc = Document(
                        page_content=hit.entity.get("text", ""),
                        metadata={
                            "file_id": hit.entity.get("file_id", ""),
                            "file_name": hit.entity.get("file_name", ""),
                            "file_path": hit.entity.get("file_path", ""),
                            "page_number": hit.entity.get("page_number", ""),
                            "chunk_number": hit.entity.get("chunk_number", 0),
                            "score": hit.score
                        }
                    )
                    documents.append(doc)
            
            # Rerank if reranker is available
            if self.reranker and documents:
                doc_contents = [doc.page_content for doc in documents]
                rerank_result = self.reranker.rerank(query=query, documents=doc_contents)
                
                # Reorder documents based on rerank results
                reranked_docs = []
                for res in rerank_result.results:
                    doc = documents[res.index]
                    doc.metadata["relevance_score"] = res.relevance_score
                    doc.metadata["index"] = len(reranked_docs)
                    reranked_docs.append(doc)
                
                documents = reranked_docs[:self.rerank_top_k]
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def format_for_llm(self, documents: List[Document]) -> str:
        """
        Format retrieved documents for LLM consumption.
        
        Args:
            documents: List of documents
            
        Returns:
            Formatted string of documents
        """
        if not documents:
            return "No relevant documents found."
        
        formatted = []
        for i, doc in enumerate(documents):
            file_name = doc.metadata.get('file_name', 'unknown')
            page_number = doc.metadata.get('page_number', 'unknown')
            content = doc.page_content
            
            formatted.append(
                f"Context {i+1}:\n"
                f"  Document: {file_name}\n"
                f"  Reference: {page_number}\n"
                f"  Content: {content}\n"
            )
        
        return "\n".join(formatted)



