from fastapi import HTTPException, status
from loguru import logger
from app.config import settings
from app.RAG.voyage_embeddings import VoyageEmbeddings
from app.RAG.document import Document
from typing import List, Dict, Any
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility


def store_documents_in_milvus(
    documents: List[Document],
    collection_name: str,
    file_id: str
) -> Dict[str, Any]:
    """
    Store documents in Milvus collection.
    
    Args:
        documents: List of Document objects to store
        collection_name: Name of the Milvus collection (use "tool_calling_dev")
        file_id: File identifier
        
    Returns:
        Dictionary with storage results
    """
    logger.info(f"[MILVUS] Storing {len(documents)} documents in collection: {collection_name}")
    
    if not documents:
        logger.warning("[MILVUS] No documents to store")
        return {"status": "no_documents", "count": 0}
    
    try:
        # Connect to Milvus
        connections.connect(
            uri=settings.MILVUS_URI,
            db_name=settings.MILVUS_DB_NAME,
            token=settings.MILVUS_TOKEN
        )
        logger.info(f"[MILVUS] Connected to Milvus")
        
        # Initialize embedding model
        embedding_model = VoyageEmbeddings(
            model="voyage-3-large",
            api_key=settings.VOYAGE_API_KEY,
            batch_size=128,
            truncation=True
        )
        logger.info("[MILVUS] Initialized Voyage embeddings")
        
        # Create collection if it doesn't exist
        if not utility.has_collection(collection_name):
            logger.info(f"[MILVUS] Creating new collection: {collection_name}")
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="page_number", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="chunk_number", dtype=DataType.INT64),
            ]
            
            schema = CollectionSchema(fields=fields, description=f"Collection for {collection_name}")
            collection = Collection(name=collection_name, schema=schema)
            
            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 256}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            logger.info(f"[MILVUS] Created index for collection: {collection_name}")
        else:
            collection = Collection(name=collection_name)
            logger.info(f"[MILVUS] Using existing collection: {collection_name}")
        
        # Generate embeddings for all documents
        logger.info(f"[MILVUS] Generating embeddings for {len(documents)} documents...")
        texts = [doc.page_content for doc in documents]
        vectors = embedding_model.embed_documents(texts)
        
        # Prepare data for insertion
        data = [
            vectors,  # vector field
            texts,  # text field
            [doc.metadata.get("file_id", file_id) for doc in documents],  # file_id
            [doc.metadata.get("file_name", "") for doc in documents],  # file_name
            [doc.metadata.get("file_path", "") for doc in documents],  # file_path
            [str(doc.metadata.get("page_number", "")) for doc in documents],  # page_number
            [doc.metadata.get("chunk_number", 0) for doc in documents],  # chunk_number
        ]
        
        # Insert data
        collection.insert(data)
        collection.flush()
        
        logger.info(f"[MILVUS] Successfully stored {len(documents)} documents with batch size 128")
        
        # Load collection for search
        collection.load()
        
        logger.info(f"[MILVUS] Documents stored in Milvus for collection: {collection_name}")
        
        # Get token usage
        token_usage = {
            "embedding_tokens": embedding_model.get_total_tokens()
        }
        logger.info(f"[MILVUS] Token usage for file {file_id}: {token_usage}")
        
        return {
            "status": "success",
            "count": len(documents),
            "collection": collection_name,
            "token_usage": token_usage
        }
        
    except Exception as e:
        logger.error(f"[MILVUS] Error storing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store documents in Milvus: {str(e)}"
        )