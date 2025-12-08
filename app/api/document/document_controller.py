"""
Simplified document controller for tool_calling_dev collection.
Endpoints: upload and delete only.
"""
from app.RAG.chunking import chunking_pdf
from app.RAG.embedding import store_documents_in_milvus
from app.api.document.services.pdf_operation import convert_docx_to_pdf_and_return_buffer, process_pdf
from app.utils.response import success_response

from fastapi import APIRouter, File, Form, UploadFile, Request
from loguru import logger
from datetime import datetime
import os
from io import BytesIO
from typing import Optional
from pymilvus import Collection, connections, utility
from app.config import settings

router = APIRouter()


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    file_id: str = Form(...),
    file_name: Optional[str] = Form(None),
):
    """
    Upload a document to the vector database.
    
    Args:
        file: The file to upload (PDF or DOCX)
        file_id: Unique identifier for the file
        file_name: Optional override for filename
        
    Returns:
        Success response with file details
    """
    logger.info(f"[UPLOAD] Starting upload for file_id: {file_id}")
    
    # Read file into memory
    file_bytes = await file.read()
    buffer = BytesIO(file_bytes)
    
    # Resolve filename and extension
    resolved_file_name = file_name or file.filename
    file_ext = resolved_file_name.split(".")[-1].lower() if "." in resolved_file_name else ""
    
    logger.info(f"[UPLOAD] File: {resolved_file_name}, Extension: {file_ext}")
    
    # Save locally for debugging
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    uploads_dir = os.path.join(base_dir, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    local_fs_path = os.path.join(uploads_dir, resolved_file_name)
    
    with open(local_fs_path, "wb") as f:
        f.write(file_bytes)
    
    logger.info(f"[UPLOAD] Saved to: {local_fs_path}")
    
    # Logical path
    logical_file_path = f"local/{resolved_file_name}"
    
    # Process document
    working_buffer = buffer
    if file_ext in ["doc", "docx"]:
        logger.info(f"[UPLOAD] Converting DOCX to PDF")
        working_buffer = convert_docx_to_pdf_and_return_buffer(working_buffer, file_ext)
    
    logger.info(f"[UPLOAD] Processing PDF")
    page_texts = process_pdf(file_id, working_buffer)
    
    logger.info(f"[UPLOAD] Chunking document")
    documents = chunking_pdf(page_texts, file_id, resolved_file_name, logical_file_path, is_artifacts=False)
    
    logger.info(f"[UPLOAD] Storing {len(documents)} chunks in Milvus")
    # Use fixed collection name
    collection_name = "tool_calling_dev"
    result = store_documents_in_milvus(documents, collection_name, file_id)
    
    logger.info(f"[UPLOAD] Successfully stored embeddings for file_id: {file_id}")
    
    return success_response(
        {
            "file_id": file_id,
            "file_name": resolved_file_name,
            "file_path": logical_file_path,
            "file_ext": file_ext,
            "local_fs_path": local_fs_path,
            "chunks_stored": len(documents),
            "collection": collection_name,
            "embedding_tokens": result["embedding_tokens"],
            "rerank_tokens": result["rerank_tokens"]
        },
        201,
    )


@router.post("/delete")
async def delete_document(
    request: Request,
    file_id: str = Form(...),
):
    """
    Delete a document from the vector database by file_id.
    
    Args:
        file_id: File identifier to delete
        
    Returns:
        Success response with deletion count
    """
    logger.info(f"[DELETE] Deleting file_id: {file_id}")
    
    collection_name = "tool_calling_dev"
    
    try:
        # Connect to Milvus
        connections.connect(
            uri=settings.MILVUS_URI,
            db_name=settings.MILVUS_DB_NAME,
            token=settings.MILVUS_TOKEN
        )
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            logger.warning(f"[DELETE] Collection {collection_name} does not exist")
            return success_response({"message": "Collection does not exist", "deleted_count": 0})
        
        # Load collection
        collection = Collection(name=collection_name)
        
        # Delete by expression
        expr = f'file_id == "{file_id}"'
        logger.info(f"[DELETE] Deleting with expression: {expr}")
        
        collection.delete(expr)
        
        logger.info(f"[DELETE] Successfully deleted documents for file_id: {file_id}")
        
        return success_response({"message": "Document deleted successfully", "file_id": file_id})
    
    except Exception as e:
        logger.error(f"[DELETE] Error deleting document: {e}")
        return success_response({"message": f"Error: {str(e)}", "deleted_count": 0}, 500)