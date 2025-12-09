from loguru import logger
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, status
from app.RAG.embedding import Document
import re
from bs4 import BeautifulSoup
import spacy
import subprocess
import sys

def load_spacy_model(model_name="en_core_web_sm", max_length=9_999_999):
    try:
        # Try loading the model
        nlp = spacy.load(model_name)
    except OSError:
        # Model not found, so download it
        print(f"spaCy model '{model_name}' not found. Downloading...")
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
        nlp = spacy.load(model_name)
    
    nlp.max_length = max_length
    return nlp

# Load the model safely
nlp = load_spacy_model()


def extract_page_texts(xhtml_content: str) -> List[str]:
    soup = BeautifulSoup(xhtml_content, 'html.parser')
    page_divs = soup.find_all('div', class_='page')

    if page_divs:
        return [div.get_text(separator=' ', strip=True) for div in page_divs]

    return [soup.get_text(separator=' ', strip=True)]

def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[\d+\]', '', text)
    return text.strip()

def chunk_text_by_words(text: str, word_limit: int = 150) -> List[str]:
    doc = nlp(text)
    chunks, current_chunk, word_count = [], [], 0

    for sent in doc.sents:
        words = sent.text.split()
        word_count += len(words)
        current_chunk.append(sent.text)

        if word_count >= word_limit:
            chunk = " ".join(current_chunk)
            if len(chunk) >= 20:
                chunks.append(chunk)
            current_chunk, word_count = [], 0

    if current_chunk:
        chunk = " ".join(current_chunk)
        if len(chunk) >= 20:
            chunks.append(chunk)

    return list(dict.fromkeys(chunks))  # Remove duplicates while preserving order

def create_document_objects(chunks: List[str], file_id: str, file_name: str, file_path: str, page_number: int, is_artifacts: bool) -> List[Document]:
    documents = []

    for idx, chunk in enumerate(chunks):
        cleaned = preprocess_text(chunk)
        metadata = {
            "chunk_number": idx + 1,
            "file_id": file_id,
            "file_name": file_name,
            "file_path": file_path,
            "page_number": str(page_number) if is_artifacts else page_number,
            "exact_data": chunk
        }
        documents.append(Document(page_content=cleaned, metadata=metadata))

    return documents

def chunking_pdf(page_texts: List[str], file_id: str, file_name: str, file_path: str, is_artifacts: bool) -> List[Document]:
    all_documents = []

    for page_number, text in enumerate(page_texts, start=1):
        chunks = chunk_text_by_words(text)
        docs = create_document_objects(chunks, file_id, file_name, file_path, page_number, is_artifacts)
        all_documents.extend(docs)

    logger.info(f"Created {len(all_documents)} document chunks from PDF.")

    return all_documents