import re
from loguru import logger
from typing import List, Dict, Any, Set

from .json_parser import parse_json_response


def format_page_number(page_number: Any) -> Any:
    """
    Extract the numerical value from a page number string.
    """
    if not page_number:
        return ""
    if isinstance(page_number, int):
        return page_number
    if isinstance(page_number, str):
        if page_number.isdigit():
            return int(page_number)
        match = re.search(r'Page\s+(\d+)', page_number, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return page_number

def format_rag_response(response: Dict, is_first_conversation: bool, user_query: str) -> Dict:
    """
    Process and format the RAG response with metadata.
    """
    try:
        response_string = response["answer"]
        logger.info(f"Processing Chat LLM response: {response_string}")

        cleaned = re.sub(
            r'(?is)("ai"\s*:\s*")([\s\S]*?)(?:\\n(?:-{3,}\\n|\\n)?\s*)?[^\n\\]*\bReferences:[\s\S]*?(")',
            r'\1\2\3',
            response_string
        )

        json_response = parse_json_response(cleaned)
        logger.info(f"User Query: {user_query} \n Parsed response: {json_response}")

        formatted_response = {
            "response": json_response.get("ai", "No answer/reference found."),
            "meta_data": [],
            "token_usage": response.get("token_usage", {})
        }

        if is_first_conversation and "title" in json_response:
            formatted_response["title"] = json_response["title"]

        context_utilized = json_response.get("context_utilized", False)
        if isinstance(context_utilized, str):
            context_utilized = context_utilized.lower() == 'true'

        if not context_utilized:
            logger.info("Context was not utilized in response")
            return formatted_response

        doc_refs = json_response.get("document_references", [])
        if not doc_refs:
            logger.info("No document references found")
            return formatted_response

        try:
            processed_refs = [int(ref) for ref in doc_refs if str(ref).strip().isdigit()]
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing document references: {e}")
            return formatted_response

        context_list = response.get("context", [])
        if not context_list:
            logger.warning("No context list found")
            return formatted_response

        seen_texts: Set[str] = set()
        meta_data: List[Dict] = []
        for idx in processed_refs:
            if 0 <= idx < len(context_list):
                context_item = context_list[idx]
                if hasattr(context_item, 'metadata'):
                    text = context_item.metadata.get("exact_data", "")
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        meta_data.append({
                            "text": text,
                            "page": format_page_number(context_item.metadata.get("page_number", "")),
                            "file_id": context_item.metadata.get("file_id", ""),
                            "file_name": context_item.metadata.get("file_name", ""),
                            "file_path": context_item.metadata.get("file_path", "")
                        })
        
        formatted_response["meta_data"] = meta_data
        logger.info(f"Final formatted response with {len(meta_data)} unique references")
        return formatted_response

    except Exception as e:
        logger.error(f"Error in format_rag_response: {e}")
        return {
            "response": "I apologize, but I encountered an error processing the response. Please try again.",
            "meta_data": [],
            "token_usage": {}
        }