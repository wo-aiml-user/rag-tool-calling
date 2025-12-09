"""
Tool execution functions for DeepSeek function calling.
All tool implementations with proper token tracking and detailed logging.
"""
import os
from typing import List, Dict, Optional,Any
from dotenv import load_dotenv
from loguru import logger
from pymilvus import Collection, connections, utility
from app.RAG.voyage_embeddings import VoyageEmbeddings
from app.RAG.embedding import Document
from app.config import settings
from perplexity import Perplexity
import requests

load_dotenv()

def _get_perplexity_client() -> Perplexity:
    """Get Perplexity API client."""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable is required for web search")
    return Perplexity(api_key=api_key)


def search_articles(query: str, max_results: int = 2) -> List[Dict]:
    """
    Perform a Perplexity search for a single query.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, content
    """
    logger.info(f"[PERPLEXITY] query='{query}' | max_results={max_results}")
    client = _get_perplexity_client()
    
    try:
        search = client.search.create(query=[query])
        
        results = []
        for result in search.results:
            norm = {
                "title": getattr(result, 'title', ''),
                "url": getattr(result, 'url', ''),
                "content": getattr(result, 'content', '') or getattr(result, 'snippet', '') or '',
                "score": getattr(result, 'score', None),
                "published_date": getattr(result, 'published_date', '') or getattr(result, 'date', '') or '',
                "type": "search"
            }
            results.append(norm)
        
        logger.info(f"[PERPLEXITY] Found {len(results)} results")
        return results[:max_results]
    except Exception as e:
        logger.error(f"[PERPLEXITY] Search failed: {e}")
        return []

def get_current_weather(location: str, unit: str = "celsius") -> Dict:
    """
    Get current weather for a location using OpenWeatherMap API.
    
    Args:
        location: City name (e.g., "London", "Tokyo")
        unit: Temperature unit ("celsius" or "fahrenheit")
        
    Returns:
        Weather information dictionary
    """
    logger.info(f"[WEATHER] location='{location}' | unit={unit}")
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        logger.warning("[WEATHER] Missing OPENWEATHER_API_KEY, using mock data")
        # Fallback to mock data
        weather_data = {
            "location": location,
            "temperature": 22 if unit == "celsius" else 72,
            "unit": unit,
            "condition": "Partly cloudy",
            "humidity": 65,
            "wind_speed": 10,
            "type": "weather"
        }
        return weather_data
    
    # Map user-friendly units to API 'units' parameter
    unit_map = {"celsius": "metric", "fahrenheit": "imperial"}
    api_unit = unit_map.get(unit.lower(), "metric")
    
    # API Endpoint
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    params = {
        "q": location,
        "appid": api_key,
        "units": api_unit
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant data
        weather_info = {
            "location": data.get("name"),
            "temperature": data["main"]["temp"],
            "unit": unit,
            "condition": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "type": "weather"
        }
        
        logger.info(f"[WEATHER] Successfully fetched weather for {location}")
        return weather_info
        
    except Exception as e:
        logger.error(f"[WEATHER] API error: {e}, using mock data")
        # Fallback to mock data on error
        return {
            "location": location,
            "temperature": 22 if unit == "celsius" else 72,
            "unit": unit,
            "condition": "Data unavailable",
            "humidity": 0,
            "wind_speed": 0,
            "error": str(e),
            "type": "weather"
        }

def retrieve_documents(
    query: str,
    collection_name: str,
    file_ids: Optional[List[str]] = None,
    top_k: int = 8
) -> tuple[List[Document], Dict[str, int]]:
    """
    Retrieve relevant documents from Milvus with token tracking.
    
    Args:
        query: Search query
        collection_name: Milvus collection name
        file_ids: Optional list of file IDs to filter by
        top_k: Number of documents to retrieve (default: 8)
        
    Returns:
        Tuple of (documents, token_usage)
    """
    logger.info(f"[RETRIEVAL] query='{query}' | collection={collection_name} | file_ids={file_ids}")
    
    token_usage = {
        "embedding_tokens": 0
    }
    
    try:
        # Connect to Milvus
        connections.connect(
            uri=settings.MILVUS_URI,
            db_name=settings.MILVUS_DB_NAME,
            token=settings.MILVUS_TOKEN
        )
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            logger.warning(f"[RETRIEVAL] Collection {collection_name} does not exist")
            return [], token_usage
        
        # Load collection
        collection = Collection(name=collection_name)
        collection.load()
        
        # Initialize embedding
        embedding = VoyageEmbeddings(
            model="voyage-3-large",
            api_key=settings.VOYAGE_API_KEY,
            batch_size=128,
            truncation=True
        )
        
        # Generate query embedding
        logger.info(f"[RETRIEVAL] Generating embedding for query...")
        query_vector = embedding.embed_query(query)
        embedding_tokens = embedding.get_total_tokens()
        token_usage["embedding_tokens"] = embedding_tokens
        logger.info(f"[RETRIEVAL] Embedding generated | tokens={embedding_tokens}")
        
        # Build search expression for file filtering
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }
        
        expr = None
        if file_ids:
            expr = " or ".join([f'file_id == "{fid}"' for fid in file_ids])
            logger.info(f"[RETRIEVAL] Filtering by file_ids: {file_ids}")
        
        # Search
        logger.info(f"[RETRIEVAL] Searching Milvus for top {top_k} documents...")
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "file_id", "file_name", "file_path", "page_number", "chunk_number"]
        )
        
        # Convert results to documents
        documents = []
        if results and len(results) > 0:
            for idx, hit in enumerate(results[0]):
                doc = Document(
                    page_content=hit.entity.get("text", ""),
                    metadata={
                        "file_id": hit.entity.get("file_id", ""),
                        "file_name": hit.entity.get("file_name", ""),
                        "file_path": hit.entity.get("file_path", ""),
                        "page_number": hit.entity.get("page_number", ""),
                        "chunk_number": hit.entity.get("chunk_number", 0),
                        "score": hit.score,
                        "index": idx
                    }
                )
                documents.append(doc)
        
        logger.info(f"[RETRIEVAL] Final result: {len(documents)} documents | total_tokens={token_usage}")
        return documents, token_usage
        
    except Exception as e:
        logger.error(f"[RETRIEVAL] Error retrieving documents: {e}")
        return [], token_usage


def format_documents_for_llm(documents: List[Document]) -> str:
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


def execute_tool(function_name: str, function_args: Dict, collection_name: Optional[str] = None) -> tuple[str, Any, Dict[str, int]]:
    """
    Execute a tool function by name and return result with token usage.
    
    Args:
        function_name: Name of the function to execute
        function_args: Arguments for the function
        collection_name: Milvus collection name (for retrieval)
        
    Returns:
        Tuple of (formatted_result, raw_context, token_usage)
    """
    logger.info(f"[TOOL_EXEC] Executing {function_name}")
    logger.info(f"[TOOL_EXEC] Arguments: {function_args}")
    
    token_usage = {
        "embedding_tokens": 0
    }
    
    try:
        if function_name == "retrieve_documents":
            if not collection_name:
                logger.error("[TOOL_EXEC] Collection name required for retrieval")
                return "Error: Collection name required for document retrieval", [], token_usage
            
            query = function_args.get("query", "")
            file_ids = function_args.get("file_ids")
            
            logger.info(f"[TOOL_EXEC] Retrieving documents for query: '{query}'")
            
            documents, retrieval_tokens = retrieve_documents(
                query=query,
                collection_name=collection_name,
                file_ids=file_ids
            )
            
            # Update token usage
            token_usage.update(retrieval_tokens)
            
            result = format_documents_for_llm(documents)
            logger.info(f"[TOOL_EXEC] Retrieved {len(documents)} documents | tokens={retrieval_tokens}")
            logger.info(f"[TOOL_EXEC] Result preview: {result}")
            
            return result, documents, token_usage
        
        elif function_name == "search_articles":
            query = function_args.get("query", "")
            max_results = function_args.get("max_results", 2)
            
            logger.info(f"[TOOL_EXEC] Searching articles for: '{query}'")
            
            results = search_articles(query=query, max_results=max_results)
            
            if not results:
                result = f"No articles found for query: {query}"
            else:
                formatted = []
                for i, article in enumerate(results):
                    formatted.append(
                        f"Article {i+1}:\n"
                        f"  Title: {article.get('title', 'N/A')}\n"
                        f"  URL: {article.get('url', 'N/A')}\n"
                        f"  Content: {article.get('content', 'N/A')}\n"
                    )
                result = "\n".join(formatted)
            
            logger.info(f"[TOOL_EXEC] Found {len(results)} articles")
            logger.info(f"[TOOL_EXEC] Result preview: {result}...")
            
            return result, results, token_usage
        
        elif function_name == "get_current_weather":
            location = function_args.get("location", "")
            unit = function_args.get("unit", "celsius")
            
            logger.info(f"[TOOL_EXEC] Getting weather for: {location}")
            
            weather = get_current_weather(location=location, unit=unit)
            
            result = (
                f"Weather in {weather['location']}:\n"
                f"  Temperature: {weather['temperature']}°{weather['unit'][0].upper()}\n"
                f"  Condition: {weather['condition']}\n"
                f"  Humidity: {weather['humidity']}%\n"
                f"  Wind Speed: {weather.get('wind_speed', 0)} km/h"
            )
            
            logger.info(f"[TOOL_EXEC] Weather result: {result}")
            
            return result, [weather], token_usage
        
        else:
            logger.error(f"[TOOL_EXEC] Unknown function: {function_name}")
            return f"Error: Unknown function '{function_name}'", [], token_usage
    
    except Exception as e:
        logger.error(f"[TOOL_EXEC] Error executing {function_name}: {e}")
        return f"Error executing {function_name}: {str(e)}", [], token_usage

