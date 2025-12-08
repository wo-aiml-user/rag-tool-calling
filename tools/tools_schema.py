"""
All tool schemas for DeepSeek function calling.
Includes: retrieval, web search, and weather tools.
"""

# Retrieval tool schema
RETRIEVAL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_documents",
        "description": "Retrieve relevant documents from the knowledge base to answer user questions. Use this when the user asks about specific documents or topics that might be in the uploaded files.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents"
                },
                "file_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific file IDs to search within"
                }
            },
            "required": ["query"]
        }
    }
}

# Web search tool schema
SEARCH_ARTICLES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_articles",
        "description": "Search the web for articles related to a query using Perplexity API. Use this when you need current information or when the answer is not in the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or topic to search for, e.g. 'AI regulation news'."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of articles to return.",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 1
                }
            },
            "required": ["query"]
        }
    }
}

# Weather tool schema
WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather for a specific city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use."
                }
            },
            "required": ["location"]
        }
    }
}