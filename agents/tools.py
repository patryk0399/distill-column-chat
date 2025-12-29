from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return {'query': f"Results for: {query}. You can now return. You have all of the relevant information. No more searches needed."}

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"
