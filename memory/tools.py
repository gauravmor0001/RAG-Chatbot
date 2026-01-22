from datetime import datetime
from langchain_core.tools import tool
# from duckduckgo_search import DDGS
# from pydantic import BaseModel, Field
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import TavilySearchResults
import wikipedia

# class SearchInput(BaseModel):
#     query: str = Field(description="The query to search for on the internet.")

@tool
def get_current_time():
    """Get the current real-time date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
@tool
def web_search(query: str):
    """
    Search the internet for real-time information, news, weather, or facts.
    Use this when the user asks about current events or topics you don't know.
    """
    try:
        
        tool = TavilySearchResults(max_results=3)
        # this automatically access the api_key.

        results = tool.invoke({"query": query})
        output = []
        for res in results:
            output.append(f"Source: {res.get('url')}\nContent: {res.get('content')}")
            
        return "\n\n".join(output)
    
    except Exception as e:
        return f"Search failed: {str(e)}"
    
@tool
def search_wikipedia(query:str):
    """
    Search Wikipedia for definitions, historical facts, or technical concepts.
    Use this for academic topics, famous people, or established knowledge.
    """
    try:
        return wikipedia.summary(query,sentences=3)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous query. Options: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "Page not found on Wikipedia."
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"


tools_list = [get_current_time , web_search,search_wikipedia]