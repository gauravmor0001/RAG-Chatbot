from datetime import datetime
from langchain_core.tools import tool
# from duckduckgo_search import DDGS
# from pydantic import BaseModel, Field
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import TavilySearchResults

# class SearchInput(BaseModel):
#     query: str = Field(description="The query to search for on the internet.")

@tool
def get_current_time():
    """Get the current real-time date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
@tool
def web_search(query: str):
    """
    Search the internet for real-time information, news, or facts.
    Use this when the user asks about current events or topics you don't know.
    """
    try:
        # with DDGS() as ddgs:
        #     results=list(ddgs.text(query,max_results=5))
        #     if not results:
        #         return "No results found."
            
        #     formatted_results=[]
        #     for result in results:
        #         formatted_results.append(f"Title: {result['title']}\nLink: {result['href']}\nSnippet: {result['body']}")
        #     return "\n\n".join(formatted_results)

        #     search = DuckDuckGoSearchRun()
        #     return search.invoke(query)
        
        tool = TavilySearchResults(max_results=5)
        # this automatically access the api_key.

        results = tool.invoke({"query": query})
        output = []
        for res in results:
            output.append(f"Source: {res.get('url')}\nContent: {res.get('content')}")
            
        return "\n\n".join(output)
    
    except Exception as e:
        return f"Search failed: {str(e)}"





# tools_schema = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_current_time",
#             "description": "Get the current real-time date and time",
#             "parameters": {
#                 "type": "object",
#                 "properties": {}, 
#                 "required": [],
#             },
#         },
#     }
# ]
# not needed with langraph,it will auto generate.

tools_list = [get_current_time , web_search]