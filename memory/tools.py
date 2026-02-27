from datetime import datetime
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
import wikipedia
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
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
            
        final_output = "\n\n".join(output)
        print(f"DEBUG: Web search returned {len(results)} results")
        print(f"DEBUG: Full output:\n{final_output}")  # ← Add this
        return final_output
    
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
    

@tool
def search_knowledge_base(query: str):
    """
    Use this tool to search for information inside the uploaded PDF documents or text files.
    Input should be a specific search query related to the documents.
    Returns the relevant text snippets from the files.
    """
    print(f"DEBUG: Searching Knowledge Base for: '{query}'")
    try:
        vector_db = QdrantVectorStore.from_existing_collection(
            embedding=embedding_model,
            url="http://localhost:6333",
            collection_name="learning-rag"
        )
        results = vector_db.similarity_search(query, k=3)
        if not results:
            return "No relevant information found in the documents."
        context = "\n\n".join([f"Snippet: {res.page_content}" for res in results])
        return context

    except Exception as e:
        return f"Error searching documents: {str(e)}"


tools_list = [get_current_time , web_search,search_wikipedia,search_knowledge_base]