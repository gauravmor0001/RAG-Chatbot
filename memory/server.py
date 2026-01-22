from mem0 import Memory
from dotenv import load_dotenv
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import re #using it to find xml.
from langchain_core.messages import AIMessage

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition

from tools import tools_list

load_dotenv()

MODEL_NAME = "llama-3.1-8b-instant" 

class ChatRequest(BaseModel):
    message: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"DEBUG: Loading Model: {MODEL_NAME}")
llm = ChatGroq(
    model=MODEL_NAME, 
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3, #this control creativity, more creativity high the value
)
llm_with_tools = llm.bind_tools(tools_list)
llm_without_tools = llm 

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "huggingface",
        "config": {
            "api_key": os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    },
    "llm": {
        "provider": "groq",
        "config": {
            "api_key": os.getenv("GROQ_API_KEY"),
            "model": MODEL_NAME,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "chat_memory",
        }
    }
}

print("DEBUG: Connecting to Memory...")
mem_client = Memory.from_config(config)
print("DEBUG: Memory Connected.")

# Groq is formating tool calls in XML,but lanchain want JSON.
def normalize_tool_calls(state: MessagesState):
    last = state["messages"][-1]

    # user dont call tools, if llm:
    if not isinstance(last, AIMessage):
        return state

    content = last.content or ""

    match = re.search(
        r'<function=([a-zA-Z0-9_\-]+)\s*(\{[\s\S]*?\})?>',
        content
    )

    if not match:
        return state

    tool_name = match.group(1)
    args_raw = match.group(2)

    try:
        args = json.loads(args_raw) if args_raw else {}
    except:
        args = {}

    new_message = AIMessage(
        content=last.content,
        tool_calls=[{
            "name": tool_name,
            "args": args
        }]
    )

    state["messages"][-1] = new_message
    return {"messages": state["messages"]}

def reasoner(state: MessagesState):
    """Reasoner WITH tools for complex questions"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def reasoner_no_tools(state: MessagesState):
    """Reasoner WITHOUT tools for factual questions"""
    response = llm_without_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# Workflow WITH tools
workflow = StateGraph(MessagesState)
workflow.add_node("agent", reasoner)
workflow.add_node("tools", ToolNode(tools_list))
workflow.add_node("normalize", normalize_tool_calls)
workflow.set_entry_point("agent")
workflow.add_edge("agent", "normalize")
workflow.add_conditional_edges("normalize", tools_condition)
workflow.add_edge("tools", "agent")
agent_app = workflow.compile()

# Workflow WITHOUT tools
simple_workflow = StateGraph(MessagesState)
simple_workflow.add_node("agent", reasoner_no_tools)
simple_workflow.set_entry_point("agent")
simple_workflow.add_edge("agent", END)
simple_agent = simple_workflow.compile()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.message
    print(f"\n--- REQUEST: '{user_query}' ---")

    realtime_keywords = [
        'current', 'today', 'now', 'latest', 'recent', 'this week', 'this month',
        'weather', 'news', 'price', 'stock', 'gdp', 'election', 'score'
    ]
    
    factual_keywords = [
        'what is', 'what are', 'define', 'explain', 'tell me about', 
        'who is', 'who are', 'how does', 'describe', 'meaning of'
    ]
    
    # Check if question needs real-time data
    needs_realtime = any(keyword in user_query.lower() for keyword in realtime_keywords)
    
    # Only treat as simple factual question if it doesn't need real-time data
    asking_for_facts = (
        any(keyword in user_query.lower() for keyword in factual_keywords) 
        and not needs_realtime
    )
    
    # 1. RETRIEVE MEMORIES
    memories = []
    
    if asking_for_facts:
        print("DEBUG: Factual question - skipping memory retrieval")
    else:
        try:
            search_results = mem_client.search(query=user_query, user_id="DareDevil", limit=3)
            if search_results:
                raw = search_results if isinstance(search_results, list) else search_results.get("results", [])
                for mem in raw:
                    score = mem.get('score', 0)
                    if score > 0.7:
                        text = mem.get('memory', str(mem))[:200]
                        memories.append(text)
                        print(f"DEBUG: Using memory (score: {score}): {text[:50]}...")
                    else:
                        print(f"DEBUG: Skipping low-relevance memory (score: {score})")
        except Exception as e:
            print(f"DEBUG: Memory Error: {e}")

    base_prompt = (
        "You are a helpful assistant. "
        "Answer the user's current question directly and accurately. "
        "For common knowledge questions, answer from your knowledge. "
        "Only use tools for current events, news, weather, or real-time information."
    )

    if memories:
        SYSTEM_PROMPT = (
            f"{base_prompt}\n\n"
            f"CONTEXT FROM PREVIOUS CONVERSATIONS:\n"
            f"{chr(10).join('- ' + m for m in memories)}\n\n"
            f"Use this context ONLY if relevant to the current question. "
            f"DO NOT repeat old answers. Always prioritize answering the user's current question."
        )
    else:
        SYSTEM_PROMPT = base_prompt

    input_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query)
    ]
    
    ai_response = ""
    try:
        
        if asking_for_facts:
            final_state = simple_agent.invoke({"messages": input_messages})
        else:
            final_state = agent_app.invoke({"messages": input_messages})
        
        # for i, msg in enumerate(final_state["messages"]):
        #     print(f"  [{i}] {type(msg).__name__}: {str(msg.content)[:200]}")
        #     if hasattr(msg, 'tool_calls') and msg.tool_calls:
        #         print(f"      Tool calls: {msg.tool_calls}")
        
        ai_response = final_state["messages"][-1].content
        print(f"DEBUG: Final response: {ai_response}")
        
        # 4. SAVE MEMORY
        try:
            mem_client.add(
                user_id="DareDevil",
                messages=[
                    {"role": "user", "content": user_query}, 
                    {"role": "assistant", "content": ai_response}
                ]
            )
            print("DEBUG: Memory saved successfully")
        except Exception as mem_err:
            print(f"DEBUG: Failed to save memory: {mem_err}")
        
        return {"response": ai_response}
        
    except Exception as e:
        error_msg = str(e)
        if "rate_limit_exceeded" in error_msg or "413" in error_msg:
            print("DEBUG: Rate Limit Hit!")
            return {"response": "I am thinking too hard (Speed Limit Reached). Please wait 30 seconds and try again."}
        else:
            print(f"DEBUG: Unknown Error: {error_msg}")
            return {"response": f"System Error: {error_msg}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)