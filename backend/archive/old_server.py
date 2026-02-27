# from mem0 import Memory
# from dotenv import load_dotenv
# import os
# import json
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import uvicorn
# import time
# import re
# from langchain_core.messages import AIMessage

# from langchain_groq import ChatGroq
# from langchain_core.messages import SystemMessage, HumanMessage
# from langgraph.graph import StateGraph, MessagesState, END
# from langgraph.prebuilt import ToolNode, tools_condition

# from tools import tools_list

# load_dotenv()

# # --- THE ONLY WORKING SMART MODEL ON GROQ ---
# # This is the new standard. It is smart but has a 6,000 token/min limit on free tier.
# MODEL_NAME = "llama-3.1-8b-instant" 

# class ChatRequest(BaseModel):
#     message: str

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# print(f"DEBUG: Loading Model: {MODEL_NAME}")
# llm = ChatGroq(
#     model=MODEL_NAME, 
#     api_key=os.getenv("GROQ_API_KEY"),
#     temperature=0,
#     #  model_kwargs={
#     #     "response_format": {"type": "json_object"}
#     # }
# )
# llm_with_tools = llm.bind_tools(tools_list)

# config = {
#     "version": "v1.1",
#     "embedder": {
#         "provider": "huggingface",
#         "config": {
#             "api_key": os.getenv("HUGGINGFACEHUB_API_TOKEN"),
#             "model": "sentence-transformers/all-MiniLM-L6-v2"
#         }
#     },
#     "llm": {
#         "provider": "groq",
#         "config": {
#             "api_key": os.getenv("GROQ_API_KEY"),
#             "model": MODEL_NAME, # Updated to 3.3
#         }
#     },
#     "vector_store": {
#         "provider": "qdrant",
#         "config": {
#             "host": "localhost",
#             "port": 6333,
#             "collection_name": "chat_memory",
#         }
#     }
# }

# print("DEBUG: Connecting to Memory...")
# mem_client = Memory.from_config(config)
# print("DEBUG: Memory Connected.")

# # 
# def normalize_tool_calls(state: MessagesState):
#     last = state["messages"][-1]

#     if not isinstance(last, AIMessage):
#         return state

#     content = last.content or ""

#     # Detect Groq XML tool call
#     # match = re.search(r'<function=(\w+)\s*(\{.*?\})?>', content)
#     match = re.search(
#     r'<function=([a-zA-Z0-9_\-]+)\s*(\{[\s\S]*?\})?>',
#     content
#     )

#     if not match:
#         return state

#     tool_name = match.group(1)
#     args_raw = match.group(2)

#     try:
#         args = json.loads(args_raw) if args_raw else {}
#     except:
#         args = {}

#     # last.tool_calls = [{
#     #     "name": tool_name,
#     #     "args": args
#     # }]

#     # return {"messages": state["messages"]}
#     new_message = AIMessage(
#     content=last.content,
#     tool_calls=[{
#         "name": tool_name,
#         "args": args
#     }]
#     )

#     state["messages"][-1] = new_message
#     return {"messages": state["messages"]}


# # 

# def reasoner(state: MessagesState):
#     response = llm_with_tools.invoke(state["messages"])
#     return {"messages": state["messages"] + [response]}

# workflow = StateGraph(MessagesState)
# workflow.add_node("agent", reasoner)
# workflow.add_node("tools", ToolNode(tools_list))
# workflow.add_node("normalize", normalize_tool_calls)
# workflow.set_entry_point("agent")
# workflow.add_edge("agent", "normalize")
# workflow.add_conditional_edges("normalize", tools_condition)
# # workflow.add_conditional_edges("agent", tools_condition)
# workflow.add_edge("tools", "agent")
# agent_app = workflow.compile()

# @app.post("/chat")
# async def chat_endpoint(request: ChatRequest):
#     user_query = request.message
#     print(f"\n--- REQUEST: '{user_query}' ---")

#     factual_keywords = ['what is', 'define', 'explain', 'tell me about', 'who is', 'how does']
#     asking_for_facts = any(keyword in user_query.lower() for keyword in factual_keywords)
    

#     # 1. RETRIEVE MEMORIES (Strict Limit = 1 to save tokens)
#     memories = []
#     if asking_for_facts:
#         print("DEBUG: Factual question - skipping memory retrieval")
#         # Skip memory for factual questions
#     else:
#         # Your existing memory retrieval code
#         try:
#             search_results = mem_client.search(query=user_query, user_id="DareDevil", limit=3)
#             if search_results:
#                 raw = search_results if isinstance(search_results, list) else search_results.get("results", [])
#                 for mem in raw:
#                     score = mem.get('score', 0)
#                     if score > 0.7:
#                         text = mem.get('memory', str(mem))[:200]
#                         memories.append(text)
#                         print(f"DEBUG: Using memory (score: {score}): {text[:50]}...")
#                     else:
#                         print(f"DEBUG: Skipping low-relevance memory (score: {score})")
#         except Exception as e:
#             print(f"DEBUG: Memory Error: {e}")

#     base_prompt = (
#         "You are a helpful assistant. "
#         "When using tools, strictly follow the JSON tool calling format."
#         "Answer the user's current question directly and accurately."
#     )

#     # 2. SYSTEM PROMPT
#     if memories:
#         SYSTEM_PROMPT = (
#             f"{base_prompt}\n\n"
#             f"CONTEXT FROM PREVIOUS CONVERSATIONS:\n"
#             f"{chr(10).join('- ' + m for m in memories)}\n\n"
#             f"Use this context ONLY if relevant to the current question. "
#             f"DO NOT repeat old answers. Always prioritize answering the user's current question."
#         )
#     else:
#         SYSTEM_PROMPT = base_prompt

#     input_messages = [
#         SystemMessage(content=SYSTEM_PROMPT),
#         HumanMessage(content=user_query)
#     ]

#     ai_response = ""
#     try:
#         print(f"DEBUG: Input messages: {[m.content[:100] if hasattr(m, 'content') else str(m)[:100] for m in input_messages]}")
        
#         final_state = agent_app.invoke({"messages": input_messages})
        
#         print(f"DEBUG: All messages in final state:")
#         for i, msg in enumerate(final_state["messages"]):
#             print(f"  [{i}] {type(msg).__name__}: {str(msg.content)[:200]}")
#             if hasattr(msg, 'tool_calls') and msg.tool_calls:
#                 print(f"      Tool calls: {msg.tool_calls}")
        
#         ai_response = final_state["messages"][-1].content
#         print(f"DEBUG: Final response: {ai_response}")
        
#         # 4. SAVE MEMORY
#         try:
#             mem_client.add(
#                 user_id="DareDevil",
#                 messages=[
#                     {"role": "user", "content": user_query}, 
#                     {"role": "assistant", "content": ai_response}
#                 ]
#             )
#             print("DEBUG: Memory saved successfully")
#         except Exception as mem_err:
#             print(f"DEBUG: Failed to save memory: {mem_err}")
        
#         return {"response": ai_response}
        
#     except Exception as e:
#         error_msg = str(e)
#         if "rate_limit_exceeded" in error_msg or "413" in error_msg:
#             print("DEBUG: Rate Limit Hit!")
#             return {"response": "I am thinking too hard (Speed Limit Reached). Please wait 30 seconds and try again."}
#         else:
#             print(f"DEBUG: Unknown Error: {error_msg}")
#             return {"response": f"System Error: {error_msg}"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=5000)

