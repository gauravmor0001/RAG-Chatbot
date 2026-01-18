from tools import tools_schema, available_functions
from mem0 import Memory
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel #used to check wether the data we receive is in correct format or not.
import uvicorn

load_dotenv()

class ChatRequest(BaseModel):
    message : str

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], #Accepts all HTTP methods (GET, POST, PUT, DELETE, etc.).
    allow_headers=["*"],
)

# client=OpenAI(
#     base_url="https://router.huggingface.co/v1",
#     api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# ) not good model for handling agents.

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

config={
    "version":"v1.1",
    "embedder":{
        "provider":"huggingface",
        "config":{
            "api_key":os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            "model":"sentence-transformers/all-MiniLM-L6-v2"
        }
        # my vector dimension is 1536, so the embedding model should produce embeddings of that size also.and since i am not able to fins any free model that does this so i have to change vector database dimesnions.
    },
    "llm":{
        "provider":"groq",
        "config":{
            "api_key":os.getenv("GROQ_API_KEY"),
            "model":"llama-3.3-70b-versatile",
        }
    }, #the librarian (decide what to store)
    "vector_store":{
        "provider":"qdrant",
        "config":{
            "host":"localhost",
            "port":6333,
            "collection_name":"chat_memory",
        }
    }
}

mem_client=Memory.from_config(config)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.message
    memories = []
    try:
        search_results = mem_client.search(query=user_query, user_id="DareDevil")
        if isinstance(search_results, list):
            memories = [f"Memory: {mem.get('memory', mem)}" for mem in search_results]
        elif isinstance(search_results, dict):
            if 'results' in search_results:
                memories = [f"Memory: {mem.get('memory', mem)}" for mem in search_results['results']]
            elif 'memories' in search_results:
                memories = [f"Memory: {mem.get('memory', mem)}" for mem in search_results['memories']]
    except Exception as e:
        print(f"Search error: {e}")

    messages = []
    if memories:
        SYSTEM_PROMPT = f"""You are a helpful assistant with access to user history.
        Context about the user:
        {json.dumps(memories)}"""
    else:
        SYSTEM_PROMPT = "You are a helpful assistant that helps user"
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_query})

    try:
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )
        ai_response = response.choices[0].message
        tool_calls = ai_response.tool_calls

        if tool_calls:
            print("Agent decided to use a tool...") # Debug print
            
            # 1. Add the Assistant's "Thought" to history
            messages.append(ai_response)

            # 2. Run the Tools
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                
                if function_to_call:
                    # Run the Python function
                    # (Currently checks time, no arguments needed)
                    function_result = function_to_call()
                    
                    # Add the "Observation" (Result) to history
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_result,
                    })
            final_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )
            ai_response = final_response.choices[0].message.content
            
        else:
            # No tool needed, just a normal chat reply
            ai_response = ai_response.content
        

        mem_client.add(
            user_id="DareDevil",
            messages=[
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": ai_response}
            ]
        )
        
        return {"response": ai_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
    # if we dont use this everytime the file is imported uvicorn will run and start the server which is not desired behaviour.
print("chat ended")