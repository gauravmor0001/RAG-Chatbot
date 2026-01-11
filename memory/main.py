from mem0 import Memory
from dotenv import load_dotenv
import os
from openai import OpenAI
import json

load_dotenv()

client=OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
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
    },
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


while True:
    user_query=input("Enter your query: ")
    if user_query.lower() in ['exit', 'quit']:
        break
    
    memories = []
    try:
        search_results = mem_client.search(query=user_query, user_id="DareDevil")
        if search_results:
            # Handle if it's a list of dicts
            if isinstance(search_results, list):
                memories = [
                    f"Memory: {mem.get('memory', mem)}" 
                    for mem in search_results
                ]
            elif isinstance(search_results, dict):
                if 'results' in search_results:
                    memories = [
                        f"Memory: {mem.get('memory', mem)}" 
                        for mem in search_results['results']
                    ]
                elif 'memories' in search_results:
                    memories = [
                        f"Memory: {mem.get('memory', mem)}" 
                        for mem in search_results['memories']
                    ]
        print(f"Found {len(memories)} memories")
    except Exception as e:
        print(f"Search error: {e}")
        # search_results = []
        memories=[]

    if memories:
        SYSTEM_PROMPT=f""" You are a helpful assistant with access to user history.
        here is the context about the user:
        {json.dumps(memories)}
        """
    else:
        SYSTEM_PROMPT = "You are a helpful assistant that helps user"
    
    response=client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user_query}
        ]
    )
    ai_response=response.choices[0].message.content
    print("AI Response:", ai_response)

    try:
        mem_client.add(
            user_id="DareDevil",
            messages=[
                {"role":"user","content":user_query},
                {"role":"assistant","content":ai_response}
            ]
        )

        print("Memory updated.")
    except Exception as e:
        print("Error updating the memory")

print("chat ended")