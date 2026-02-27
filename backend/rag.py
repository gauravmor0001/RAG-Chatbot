from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="rag"
)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0 #creativity
)

user_query = input("Enter your question: ")
print("Searching database...")
search_result = vector_db.similarity_search(
    query=user_query,
    k=4
)
context_text = "\n\n".join([
    f"Content: {res.page_content}\nSource: {res.metadata.get('source', 'Unknown')}" 
    for res in search_result
])
if not context_text:
    print("No relevant documents found in document.")
    exit()