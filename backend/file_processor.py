from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from datetime import datetime, timedelta
from typing import Optional
import os

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

document_stores = {}

def cleanup_expired_documents():
    """
    Remove documents that have expired (older than 30 minutes)
    Runs automatically before every operation
    """
    current_time = datetime.now()
    expired_conversations = []
    
    for conv_id, data in document_stores.items():
        if current_time > data["expiry"]:
            expired_conversations.append(conv_id)
    
    for conv_id in expired_conversations:
        print(f"DEBUG: Removing expired document for conversation {conv_id}")
        del document_stores[conv_id]
    
    if expired_conversations:
        print(f"DEBUG: Cleaned up {len(expired_conversations)} expired documents")



def process_and_store_file(file_path: str, conversation_id: str, filename: str):
    """
    Process uploaded file and store in RAM with 30-minute expiration
    
    Steps:
    1. Load PDF/TXT file
    2. Split into chunks (1000 chars each)
    3. Create embeddings (vectors)
    4. Store in FAISS (in RAM)
    5. Set expiry timer (30 minutes)
    
    Args:
        file_path: Path to the temporary uploaded file
        conversation_id: Current conversation ID
        filename: Original filename (e.g., "report.pdf")
        
    Returns:
        dict with success status, message, and metadata
    """
    try:
        # Step 0: Clean up any expired documents first
        cleanup_expired_documents()
        
        # Step 1: Load the document based on file type
        if filename.lower().endswith('.pdf'):
            print(f"DEBUG: Loading PDF file: {filename}")
            loader = PyPDFLoader(file_path)
        elif filename.lower().endswith('.txt'):
            print(f"DEBUG: Loading TXT file: {filename}")
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            return {
                "success": False,
                "message": "Unsupported file type. Only PDF and TXT files are supported."
            }
        
        # Load documents (for PDFs, each page is a document)
        documents = loader.load()
        print(f"DEBUG: Loaded {len(documents)} page(s) from {filename}")
        
        if not documents:
            return {
                "success": False,
                "message": "File appears to be empty or couldn't be read."
            }
        
        # Step 2: Split into smaller chunks for better search
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # Each chunk ~1000 characters
            chunk_overlap=200,      # 200 chars overlap between chunks (for context)
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"DEBUG: Split into {len(chunks)} chunks")
        
        # Step 3: Add metadata to each chunk
        for chunk in chunks:
            chunk.metadata.update({
                "filename": filename,
                "conversation_id": conversation_id
            })
        
        # Step 4: Create FAISS vector store in RAM
        # This converts text chunks into embeddings and stores them
        print(f"DEBUG: Creating FAISS vector store (this may take a moment)...")
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        print(f"DEBUG: FAISS vector store created successfully")
        
        # Step 5: Store with expiration time (30 minutes from now)
        expiry_time = datetime.now() + timedelta(minutes=30)
        
        document_stores[conversation_id] = {
            "vector_store": vector_store,
            "expiry": expiry_time,
            "filename": filename,
            "chunks_count": len(chunks)
        }
        
        print(f"DEBUG: Stored {len(chunks)} chunks in RAM for conversation {conversation_id}")
        print(f"DEBUG: Document will expire at {expiry_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            "success": True,
            "message": f"File '{filename}' processed successfully. {len(chunks)} chunks stored. Will expire in 30 minutes.",
            "chunks_count": len(chunks),
            "expiry": expiry_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        print(f"DEBUG: Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Error processing file: {str(e)}"
        }



def search_conversation_documents(query: str, conversation_id: str) -> Optional[str]:
    """
    Search documents for a specific conversation using similarity search
    
    Steps:
    1. Check if conversation has documents
    2. Convert query to embedding
    3. Find most similar chunks
    4. Return relevant text
    
    Args:
        query: User's search query (e.g., "What is the revenue?")
        conversation_id: Conversation ID to search in
        
    Returns:
        Relevant text snippets from the document, or None if no documents found
    """
    # Clean up expired documents first
    cleanup_expired_documents()
    
    # Check if this conversation has any uploaded documents
    if conversation_id not in document_stores:
        print(f"DEBUG: No documents found for conversation {conversation_id}")
        return None
    
    data = document_stores[conversation_id]
    vector_store = data["vector_store"]
    filename = data["filename"]
    
    print(f"DEBUG: Searching in '{filename}' for query: '{query}'")
    
    try:
        # Perform similarity search (find top 3 most relevant chunks)
        results = vector_store.similarity_search(query, k=3)
        
        if not results:
            print(f"DEBUG: No relevant chunks found")
            return None
        
        # Format results nicely
        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"[Excerpt {i} from {filename}]:\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        print(f"DEBUG: Found {len(results)} relevant chunks")
        
        return context
        
    except Exception as e:
        print(f"DEBUG: Error searching documents: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_conversation_document_info(conversation_id: str) -> Optional[dict]:
    """
    Get information about uploaded document for a conversation
    Useful for showing status in the UI
    
    Args:
        conversation_id: Conversation ID
        
    Returns:
        Document info dict with filename, chunks_count, expiry, or None if no document
    """
    cleanup_expired_documents()
    
    if conversation_id not in document_stores:
        return None
    
    data = document_stores[conversation_id]
    return {
        "filename": data["filename"],
        "chunks_count": data["chunks_count"],
        "expiry": data["expiry"].strftime('%Y-%m-%d %H:%M:%S')
    }