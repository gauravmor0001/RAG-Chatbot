import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

def process_and_store_pdf(file_path: str):
    embedding_model=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(file_path):
        print("File not found:")
        return False
    
    print("Loading Documnet...")
    loader=PyPDFLoader(file_path)
    docs=loader.load()
    print(f"Found {len(docs)} pages")

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    splits=text_splitter.split_documents(docs)
    QdrantVectorStore.from_documents(splits,embedding_model,collection_name="learning-rag",url="http://localhost:6333")
    print("Document ingested")
    return True

if __name__ == "__main__":
    target_file = input("Enter the path to your PDF (e.g., sample.pdf): ").strip()
    
    if target_file.startswith('"') and target_file.endswith('"'):
        target_file = target_file[1:-1]
        
    process_and_store_pdf(target_file)