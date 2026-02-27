from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)
collection_name = "chat_memory"

# 1. DELETE the old collection (just in case)
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
    print(f"🗑️  Deleted old '{collection_name}' collection.")

# 2. CREATE a new one explicitly locked to 384 dimensions
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

print(f"✅ Success! Created new '{collection_name}' with size 384.")
print("🚀 Now run your main.py - it will work.")