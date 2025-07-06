from backend.database.database import db
import numpy as np

collection = db["face_embeddings"]

def get_all_embeddings():
    collection = db["face_embeddings"]  # ✅ Make sure this matches your insert code
    embeddings = []

    for doc in collection.find():
        name = doc.get("name", "")
        embedding = doc.get("embedding", [])
        if name and embedding:
            embeddings.append((name, embedding))
    return embeddings

# Test retrieval
if __name__ == "__main__":
    embeddings = get_all_embeddings()
    for name, embedding in embeddings:
        print(f"🧑 Name: {name}, Embedding Length: {len(embedding)}")
