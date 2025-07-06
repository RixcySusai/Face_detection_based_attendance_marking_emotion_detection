from pymongo import MongoClient

def get_database():
    """Connects to MongoDB and returns the database instance."""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["Student_image"]  # Ensure this matches your MongoDB database name
    return db

# Connect to database
db = get_database()

# Define the collection for embeddings
collection = db["face_embeddings"]
print("📦 Stored Embeddings:")
for doc in collection.find():
    name = doc.get("name", "No Name")
    embedding = doc.get("embedding", [])
    print(f"Name: {name}, Embedding Length: {len(embedding)}")
