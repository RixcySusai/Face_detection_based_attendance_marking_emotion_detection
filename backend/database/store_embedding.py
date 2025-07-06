import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from backend.database.database import db

# Collection to store embeddings
collection = db["face_embeddings"]

# Load Face Recognition Model
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Image folder (Update path if needed)
image_folder = r"C:\Users\Rixcy.S.NuStartz\OneDrive\Documents\images"

def store_embeddings():
    """Extracts and stores face embeddings in MongoDB."""
    for filename in os.listdir(image_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"❌ Could not read image: {filename}")
                continue

            faces = app.get(img)
            if faces:
                embedding = faces[0].normed_embedding.tolist()

                # Store in MongoDB (with debug insert)
                student_data = {
                    "name": filename.split(".")[0],
                    "embedding": embedding
                }

                try:
                    result = collection.insert_one(student_data)
                    if result.acknowledged:
                        print(f"✅ Stored embedding for {filename}")
                    else:
                        print(f"⚠️ Insert not acknowledged for {filename}")
                except Exception as e:
                    print(f"❌ Error inserting {filename}: {e}")

if __name__ == "__main__":
    store_embeddings()

