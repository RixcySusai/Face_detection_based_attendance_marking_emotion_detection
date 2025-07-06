import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Load model
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def extract_embedding(image_path):
    """Extracts face embedding from an image."""
    img = cv2.imread(image_path)
    faces = app.get(img)

    if faces:
        return np.array(faces[0].normed_embedding)
    return None

# Test extraction
if __name__ == "__main__":
    test_path = "images/test_person.jpg"
    embedding = extract_embedding(test_path)
    if embedding is not None:
        print(f"✅ Extracted embedding for {test_path}")
    else:
        print("❌ No face detected!")
