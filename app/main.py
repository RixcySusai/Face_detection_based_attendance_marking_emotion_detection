import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.face_recognition.face_recognition import recognize_face
from backend.database.database import db  # Ensure 'db' is imported properly
from backend.database.store_embedding import store_embeddings


def main():
    while True:
        print("\n📌 Face Recognition System")
        print("1️⃣ Store Face Embeddings")
        print("2️⃣ Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            store_embeddings()

        elif choice == "2":
            print("👋 Exiting...")
            break

        else:
            print("❌ Invalid choice! Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
