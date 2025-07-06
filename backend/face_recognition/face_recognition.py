
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from backend.database.retrieve_embedding import get_all_embeddings
from backend.database.database import db
from datetime import datetime
import csv
import os
import pyttsx3
from deepface import DeepFace
import re

# Initialize emotion counters
emotion_count = {
    'neutral': 0,
    'happy': 0,
    'surprise': 0,
    'sad': 0,
    'angry': 0,
    'fear': 0,
    'disgust': 0,
    'contempt': 0,
    'distracted': 0,  # Assuming this means non-attentive students
}

# Load Face Recognition Model
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

THRESHOLD = 0.6

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(name):
    engine.say(f"{name} is present")
    engine.runAndWait()

def load_attendance(file_path="attendanc_log.csv"):
    attendance_data = []
    if os.path.exists(file_path):
        with open(file_path, mode='r', newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                attendance_data.append(row)
    return attendance_data

def mark_attendance(name, reg_no):
    today = datetime.now().date().isoformat()
    existing = db["attendance_records"].find_one({
        "reg_no": reg_no,
        "timestamp": {"$regex": f"^{today}"}
    })

    if not existing:
        db["attendance_records"].insert_one({
            "reg_no": reg_no,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "status": "Present"
        })
        print(f"✅ Attendance marked for {name} ({reg_no})")
        speak(name)
        
    else:
        if reg_no:
            print(f"ℹ️ Already marked for today: {name} ({reg_no})")
        else:
            print(f"ℹ️ Already marked for today: {name}")



def parse_name_regno(filename):
    base = os.path.splitext(filename)[0]
    base = re.sub(r"\(.*\)", "", base)  # Remove anything in parentheses
    base = base.replace("Unknown", "").strip()
    parts = base.split("-")
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return parts[0].strip(), ""





def recognize_face():
    """Recognizes a face from the webcam and detects emotion."""
    cap = cv2.VideoCapture(0)

    # Emotion counter
    global emotion_count
    emotion_count = {
        'happy': 0,
        'neutral': 0,
        'surprise': 0,
        'sad': 0,
        'angry': 0,
        'fear': 0,
        'disgust': 0,
        'contempt': 0,
        'distracted': 0
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        if faces:
            query_embedding = np.array(faces[0].normed_embedding).reshape(1, -1)

            best_match = None
            best_reg_no = None
            highest_similarity = 0.0

            # Get all embeddings from the database
            for file_name, stored_embedding in get_all_embeddings():
                name, reg_no = parse_name_regno(file_name)

                similarity = cosine_similarity(query_embedding, np.array(stored_embedding).reshape(1, -1))[0][0]

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = name
                    best_reg_no = reg_no

            # Display result
            if highest_similarity > THRESHOLD:
                if best_reg_no:
                    print(f"✅ Match Found: {best_match} ({best_reg_no}) | Similarity: {highest_similarity:.2f}")
                else:
                    print(f"✅ Match Found: {best_match} | Similarity: {highest_similarity:.2f}")

                mark_attendance(best_match, best_reg_no)
                                
                # ✅ Emotion Detection Block
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                    print(f"🧠 Detected Emotion for {best_match}: {emotion}")

                    # Count emotion
                    if emotion in emotion_count:
                        emotion_count[emotion] += 1

                    # Live attentiveness feedback
                    attentive_emotions = ['neutral', 'happy', 'surprise']
                    if emotion in attentive_emotions:
                        print("🎧 Student is likely attentive.")
                    else:
                        print("😴 Student may be distracted.")
                        emotion_count['distracted'] += 1

                except Exception as e:
                    print("⚠️ Emotion detection failed:", e)

            else:
                print("❌ No Match Found")

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ✅ Emotion Summary at End
    print("\n📊 Emotion Analysis Summary:")
    for emotion, count in emotion_count.items():
        print(f"{emotion.capitalize()}: {count}")


# Run the face recognition and attendance marking
recognize_face()
