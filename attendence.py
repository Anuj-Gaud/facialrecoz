import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle
import sqlite3

# --- CONFIG ---
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DB_PATH       = os.path.join(BASE_DIR, 'attendance.db')
DATASET_DIR   = os.path.join(BASE_DIR, 'dataset_images')
ENCODING_FILE = os.path.join(BASE_DIR, 'known_face_encodings.pkl')
RELOAD_FLAG   = os.path.join(BASE_DIR, 'reload_encodings.flag')

# --- 1. DATABASE SETUP ---
def setup_database():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            UNIQUE(name, date)
        )
    ''')
    conn.commit()
    conn.close()
    print("Database ready.")

# --- 2. LOAD ENCODINGS ---
def load_encodings_from_dataset():
    images, names = [], []
    for cl in os.listdir(DATASET_DIR):
        if cl.startswith('.'): continue
        person_dir = os.path.join(DATASET_DIR, cl)
        if not os.path.isdir(person_dir): continue
        files = [f for f in os.listdir(person_dir) if not f.startswith('.')]
        if not files: continue
        img = cv2.imread(os.path.join(person_dir, files[0]))
        if img is not None:
            images.append(img)
            names.append(cl)

    encode_list = []
    valid_names  = []  # keep names in sync with encodings
    for img, name in zip(images, names):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(img_rgb)
        if encs:
            encode_list.append(encs[0])
            valid_names.append(name)
        else:
            print(f"Warning: No face found for '{name}', skipping.")
    return encode_list, valid_names

def load_encodings():
    if os.path.exists(ENCODING_FILE):
        print("Loading encodings from file...")
        with open(ENCODING_FILE, 'rb') as f:
            return pickle.load(f)
    else:
        print("Encoding faces from dataset...")
        enc, names = load_encodings_from_dataset()
        with open(ENCODING_FILE, 'wb') as f:
            pickle.dump((enc, names), f)
        return enc, names

# --- 3. MARK ATTENDANCE ---
def mark_attendance(name):
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now()
    try:
        conn.execute(
            "INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
            (name, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'))
        )
        conn.commit()
        print(f"Marked: {name}")
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()

# --- 4. MAIN LOOP ---
if __name__ == '__main__':
    setup_database()
    known_encodings, class_names = load_encodings()
    print(f"Encoding complete. {len(known_encodings)} faces loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera. Check that a webcam is connected and not in use by another program.")
        exit(1)

    frame_count = 0

    while True:
        success, img = cap.read()
        if not success:
            print("WARNING: Failed to grab frame from camera.")
            break

        frame_count += 1

        # Hot-reload check every 30 frames
        if frame_count % 30 == 0 and os.path.exists(RELOAD_FLAG):
            print("Reloading encodings (new student added via web app)...")
            known_encodings, class_names = load_encodings()
            print(f"Reloaded: {len(known_encodings)} faces.")
            os.remove(RELOAD_FLAG)

        # Process every 3rd frame only (speed boost)
        if frame_count % 3 != 0:
            cv2.imshow('FaceAttend — Press Q to quit', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faces_cur   = face_recognition.face_locations(imgS)
        encodes_cur = face_recognition.face_encodings(imgS, faces_cur)

        for encode_face, face_loc in zip(encodes_cur, faces_cur):
            if not known_encodings:
                break
            matches   = face_recognition.compare_faces(known_encodings, encode_face, tolerance=0.50)
            face_dis  = face_recognition.face_distance(known_encodings, encode_face)
            match_idx = np.argmin(face_dis)

            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            if matches[match_idx]:
                name       = class_names[match_idx].upper()
                confidence = int((1 - face_dis[match_idx]) * 100)
                label      = f"{name}  {confidence}%"
                color      = (0, 220, 100)
                mark_attendance(name)
            else:
                label = 'UNKNOWN'
                color = (0, 0, 220)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 38), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, label, (x1 + 6, y2 - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1)

        cv2.imshow('FaceAttend — Press Q to quit', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()