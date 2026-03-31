import os
import cv2
import pickle
import numpy as np
import face_recognition
from datetime import datetime


DATABASE_PATH = "models/embedding_database.pkl"
LOG_FILE = "logs/access_log.txt"

MATCH_THRESHOLD = 0.50
CONSISTENT_FRAMES_REQUIRED = 3


def load_embedding_database():
    if not os.path.exists(DATABASE_PATH):
        raise FileNotFoundError(f"Embedding database not found: {DATABASE_PATH}")

    with open(DATABASE_PATH, "rb") as file:
        return pickle.load(file)


def log_access(name, status, distance):
    os.makedirs("logs", exist_ok=True)

    with open(LOG_FILE, "a") as file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(
            f"{current_time} - {name} - {status} - Distance: {distance:.4f}\n"
        )


def find_best_match(live_embedding, database):
    best_name = "Unknown"
    best_distance = float("inf")

    for person_name, person_data in database.items():
        mean_embedding = person_data["mean_embedding"]
        distance = np.linalg.norm(live_embedding - mean_embedding)

        if distance < best_distance:
            best_distance = distance
            best_name = person_name

    if best_distance < MATCH_THRESHOLD:
        return best_name, "Access Granted", best_distance

    return "Unknown", "Access Denied", best_distance


def main():
    print("Loading embedding database...")
    database = load_embedding_database()

    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Could not access webcam.")
        return

    print("Knock Knock! Embedding-Based Security System Started")
    print("Press 'q' to quit.")

    stable_name = None
    stable_status = None
    stable_count = 0

    last_logged_name = None
    last_logged_status = None

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to capture frame.")
            break

        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_locations) == 1 and len(face_encodings) == 1:
            (top, right, bottom, left) = face_locations[0]
            live_embedding = face_encodings[0]

            name, status, distance = find_best_match(live_embedding, database)

            print(f"Predicted: {name}, Status: {status}, Distance: {distance:.4f}")

            if name == stable_name and status == stable_status:
                stable_count += 1
            else:
                stable_name = name
                stable_status = status
                stable_count = 1

            if stable_count >= CONSISTENT_FRAMES_REQUIRED:
                display_name = name
                display_status = status
            else:
                if status == "Access Granted":
                    display_name = "Verifying..."
                    display_status = "Hold Still"
                else:
                    display_name = "Unknown"
                    display_status = "Access Denied"

            color = (0, 255, 0) if display_status == "Access Granted" else (0, 0, 255)
            if display_status == "Hold Still":
                color = (0, 255, 255)

            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(display_frame, (left, bottom - 90), (right, bottom), color, cv2.FILLED)

            cv2.putText(
                display_frame,
                display_name,
                (left + 8, bottom - 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                display_status,
                (left + 8, bottom - 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Dist: {distance:.3f}",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            if stable_count >= CONSISTENT_FRAMES_REQUIRED:
                if name != last_logged_name or status != last_logged_status:
                    log_access(name, status, distance)
                    last_logged_name = name
                    last_logged_status = status

        elif len(face_locations) > 1:
            stable_name = None
            stable_status = None
            stable_count = 0

            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.putText(
                display_frame,
                "Multiple faces detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        else:
            stable_name = None
            stable_status = None
            stable_count = 0

            cv2.putText(
                display_frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("Knock Knock! Security Feed", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()