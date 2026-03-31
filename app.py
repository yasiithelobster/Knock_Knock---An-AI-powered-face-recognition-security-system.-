import os
import cv2
import pickle
import face_recognition
import numpy as np
from datetime import datetime

ENCODINGS_FILE = "data/encodings.pkl"
LOG_FILE = "data/access_log.txt"


def load_known_faces():
    if not os.path.exists(ENCODINGS_FILE):
        print("No enrolled faces found. Please run enroll.py first.")
        return {"names": [], "encodings": []}

    with open(ENCODINGS_FILE, "rb") as file:
        return pickle.load(file)


def log_access(name, status):
    with open(LOG_FILE, "a") as file:
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{time} - {name} - {status}\n")


def main():
    data = load_known_faces()

    known_names = data["names"]
    known_encodings = data["encodings"]

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Could not access webcam.")
        return

    print("Starting Knock Knock! Face Recognition System...")
    print("Press 'q' to quit.")

    last_logged_name = None
    last_logged_status = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        current_detected_name = None
        current_detected_status = None

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            access_status = "Access Denied"

            if len(known_encodings) > 0:
                matches = face_recognition.compare_faces(
                    known_encodings,
                    face_encoding,
                    tolerance=0.5
                )
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        access_status = "Access Granted"

            current_detected_name = name
            current_detected_status = access_status

            color = (0, 255, 0) if access_status == "Access Granted" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 50), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1
            )
            cv2.putText(
                frame,
                access_status,
                (left + 6, bottom - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        if current_detected_name is not None and current_detected_status is not None:
            if (
                current_detected_name != last_logged_name
                or current_detected_status != last_logged_status
            ):
                log_access(current_detected_name, current_detected_status)
                last_logged_name = current_detected_name
                last_logged_status = current_detected_status

        cv2.imshow("Knock Knock! Security Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()