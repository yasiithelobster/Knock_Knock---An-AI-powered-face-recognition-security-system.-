import cv2
import os
import pickle
import face_recognition

ENCODINGS_FILE = "data/encodings.pkl"


def load_existing_data():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as file:
            return pickle.load(file)
    return {"names": [], "encodings": []}


def save_data(data):
    with open(ENCODINGS_FILE, "wb") as file:
        pickle.dump(data, file)


def enroll_from_webcam():
    name = input("Enter person's name: ").strip()

    video = cv2.VideoCapture(0)

    print("Press SPACE to capture face")
    print("Press Q to quit")

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to access camera")
            break

        cv2.imshow("Enrollment - Knock Knock!", frame)

        key = cv2.waitKey(1)

        if key == ord(" "):  # SPACE to capture
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)

            if len(face_locations) != 1:
                print("Make sure exactly one face is visible.")
                continue

            encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

            data = load_existing_data()
            data["names"].append(name)
            data["encodings"].append(encoding)
            save_data(data)

            print(f"{name} enrolled successfully!")
            break

        elif key == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    enroll_from_webcam()