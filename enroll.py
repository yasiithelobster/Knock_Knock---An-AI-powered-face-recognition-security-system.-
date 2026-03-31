import os
import pickle
import face_recognition

KNOWN_FACES_DIR = "data/known_faces"
ENCODINGS_FILE = "data/encodings.pkl"


def load_existing_data():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as file:
            return pickle.load(file)
    return {"names": [], "encodings": []}


def save_data(data):
    with open(ENCODINGS_FILE, "wb") as file:
        pickle.dump(data, file)


def enroll_face():
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    person_name = input("Enter person's name: ").strip()
    image_path = input("Enter image path: ").strip().strip('"').strip("'")

    if not os.path.exists(image_path):
        print("Image file not found.")
        return

    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        print("No face found in the image.")
        return

    if len(face_locations) > 1:
        print("More than one face found. Please use an image with only one face.")
        return

    face_encoding = face_recognition.face_encodings(image, face_locations)[0]

    data = load_existing_data()
    data["names"].append(person_name)
    data["encodings"].append(face_encoding)
    save_data(data)

    print(f"{person_name} enrolled successfully.")


if __name__ == "__main__":
    enroll_face()