import os
import pickle
import numpy as np
import face_recognition


DATASET_DIR = "dataset"
MODEL_DIR = "models"
DATABASE_PATH = os.path.join(MODEL_DIR, "embedding_database.pkl")


def get_person_folders():
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    person_folders = [
        name for name in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, name))
    ]

    if not person_folders:
        raise ValueError("No person folders found inside dataset.")

    return sorted(person_folders)


def process_person_images(person_name):
    person_folder = os.path.join(DATASET_DIR, person_name)

    image_files = sorted(
        [
            file for file in os.listdir(person_folder)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    if not image_files:
        print(f"[WARNING] No images found for {person_name}")
        return []

    embeddings = []

    for image_file in image_files:
        image_path = os.path.join(person_folder, image_file)

        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
        except Exception as error:
            print(f"[SKIPPED] Could not read {image_path}: {error}")
            continue

        if len(face_locations) == 0:
            print(f"[SKIPPED] No face found in {image_path}")
            continue

        if len(face_locations) > 1:
            print(f"[SKIPPED] Multiple faces found in {image_path}")
            continue

        encodings = face_recognition.face_encodings(image, face_locations)

        if not encodings:
            print(f"[SKIPPED] No embedding generated for {image_path}")
            continue

        embeddings.append(encodings[0])
        print(f"[OK] Processed {image_path}")

    return embeddings


def build_embedding_database():
    os.makedirs(MODEL_DIR, exist_ok=True)

    database = {}
    person_folders = get_person_folders()

    print("Building embedding database...\n")

    for person_name in person_folders:
        print(f"Processing person: {person_name}")

        embeddings = process_person_images(person_name)

        if not embeddings:
            print(f"[WARNING] No valid embeddings collected for {person_name}\n")
            continue

        embeddings_array = np.array(embeddings)
        mean_embedding = np.mean(embeddings_array, axis=0)

        database[person_name] = {
            "embeddings": embeddings_array,
            "mean_embedding": mean_embedding,
            "num_samples": len(embeddings)
        }

        print(f"[DONE] {person_name}: {len(embeddings)} embeddings saved\n")

    if not database:
        raise ValueError("No valid embeddings were generated for any person.")

    with open(DATABASE_PATH, "wb") as file:
        pickle.dump(database, file)

    print("Embedding database saved successfully.")
    print(f"Saved to: {DATABASE_PATH}")

    print("\nSummary:")
    for person_name, info in database.items():
        print(f"  {person_name}: {info['num_samples']} samples")


if __name__ == "__main__":
    build_embedding_database()