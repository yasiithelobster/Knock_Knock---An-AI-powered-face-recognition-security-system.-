import os
import json
import numpy as np
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_classifier.keras")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

IMAGE_SIZE = (224, 224)
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 15
BATCH_SIZE = 16


def load_dataset():
    images = []
    labels = []
    label_map = {}
    current_label = 0

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    person_names = sorted(
        [
            name for name in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, name))
        ]
    )

    if not person_names:
        raise ValueError("No person folders found inside dataset.")

    for person_name in person_names:
        person_folder = os.path.join(DATASET_DIR, person_name)
        image_files = sorted(
            [
                file for file in os.listdir(person_folder)
                if file.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        if not image_files:
            continue

        label_map[current_label] = person_name

        for image_file in image_files:
            image_path = os.path.join(person_folder, image_file)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            image = cv2.resize(image, IMAGE_SIZE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float32") / 255.0

            images.append(image)
            labels.append(current_label)

        current_label += 1

    if not images:
        raise ValueError("No valid images found in dataset.")

    return np.array(images), np.array(labels), label_map


def build_model(num_classes):
    model = Sequential([
        Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation="relu"),
        Dropout(0.5),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_label_map(label_map):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(LABEL_MAP_PATH, "w") as file:
        json.dump(label_map, file, indent=4)


def main():
    print("Loading dataset...")
    X, y, label_map = load_dataset()

    print(f"Total images loaded: {len(X)}")
    print(f"Classes found: {len(label_map)}")
    print("Label map:")
    for label, name in label_map.items():
        print(f"  {label} -> {name}")

    if len(label_map) < 2:
        print("You need at least 2 different people to train a classifier.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    model = build_model(num_classes=len(label_map))

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    print("\nTraining model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping]
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    save_label_map(label_map)

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    print("\nTraining complete.")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Label map saved to: {LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()