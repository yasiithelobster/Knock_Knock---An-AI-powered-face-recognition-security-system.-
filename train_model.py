import os
import json
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATASET_DIR = "dataset"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

MODEL_PATH = os.path.join(MODEL_DIR, "face_classifier.keras")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

IMAGE_SIZE = (224, 224)
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 25
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

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(256, activation="relu"),
        Dropout(0.5),

        Dense(128, activation="relu"),
        Dropout(0.3),

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


def plot_training_history(history):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_accuracy.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, label_map):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    class_names = [label_map[i] for i in sorted(label_map.keys())]
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()


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

    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.1,
        brightness_range=(0.9, 1.1),
        horizontal_flip=False
    )

    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_generator = val_datagen.flow(
        X_val,
        y_val,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = build_model(num_classes=len(label_map))

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    print("\nTraining model...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr]
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    save_label_map(label_map)

    print("\nEvaluating model...")
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

    predictions = model.predict(X_val, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    print("\nClassification Report:")
    target_names = [label_map[i] for i in sorted(label_map.keys())]
    print(classification_report(y_val, y_pred, target_names=target_names))

    plot_training_history(history)
    plot_confusion_matrix(y_val, y_pred, label_map)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Label map saved to: {LABEL_MAP_PATH}")
    print(f"Plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()