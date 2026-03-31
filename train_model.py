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
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2


DATASET_DIR = "dataset"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

MODEL_PATH = os.path.join(MODEL_DIR, "face_classifier.keras")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

IMAGE_SIZE = (128, 128)
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 30
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

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = gray.astype("float32") / 255.0

            gray = np.expand_dims(gray, axis=-1)

            images.append(gray)
            labels.append(current_label)

        current_label += 1

    if not images:
        raise ValueError("No valid images found in dataset.")

    return np.array(images), np.array(labels), label_map


def build_model(num_classes):
    model = Sequential([
        Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),

        Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        GlobalAveragePooling2D(),

        Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.4),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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


def plot_confusion(y_true, y_pred, label_map):
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


def print_confusion_pairs(y_true, y_pred, label_map):
    cm = confusion_matrix(y_true, y_pred)
    class_names = [label_map[i] for i in sorted(label_map.keys())]

    print("\nMain Misclassification Clues:")
    found_any = False

    for true_idx in range(len(class_names)):
        for pred_idx in range(len(class_names)):
            if true_idx != pred_idx and cm[true_idx][pred_idx] > 0:
                print(
                    f"  True: {class_names[true_idx]} -> Predicted: {class_names[pred_idx]} "
                    f"({cm[true_idx][pred_idx]} samples)"
                )
                found_any = True

    if not found_any:
        print("  No cross-person confusion found in validation predictions.")


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
        rotation_range=8,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.08,
        brightness_range=(0.95, 1.05)
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
        patience=6,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
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

    target_names = [label_map[i] for i in sorted(label_map.keys())]

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=target_names))

    print_confusion_pairs(y_val, y_pred, label_map)
    plot_training_history(history)
    plot_confusion(y_val, y_pred, label_map)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Label map saved to: {LABEL_MAP_PATH}")
    print(f"Plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()