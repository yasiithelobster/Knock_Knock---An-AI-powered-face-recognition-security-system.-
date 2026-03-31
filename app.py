import os
import json
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime


MODEL_PATH = "models/face_classifier.keras"
LABEL_MAP_PATH = "models/label_map.json"
LOG_FILE = "logs/access_log.txt"

IMAGE_SIZE = (128, 128)

CONFIDENCE_THRESHOLD = 0.6
MARGIN_THRESHOLD = 0.08
CONSISTENT_FRAMES_REQUIRED = 3


def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABEL_MAP_PATH, "r") as file:
        label_map = json.load(file)

    label_map = {int(k): v for k, v in label_map.items()}
    return model, label_map


def log_access(name, status, confidence):
    os.makedirs("logs", exist_ok=True)

    with open(LOG_FILE, "a") as file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(
            f"{current_time} - {name} - {status} - Confidence: {confidence:.4f}\n"
        )


def preprocess_face(face_bgr):
    face_resized = cv2.resize(face_bgr, IMAGE_SIZE)
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    face_normalized = face_gray.astype("float32") / 255.0
    face_input = np.expand_dims(face_normalized, axis=-1)
    face_input = np.expand_dims(face_input, axis=0)
    return face_input


def predict_identity(model, label_map, face_crop):
    face_input = preprocess_face(face_crop)
    predictions = model.predict(face_input, verbose=0)[0]

    sorted_indices = np.argsort(predictions)[::-1]
    best_index = int(sorted_indices[0])
    best_confidence = float(predictions[best_index])

    if len(sorted_indices) > 1:
        second_index = int(sorted_indices[1])
        second_confidence = float(predictions[second_index])
    else:
        second_index = best_index
        second_confidence = 0.0

    confidence_margin = best_confidence - second_confidence

    if (
        best_confidence >= CONFIDENCE_THRESHOLD
        and confidence_margin >= MARGIN_THRESHOLD
    ):
        predicted_name = label_map.get(best_index, "Unknown")
        access_status = "Access Granted"
    else:
        predicted_name = "Unknown"
        access_status = "Access Denied"

    return {
        "name": predicted_name,
        "status": access_status,
        "best_confidence": best_confidence,
        "second_confidence": second_confidence,
        "margin": confidence_margin,
    }


def main():
    print("Loading trained model...")
    model, label_map = load_model_and_labels()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Could not access webcam.")
        return

    print("Knock Knock! Security System Started")
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )

        current_name = None
        current_status = None
        current_confidence = 0.0
        current_margin = 0.0

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            face_crop = frame[y:y + h, x:x + w]

            if face_crop.size != 0:
                result = predict_identity(model, label_map, face_crop)

                print(
                    f"Predicted: {result['name']}, "
                    f"Best: {result['best_confidence']:.2f}, "
                    f"Second: {result['second_confidence']:.2f}, "
                    f"Margin: {result['margin']:.2f}"
                )

                current_name = result["name"]
                current_status = result["status"]
                current_confidence = result["best_confidence"]
                current_margin = result["margin"]

                if current_name == stable_name and current_status == stable_status:
                    stable_count += 1
                else:
                    stable_name = current_name
                    stable_status = current_status
                    stable_count = 1

                if (
                    stable_count >= CONSISTENT_FRAMES_REQUIRED
                    and current_status == "Access Granted"
                ):
                    display_name = current_name
                    display_status = "Access Granted"
                    color = (0, 255, 0)
                else:
                    if current_status == "Access Granted":
                        display_name = "Verifying..."
                        display_status = "Hold Still"
                        color = (0, 255, 255)
                    else:
                        display_name = "Unknown"
                        display_status = "Access Denied"
                        color = (0, 0, 255)

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(display_frame, (x, y + h - 90), (x + w, y + h), color, cv2.FILLED)

                cv2.putText(
                    display_frame,
                    display_name,
                    (x + 8, y + h - 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display_frame,
                    display_status,
                    (x + 8, y + h - 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Conf: {current_confidence:.2f}",
                    (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Margin: {current_margin:.2f}",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                if stable_count >= CONSISTENT_FRAMES_REQUIRED:
                    final_name = current_name
                    final_status = current_status
                else:
                    final_name = None
                    final_status = None

                if final_name is not None and final_status is not None:
                    if (
                        final_name != last_logged_name
                        or final_status != last_logged_status
                    ):
                        log_access(final_name, final_status, current_confidence)
                        last_logged_name = final_name
                        last_logged_status = final_status

        elif len(faces) > 1:
            stable_name = None
            stable_status = None
            stable_count = 0

            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

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