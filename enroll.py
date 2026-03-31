import os
import cv2


DATASET_DIR = "dataset"
IMAGE_SIZE = (224, 224)
MAX_IMAGES = 30


def create_person_folder(person_name):
    person_folder = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_folder, exist_ok=True)
    return person_folder


def get_next_image_index(folder_path):
    existing_files = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith(".jpg")
    ]

    if not existing_files:
        return 1

    numbers = []
    for file in existing_files:
        try:
            number = int(os.path.splitext(file)[0])
            numbers.append(number)
        except ValueError:
            continue

    return max(numbers, default=0) + 1


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)

    person_name = input("Enter person's name: ").strip()

    if not person_name:
        print("Name cannot be empty.")
        return

    person_folder = create_person_folder(person_name)
    next_index = get_next_image_index(person_folder)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Could not access webcam.")
        return

    print("\nEnrollment started.")
    print("Rules:")
    print("- Only one face should appear")
    print("- Keep your face front-facing")
    print("- Press SPACE to save an image")
    print("- Press Q to quit\n")

    saved_count = 0

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

        status_text = "No face detected"
        status_color = (0, 0, 255)

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            status_text = "One face detected"
            status_color = (0, 255, 0)

        elif len(faces) > 1:
            status_text = "Multiple faces detected"
            status_color = (0, 0, 255)

            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.putText(
            display_frame,
            f"Saved: {saved_count}/{MAX_IMAGES}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            status_text,
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

        cv2.imshow("Knock Knock! Enrollment", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if len(faces) != 1:
                print("Exactly one front-facing face is required.")
                continue

            (x, y, w, h) = faces[0]

            face_crop = frame[y:y + h, x:x + w]

            if face_crop.size == 0:
                print("Invalid crop. Try again.")
                continue

            resized_face = cv2.resize(face_crop, IMAGE_SIZE)

            file_name = f"{next_index:03d}.jpg"
            save_path = os.path.join(person_folder, file_name)

            cv2.imwrite(save_path, resized_face)

            print(f"Saved: {save_path}")

            saved_count += 1
            next_index += 1

            if saved_count >= MAX_IMAGES:
                print(f"\nEnrollment complete. {MAX_IMAGES} images saved for {person_name}.")
                break

        elif key == ord("q"):
            print("\nEnrollment stopped by user.")
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()