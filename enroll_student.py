"""
enroll_student.py
------------------
Registers a new PERMANENT student into the system by capturing face
images from the webcam. For temporary visitors/parents, use
visitor_enroll.py instead (their access auto-expires after 24h).

Usage:
    python enroll_student.py
"""

import importlib
import os

from student_data import (
    STUDENT_IMAGES_DIR,
    load_student_labels,
    save_student_labels,
)

NUM_SAMPLES = 3


def _load_opencv():
    try:
        return importlib.import_module("cv2")
    except ImportError:
        print("OpenCV is not installed. Install it with: python -m pip install opencv-python")
        return None


def enroll():
    cv2 = _load_opencv()
    if cv2 is None:
        return

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    student_id = input("Enter Student ID (number): ").strip()
    student_name = input("Enter Full Name: ").strip()

    if not student_id.isdigit():
        print("Student ID must be a number. Aborting.")
        return

    labels = load_student_labels()
    labels[student_id] = student_name
    save_student_labels(labels)

    student_folder = os.path.join(STUDENT_IMAGES_DIR, student_id)
    os.makedirs(student_folder, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cascade_path)
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Could not open camera. Check your webcam connection.")
        return

    print(f"\nEnrolling '{student_name}' (ID {student_id}).")
    print("Look at the camera. Slowly turn your head slightly left/right.")
    print("Press 'q' to cancel early.\n")

    count = 0
    while count < NUM_SAMPLES:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            filename = os.path.join(student_folder, f"{count}.jpg")
            cv2.imwrite(filename, face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured {count}/{NUM_SAMPLES}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            break  # only capture one face per frame during enrollment

        cv2.imshow("Enrolling - press q to cancel", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Enrollment cancelled by user.")
            break

    cam.release()
    cv2.destroyAllWindows()

    if count >= NUM_SAMPLES:
        print(f"\nDone. Captured {count} images for {student_name} (ID {student_id}).")
        print("Next step: run train_recognizer.py to update the face model.")
    else:
        print(f"\nOnly captured {count}/{NUM_SAMPLES} images. You can re-run this script to add more later.")


if __name__ == "__main__":
    enroll()
