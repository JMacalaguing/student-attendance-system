"""
visitor_enroll.py
-------------------
Feature 5: Visitor / parent temporary enrollment with auto-expiry.

Registers someone (a parent, a visiting teacher, etc.) who needs
temporary face recognition access — for example so they don't
trigger the "unregistered student" alarm while picking up a child.
Their face data and access are AUTOMATICALLY DELETED 24 hours after
enrollment (data minimization: we don't keep biometric data on
non-students any longer than needed for it to be useful).

Visitor IDs are auto-assigned starting at 9000 so they never collide
with teacher-assigned student IDs.

Usage:
    python visitor_enroll.py
"""

import importlib


try:
    cv2 = importlib.import_module("cv2")
except ImportError as exc:
    raise ImportError(
        "OpenCV is required. Install it with: python -m pip install opencv-python"
    ) from exc
import os

from student_data import register_visitor

NUM_SAMPLES = 20  # fewer than students - this is a short-lived enrollment
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def enroll_visitor():
    name = input("Visitor's full name: ").strip()
    note = input("Reason / relation (e.g. 'parent of Juan dela Cruz'): ").strip()
    hours_input = input("Hours of access before auto-expiry [default 24]: ").strip()
    hours_valid = int(hours_input) if hours_input.isdigit() else 24

    if not name:
        print("Name is required. Aborting.")
        return

    visitor_id, visitor_folder = register_visitor(name, note, hours_valid)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Could not open camera. Check your webcam connection.")
        return

    print(f"\nEnrolling visitor '{name}' (temp ID {visitor_id}).")
    print(f"This access will auto-expire in {hours_valid} hour(s).")
    print("Look at the camera. Press 'q' to cancel early.\n")

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
            filename = os.path.join(visitor_folder, f"{count}.jpg")
            cv2.imwrite(filename, face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, f"Captured {count}/{NUM_SAMPLES}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            break

        cv2.imshow("Enrolling visitor - press q to cancel", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Enrollment cancelled by user.")
            break

    cam.release()
    cv2.destroyAllWindows()

    if count >= NUM_SAMPLES:
        print(f"\nDone. Visitor '{name}' enrolled with temp ID {visitor_id}.")
        print(f"Access expires automatically at enrollment time + {hours_valid}h.")
        print("Next step: run train_recognizer.py to update the face model.")
    else:
        print(f"\nOnly captured {count}/{NUM_SAMPLES} images. Re-run this script to add more.")


if __name__ == "__main__":
    enroll_visitor()
