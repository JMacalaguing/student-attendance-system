"""
train_recognizer.py
--------------------
Trains the LBPH face recognizer using images from BOTH:
  - enrolled students (data/student_images/), and
  - currently non-expired visitors (data/visitor_images/)

Expired visitors are cleaned up (photos + label entry deleted)
automatically before training, so their face data never lingers in
the trained model past its expiry.

Run this after enrolling a student or visitor, and re-run it any
time enrollment changes.

Usage:
    python train_recognizer.py
"""

import cv2  # type: ignore[reportMissingImports]
import os
import numpy as np

from student_data import (
    STUDENT_IMAGES_DIR,
    VISITOR_IMAGES_DIR,
    load_student_labels,
    load_visitor_labels,
    cleanup_expired_visitors,
)

MODEL_PATH = "data/trainer/trainer.yml"


def collect_samples(images_dir, ids_and_names):
    samples = []
    ids = []
    for person_id in ids_and_names:
        folder = os.path.join(images_dir, str(person_id))
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            samples.append(img)
            ids.append(int(person_id))
    return samples, ids


def train():
    removed = cleanup_expired_visitors()
    for visitor_id, name in removed:
        print(f"Expired visitor removed before training: {name} (ID {visitor_id})")

    student_labels = load_student_labels()
    visitor_labels = load_visitor_labels()

    if not student_labels and not visitor_labels:
        print("No one enrolled yet. Run enroll_student.py or visitor_enroll.py first.")
        return

    student_samples, student_ids = collect_samples(STUDENT_IMAGES_DIR, student_labels.keys())
    visitor_samples, visitor_ids = collect_samples(VISITOR_IMAGES_DIR, visitor_labels.keys())

    face_samples = student_samples + visitor_samples
    ids = student_ids + visitor_ids

    if not face_samples:
        print("No face images found. Enroll students or visitors first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_samples,np.array(ids, dtype=np.int32))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    recognizer.save(MODEL_PATH)

    print(f"Training complete. {len(student_samples)} student image(s) across "
          f"{len(student_labels)} student(s), {len(visitor_samples)} visitor image(s) "
          f"across {len(visitor_labels)} active visitor(s).")
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
