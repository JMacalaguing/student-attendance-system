"""
student_data.py
-----------------
Shared data access for both permanent students and temporary
visitors (feature 5: visitor/parent temp enrollment with auto-expiry).

Permanent students and visitors are kept in SEPARATE label files and
SEPARATE numeric ID ranges so they can never collide:
  - Students: IDs are whatever the teacher assigns (e.g. 1-999)
  - Visitors: IDs auto-assigned starting at 9000

This module is imported by enroll_student.py, visitor_enroll.py,
train_recognizer.py, attendance_system.py, and app.py so all of them
agree on where data lives and what "expired" means.
"""

import os
import json
import shutil
from datetime import datetime, timedelta

DATA_DIR = "data"
STUDENT_LABELS_FILE = os.path.join(DATA_DIR, "student_labels.json")
STUDENT_IMAGES_DIR = os.path.join(DATA_DIR, "student_images")

VISITOR_LABELS_FILE = os.path.join(DATA_DIR, "visitor_labels.json")
VISITOR_IMAGES_DIR = os.path.join(DATA_DIR, "visitor_images")
VISITOR_COUNTER_FILE = os.path.join(DATA_DIR, "visitor_counter.json")
VISITOR_ID_START = 9000
VISITOR_EXPIRY_HOURS = 24


# ---------- Students ----------

def load_student_labels():
    if os.path.exists(STUDENT_LABELS_FILE):
        with open(STUDENT_LABELS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_student_labels(labels):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STUDENT_LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)


def delete_student(student_id):
    """Removes a student's label entry and their enrollment photos."""
    labels = load_student_labels()
    student_id = str(student_id)
    if student_id in labels:
        del labels[student_id]
        save_student_labels(labels)
    folder = os.path.join(STUDENT_IMAGES_DIR, student_id)
    if os.path.isdir(folder):
        shutil.rmtree(folder)


# ---------- Visitors ----------

def load_visitor_labels():
    if os.path.exists(VISITOR_LABELS_FILE):
        with open(VISITOR_LABELS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_visitor_labels(labels):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(VISITOR_LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)


def next_visitor_id():
    """Auto-generates the next visitor ID, starting at VISITOR_ID_START."""
    os.makedirs(DATA_DIR, exist_ok=True)
    counter = VISITOR_ID_START
    if os.path.exists(VISITOR_COUNTER_FILE):
        with open(VISITOR_COUNTER_FILE, "r") as f:
            counter = json.load(f).get("next_id", VISITOR_ID_START)
    with open(VISITOR_COUNTER_FILE, "w") as f:
        json.dump({"next_id": counter + 1}, f)
    return counter


def register_visitor(name, note, hours_valid=VISITOR_EXPIRY_HOURS):
    """Creates a new visitor entry and returns (visitor_id, image_folder)."""
    visitor_id = str(next_visitor_id())
    labels = load_visitor_labels()
    now = datetime.now()
    labels[visitor_id] = {
        "name": name,
        "note": note,
        "enrolled_at": now.isoformat(timespec="seconds"),
        "expires_at": (now + timedelta(hours=hours_valid)).isoformat(timespec="seconds"),
    }
    save_visitor_labels(labels)
    folder = os.path.join(VISITOR_IMAGES_DIR, visitor_id)
    os.makedirs(folder, exist_ok=True)
    return visitor_id, folder


def cleanup_expired_visitors():
    """
    Removes any visitor whose expiry has passed: deletes their photo
    folder and their label entry. Returns the list of removed
    (id, name) pairs. Call this before training and periodically
    while the attendance system / webview is running.
    """
    labels = load_visitor_labels()
    now = datetime.now()
    removed = []

    for visitor_id in list(labels.keys()):
        entry = labels[visitor_id]
        try:
            expires_at = datetime.fromisoformat(entry["expires_at"])
        except (KeyError, ValueError):
            continue

        if now >= expires_at:
            removed.append((visitor_id, entry.get("name", "unknown")))
            folder = os.path.join(VISITOR_IMAGES_DIR, visitor_id)
            if os.path.isdir(folder):
                shutil.rmtree(folder)
            del labels[visitor_id]

    if removed:
        save_visitor_labels(labels)

    return removed


# ---------- Combined lookup ----------

def load_all_labels():
    """
    Returns a dict: id_str -> {"name": str, "type": "student"|"visitor"}
    covering both students and non-expired visitors. Expired visitors
    are cleaned up as a side effect of calling this.
    """
    cleanup_expired_visitors()

    combined = {}
    for sid, name in load_student_labels().items():
        combined[sid] = {"name": name, "type": "student"}
    for vid, entry in load_visitor_labels().items():
        combined[vid] = {"name": entry["name"], "type": "visitor"}
    return combined
