"""
attendance_system.py
---------------------
Main program. Run this every day to take attendance.

Integrates all 5 planned features (three-tier confidence, tiered
beeping, live dashboard, manual guard call, visitor auto-expiry),
plus presentation-ready polish:
  - Runs in FULLSCREEN so it reads as a real standalone application
    rather than a script window.
  - Overlays a small QR code (bottom-right corner) linking to the
    live dashboard, so a guard, teacher, or panelist can scan it
    with their phone and instantly see attendance logs - no manual
    IP typing needed. Requires the 'qrcode' package; if it isn't
    installed, the system runs normally with no overlay rather than
    crashing.

Usage:
    python3 attendance_system.py
Press 'q' to quit, 'g' to call the guard manually, 'f' to toggle
fullscreen on/off if you need windowed mode for debugging.
"""

import cv2
import os
import csv
import uuid
from datetime import datetime, date

from color_utils import get_dominant_shirt_color
from student_data import load_all_labels, cleanup_expired_visitors
from alarm_utils import escalate as escalate_alarm
from qr_utils import overlay_qr_on_frame, get_dashboard_url

ATTENDANCE_DIR = "attendance"
UNREGISTERED_LOG = "attendance/unregistered_log.csv"
REVIEW_QUEUE_FILE = "attendance/review_queue.csv"
GUARD_CALLS_FILE = "attendance/guard_calls.csv"
DAILY_SUMMARY_FILE = "attendance/daily_summary.csv"
SNAPSHOT_DIR = "unregistered_snapshots"
REVIEW_SNAPSHOT_DIR = "review_snapshots"
MODEL_PATH = "data/trainer/trainer.yml"

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
WINDOW_NAME = "SecureCheck"

# --- Feature 1: three-tier confidence ---
# LBPH confidence = distance, LOWER is a BETTER match.
CONFIDENT_THRESHOLD = 75     # below this: confident match, mark attendance automatically
UNCERTAIN_THRESHOLD = 85     # between CONFIDENT and UNCERTAIN: send to guard review queue
                              # above UNCERTAIN: confident no-match -> unregistered flow

CONFIRM_FRAMES = 8           # consecutive frames before acting on unknown/uncertain
POSITION_TOLERANCE = 40      # px tolerance for "same face" between frames
MAX_MISSED_FRAMES = 15       # frames a tracker survives without being re-seen
ALARM_COOLDOWN_SECONDS = 30  # don't re-log/re-alarm the same tracked face too often


def get_today_attendance_path():
    return os.path.join(ATTENDANCE_DIR, f"attendance_{date.today().isoformat()}.csv")


def load_already_marked(path):
    marked = set()
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                marked.add(row["student_id"])
    return marked


def mark_attendance(path, person_id, name, person_type, shirt_color):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["student_id", "name", "type", "time", "shirt_color"])
        writer.writerow([person_id, name, person_type, datetime.now().strftime("%H:%M:%S"), shirt_color])


def append_csv_row(path, header, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def mark_day_had_unregistered():
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    today_str = date.today().isoformat()
    rows = []
    found = False

    if os.path.exists(DAILY_SUMMARY_FILE):
        with open(DAILY_SUMMARY_FILE, "r", newline="") as f:
            for row in csv.DictReader(f):
                if row["date"] == today_str:
                    row["had_unregistered_student"] = "YES"
                    found = True
                rows.append(row)

    if not found:
        rows.append({"date": today_str, "had_unregistered_student": "YES"})

    with open(DAILY_SUMMARY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "had_unregistered_student"])
        writer.writeheader()
        writer.writerows(rows)


def log_unregistered_event(snapshot_path):
    append_csv_row(UNREGISTERED_LOG, ["date", "time", "snapshot"],
                    [date.today().isoformat(), datetime.now().strftime("%H:%M:%S"), snapshot_path])
    mark_day_had_unregistered()


def log_review_entry(snapshot_path, guessed_id, guessed_name, confidence):
    entry_id = uuid.uuid4().hex[:10]
    append_csv_row(
        REVIEW_QUEUE_FILE,
        ["entry_id", "date", "time", "guessed_id", "guessed_name", "confidence", "snapshot", "status"],
        [entry_id, date.today().isoformat(), datetime.now().strftime("%H:%M:%S"),
         guessed_id, guessed_name, f"{confidence:.1f}", snapshot_path, "pending"],
    )


def log_guard_call(source="camera_station", note=""):
    append_csv_row(GUARD_CALLS_FILE, ["date", "time", "source", "note"],
                    [date.today().isoformat(), datetime.now().strftime("%H:%M:%S"), source, note])
    print("*** Guard called manually. Logged to guard_calls.csv ***")


def find_matching_tracker(trackers, box):
    x, y, w, h = box
    for tracker in trackers:
        tx, ty, tw, th = tracker["box"]
        if abs(x - tx) < POSITION_TOLERANCE and abs(y - ty) < POSITION_TOLERANCE:
            return tracker
    return None


def setup_display_window(fullscreen=True):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def run():
    for visitor_id, name in cleanup_expired_visitors():
        print(f"Visitor access expired and removed: {name} (ID {visitor_id})")

    labels = load_all_labels()
    if not labels:
        print("No one enrolled yet. Run enroll_student.py (and/or visitor_enroll.py) first.")
        return

    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Run train_recognizer.py after enrolling.")
        return

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(REVIEW_SNAPSHOT_DIR, exist_ok=True)

    attendance_path = get_today_attendance_path()
    already_marked = load_already_marked(attendance_path)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open camera.")
        return

    unknown_trackers = []
    uncertain_trackers = []
    is_fullscreen = True
    setup_display_window(fullscreen=is_fullscreen)

    print("Attendance system running.")
    print(f"Live dashboard: {get_dashboard_url()}  (scan the on-screen QR code with your phone)")
    print("Press 'q' to quit, 'g' to call the guard manually, 'f' to toggle fullscreen.\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Camera read failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        seen_unknown_boxes = []
        seen_uncertain_boxes = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            person_id, confidence = recognizer.predict(face_roi)
            person_id = str(person_id)
            info = labels.get(person_id)

            if confidence < CONFIDENT_THRESHOLD and info is not None:
                name = info["name"]
                person_type = info["type"]
                shirt_color = get_dominant_shirt_color(frame, (x, y, w, h)) or "Unknown"

                if person_id not in already_marked:
                    mark_attendance(attendance_path, person_id, name, person_type, shirt_color)
                    already_marked.add(person_id)
                    print(f"Attendance marked: {name} ({person_type}, ID {person_id}) - Shirt: {shirt_color}")

                color = (0, 255, 0)
                label = f"{name} ({shirt_color})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elif confidence < UNCERTAIN_THRESHOLD and info is not None:
                color = (0, 200, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"Unsure: {info['name']}?", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                seen_uncertain_boxes.append(((x, y, w, h), person_id, info["name"], confidence))

            else:
                color = (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, "Unregistered?", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                seen_unknown_boxes.append((x, y, w, h))

        updated_unknown = []
        for box in seen_unknown_boxes:
            tracker = find_matching_tracker(unknown_trackers, box)
            if tracker:
                tracker["box"] = box
                tracker["frames_seen"] += 1
                tracker["missed_frames"] = 0
            else:
                tracker = {"box": box, "frames_seen": 1, "missed_frames": 0,
                           "alarm_count": 0, "last_alarm": None}
            updated_unknown.append(tracker)

            if tracker["frames_seen"] >= CONFIRM_FRAMES:
                now = datetime.now()
                can_alarm = (tracker["last_alarm"] is None
                             or (now - tracker["last_alarm"]).total_seconds() > ALARM_COOLDOWN_SECONDS)
                if can_alarm:
                    escalate_alarm(tracker["alarm_count"])
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    snapshot_path = os.path.join(SNAPSHOT_DIR, f"unregistered_{timestamp}.jpg")
                    cv2.imwrite(snapshot_path, frame)
                    log_unregistered_event(snapshot_path)
                    tracker["last_alarm"] = now
                    tracker["alarm_count"] += 1
                    tracker["frames_seen"] = 0

        for tracker in unknown_trackers:
            if tracker not in updated_unknown:
                tracker["missed_frames"] += 1
                if tracker["missed_frames"] <= MAX_MISSED_FRAMES:
                    updated_unknown.append(tracker)
        unknown_trackers = updated_unknown

        updated_uncertain = []
        for box, guessed_id, guessed_name, conf in seen_uncertain_boxes:
            tracker = find_matching_tracker(uncertain_trackers, box)
            if tracker:
                tracker["box"] = box
                tracker["frames_seen"] += 1
                tracker["missed_frames"] = 0
                tracker["confidence"] = conf
            else:
                tracker = {"box": box, "frames_seen": 1, "missed_frames": 0,
                           "guessed_id": guessed_id, "guessed_name": guessed_name,
                           "confidence": conf, "last_logged": None}
            updated_uncertain.append(tracker)

            if tracker["frames_seen"] >= CONFIRM_FRAMES:
                now = datetime.now()
                can_log = (tracker["last_logged"] is None
                           or (now - tracker["last_logged"]).total_seconds() > ALARM_COOLDOWN_SECONDS)
                if can_log:
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    snapshot_path = os.path.join(REVIEW_SNAPSHOT_DIR, f"review_{timestamp}.jpg")
                    cv2.imwrite(snapshot_path, frame)
                    log_review_entry(snapshot_path, tracker["guessed_id"], tracker["guessed_name"], tracker["confidence"])
                    print(f"Sent to guard review queue: possibly {tracker['guessed_name']} (unsure)")
                    tracker["last_logged"] = now
                    tracker["frames_seen"] = 0

        for tracker in uncertain_trackers:
            if tracker not in updated_uncertain:
                tracker["missed_frames"] += 1
                if tracker["missed_frames"] <= MAX_MISSED_FRAMES:
                    updated_uncertain.append(tracker)
        uncertain_trackers = updated_uncertain

        cv2.putText(frame, f"Present today: {len(already_marked)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        frame = overlay_qr_on_frame(frame)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("g"):
            log_guard_call(source="camera_station_keypress")
        elif key == ord("f"):
            is_fullscreen = not is_fullscreen
            setup_display_window(fullscreen=is_fullscreen)

    cam.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. {len(already_marked)} student(s)/visitor(s) marked present today.")
    print(f"Attendance file: {attendance_path}")


if __name__ == "__main__":
    run()
