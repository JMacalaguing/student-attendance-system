"""
attendance_system.py
--------------------
Main program. Run this every day to take attendance.

NEW VERSION: face recognition now uses InsightFace (SCRFD detection + ArcFace
recognition) instead of Haar Cascade + LBPH. It compares each face against the
embeddings saved by build_embeddings.py (data/embeddings.npz).

Everything else is the same as before: three-tier confidence (green / yellow / red),
tiered beeping, live dashboard, manual guard call, visitor auto-expiry,
fullscreen window and the QR code overlay.

What is new in the recognition:
  - Similarity score (higher = better match, max 1.0) instead of the LBPH distance.
  - A "margin" check: if the best and second-best student are too close, the face
    goes to the guard (yellow) instead of being marked automatically.
  - Attendance is only marked after the same student is seen in several frames in a row.

Usage:
    python attendance_system.py
Press 'q' to quit, 'g' to call the guard manually, 'f' to toggle fullscreen.
"""

import cv2
import os
import csv
import uuid
import time
import threading
import unicodedata
import numpy as np
from datetime import datetime, date
from insightface.app import FaceAnalysis

from color_utils import get_dominant_shirt_color
from student_data import load_all_labels, cleanup_expired_visitors, DATA_DIR
from alarm_utils import escalate as escalate_alarm
from qr_utils import overlay_qr_on_frame, get_dashboard_url

try:
    import winsound              # built into Windows, no install needed
except ImportError:
    winsound = None              # on the Raspberry Pi the alarm_utils sound is used instead

ATTENDANCE_DIR = "attendance"
UNREGISTERED_LOG = "attendance/unregistered_log.csv"
REVIEW_QUEUE_FILE = "attendance/review_queue.csv"
GUARD_CALLS_FILE = "attendance/guard_calls.csv"
DAILY_SUMMARY_FILE = "attendance/daily_summary.csv"
SNAPSHOT_DIR = "unregistered_snapshots"
REVIEW_SNAPSHOT_DIR = "review_snapshots"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npz")
WINDOW_NAME = "SecureCheck"

# --- Face model (must be the same model used in build_embeddings.py) ---
MODEL_NAME = "buffalo_l"      # use "buffalo_s" on the Raspberry Pi if it is too slow
DET_SIZE = (320, 320)         # smaller = faster. Raise to (640, 640) for far-away faces
CAMERA_WIDTH = 640            # lower camera resolution = smoother video
CAMERA_HEIGHT = 480
CAMERA_FLIP = None            # None = normal | 1 = mirror (left-right) | 0 = upside down | -1 = rotate 180
MIN_FACE_PX = 60              # ignore faces smaller than this (pixels)
MIN_DET_SCORE = 0.5           # ignore very uncertain face detections

# --- Feature 1: three-tier confidence (similarity: HIGHER is a BETTER match) ---
CONFIDENT_THRESHOLD = 0.45    # at or above this (and clear margin): green, mark attendance
UNCERTAIN_THRESHOLD = 0.30    # between UNCERTAIN and CONFIDENT: yellow, guard review queue
                              # below UNCERTAIN: red, unregistered flow
MIN_MARGIN = 0.05             # best match must beat the 2nd best student by at least this

CONFIRM_GREEN_FRAMES = 3      # same student seen this many checks in a row before marking
CONFIRM_FRAMES = 5            # consecutive checks before acting on unknown/uncertain
POSITION_TOLERANCE = 60       # px tolerance for "same face" between checks
MAX_MISSED_FRAMES = 8         # checks a tracker survives without being re-seen
ALARM_COOLDOWN_SECONDS = 30   # don't re-log/re-alarm the same tracked face too often

# --- Unregistered person alert: soft tone first, loud alarm only if seen again and again ---
SOFT_TONE_FRAMES = 2          # seen as unregistered this many checks in a row -> SOFT tone
LOUD_ALARM_FRAMES = 5         # seen this many times -> LOUD alarm + snapshot + log (avoids false alarms)

RECOGNITION_PAUSE = 0.12      # seconds the background worker rests between checks (keeps the video smooth)
SHOW_SCORES = True            # show the similarity score under each face (turn off for the demo)
SHOW_FPS = True               # show the video speed (frames per second) on screen


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
         guessed_id, guessed_name, f"{confidence:.2f}", snapshot_path, "pending"],
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


def play_alert(level, alarm_count=0):
    """
    Plays the alert sound in the background so the video never freezes.
    level = "soft" (one short beep) or "loud" (siren-like beeps).
    Uses the Windows beep when available, otherwise the alarm_utils sound (Raspberry Pi).
    """
    def _play():
        try:
            if winsound is not None:
                if level == "soft":
                    winsound.Beep(700, 250)
                else:
                    for _ in range(3):
                        winsound.Beep(1800, 350)
                        winsound.Beep(1200, 350)
            else:
                escalate_alarm(0 if level == "soft" else max(alarm_count, 1))
        except Exception as error:
            print(f"Sound error: {error}")

    threading.Thread(target=_play, daemon=True).start()


def setup_display_window(fullscreen=True):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# ---------------------------------------------------------------------------
# InsightFace recognition helpers
# ---------------------------------------------------------------------------

def ascii_text(text):
    """OpenCV's font cannot draw letters like N with tilde, so show them as plain letters."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def load_enrolled_embeddings(labels):
    """Loads data/embeddings.npz, keeping only people who are still in the labels
    (so expired visitors and deleted students are ignored)."""
    if not os.path.exists(EMBEDDINGS_FILE):
        print("No face embeddings found. Run build_embeddings.py first.")
        return None

    data = np.load(EMBEDDINGS_FILE)
    all_embs = data["embeddings"].astype(np.float32)
    all_ids = [str(i) for i in data["ids"].tolist()]

    keep = [i for i, sid in enumerate(all_ids) if sid in labels]
    if not keep:
        print("No enrolled people match the labels. Run build_embeddings.py again.")
        return None

    embs = all_embs[keep]
    kept_ids = [all_ids[i] for i in keep]
    student_ids = sorted(set(kept_ids), key=int)
    index_of = {sid: n for n, sid in enumerate(student_ids)}
    row_to_student = np.array([index_of[s] for s in kept_ids])
    return embs, row_to_student, student_ids


def identify_face(embedding, enrolled_embs, row_to_student, student_ids):
    """Returns (best_id, best_similarity, margin_over_second_best)."""
    sims = enrolled_embs @ embedding
    per_student = np.full(len(student_ids), -1.0, dtype=np.float32)
    np.maximum.at(per_student, row_to_student, sims)

    order = np.argsort(per_student)[::-1]
    best = float(per_student[order[0]])
    second = float(per_student[order[1]]) if len(order) > 1 else -1.0
    return student_ids[order[0]], best, best - second


def classify_tier(similarity, margin, info):
    if info is None:
        return "red"
    if similarity >= CONFIDENT_THRESHOLD and margin >= MIN_MARGIN:
        return "green"
    if similarity >= UNCERTAIN_THRESHOLD:
        return "yellow"
    return "red"


def hsv_to_color_name(hue, sat, val):
    """Turns an HSV color (OpenCV scale: hue 0-179) into a simple color name."""
    if val < 50:
        return "Black"
    if sat < 40:
        return "White" if val > 190 else "Gray"
    if hue < 8 or hue >= 170:
        return "Pink" if (sat < 120 and val > 150) else "Red"
    if hue < 20:
        return "Brown" if val < 150 else "Orange"
    if hue < 35:
        return "Yellow"
    if hue < 85:
        return "Green"
    if hue < 130:
        return "Blue"
    if hue < 160:
        return "Purple"
    return "Pink"


def backup_shirt_color(frame, box):
    """Built-in shirt color check, used when color_utils cannot give a color."""
    x, y, w, h = box
    frame_h, frame_w = frame.shape[:2]

    left, right = max(int(x + w * 0.1), 0), min(int(x + w * 0.9), frame_w)
    top, bottom = max(int(y + h * 1.15), 0), min(int(y + h * 2.2), frame_h)

    if bottom - top < 10 or right - left < 10:
        return "Unknown"          # no room below the face (person too close to the bottom)

    hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    hue = float(np.median(hsv[:, :, 0]))
    sat = float(np.median(hsv[:, :, 1]))
    val = float(np.median(hsv[:, :, 2]))
    return hsv_to_color_name(hue, sat, val)


def detect_shirt_color(frame, box):
    """Shirt color for ANY person (registered or not)."""
    try:
        color = get_dominant_shirt_color(frame, box)
        if color and str(color).strip().lower() != "unknown":
            return color
    except Exception:
        pass
    return backup_shirt_color(frame, box)


def detect_and_classify(frame, face_app, labels, enrolled_embs, row_to_student, student_ids):
    """Finds every face in the frame and decides green / yellow / red for each."""
    detections = []
    for face in face_app.get(frame):
        x1, y1, x2, y2 = face.bbox.astype(int)
        x, y = max(int(x1), 0), max(int(y1), 0)
        w, h = int(x2) - x, int(y2) - y
        if w < MIN_FACE_PX or h < MIN_FACE_PX or face.det_score < MIN_DET_SCORE:
            continue

        person_id, sim, margin = identify_face(face.normed_embedding, enrolled_embs,
                                               row_to_student, student_ids)
        info = labels.get(person_id)
        detections.append({
            "box": (x, y, w, h),
            "id": person_id,
            "info": info,
            "sim": sim,
            "tier": classify_tier(sim, margin, info),
            "confirmed": False,
            "label": "",
            "shirt": "",
        })
    return detections


def draw_detections(frame, detections):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for d in detections:
        x, y, w, h = d["box"]

        shirt_text = f" ({d['shirt']})" if d["shirt"] else ""

        if d["tier"] == "green":
            if d["confirmed"]:
                color, label = (0, 255, 0), d["label"]
            else:
                color, label = (255, 255, 255), f"Verifying...{shirt_text}"
        elif d["tier"] == "yellow":
            color = (0, 200, 255)
            label = f"Unsure: {ascii_text(d['info']['name'])}?{shirt_text}"
        else:
            color, label = (0, 0, 255), f"Unregistered?{shirt_text}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), font, 0.6, color, 2)
        if SHOW_SCORES:
            cv2.putText(frame, f"{d['sim']:.2f}", (x, y + h + 18), font, 0.5, color, 1)



class RecognitionWorker(threading.Thread):
    """
    Runs face recognition in the BACKGROUND so the camera video never freezes.
    The main loop keeps showing smooth video and just draws the latest results.
    """

    def __init__(self, face_app, labels, enrolled_embs, row_to_student, student_ids):
        super().__init__(daemon=True)
        self.face_app = face_app
        self.labels = labels
        self.enrolled_embs = enrolled_embs
        self.row_to_student = row_to_student
        self.student_ids = student_ids

        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.result = []
        self.result_id = 0
        self.running = True

    def submit(self, frame):
        with self.lock:
            self.frame = frame.copy()
            self.frame_id += 1

    def get_result(self):
        with self.lock:
            return self.result, self.result_id

    def stop(self):
        self.running = False

    def run(self):
        last_id = 0
        while self.running:
            with self.lock:
                frame, frame_id = self.frame, self.frame_id
            if frame is None or frame_id == last_id:
                time.sleep(0.005)
                continue
            last_id = frame_id

            try:
                detections = detect_and_classify(frame, self.face_app, self.labels,
                                                 self.enrolled_embs, self.row_to_student,
                                                 self.student_ids)
            except Exception as error:
                print(f"Recognition error: {error}")
                detections = []

            with self.lock:
                self.result = detections
                self.result_id += 1

            time.sleep(RECOGNITION_PAUSE)


def run():
    for visitor_id, name in cleanup_expired_visitors():
        print(f"Visitor access expired and removed: {name} (ID {visitor_id})")

    labels = load_all_labels()
    if not labels:
        print("No one enrolled yet. Run build_embeddings.py first.")
        return

    enrolled = load_enrolled_embeddings(labels)
    if enrolled is None:
        return
    enrolled_embs, row_to_student, student_ids = enrolled
    print(f"Loaded {len(student_ids)} enrolled people ({len(enrolled_embs)} face samples).")

    print("Loading face model (this can take a few seconds)...")
    face_app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=DET_SIZE)

    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(REVIEW_SNAPSHOT_DIR, exist_ok=True)

    attendance_path = get_today_attendance_path()
    already_marked = load_already_marked(attendance_path)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # DirectShow is usually faster on Windows
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open camera.")
        return
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # lets many USB webcams reach 30 FPS
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    worker = RecognitionWorker(face_app, labels, enrolled_embs, row_to_student, student_ids)
    worker.start()

    known_trackers = []
    unknown_trackers = []
    uncertain_trackers = []
    detections = []
    last_result_id = 0
    fps = 0.0
    fps_frames = 0
    fps_start = time.time()
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

        if CAMERA_FLIP is not None:
            frame = cv2.flip(frame, CAMERA_FLIP)

        # Hand the newest frame to the background worker (it picks it up when it is free)
        worker.submit(frame)

        # Did the worker finish a new recognition result?
        latest, result_id = worker.get_result()
        new_result = (result_id != last_result_id)

        seen_unknown_boxes = []
        seen_uncertain_boxes = []

        if new_result:
            last_result_id = result_id
            detections = latest

            for det in detections:
                det["shirt"] = detect_shirt_color(frame, det["box"])

            # --- Green: only mark attendance after the same student is seen several times in a row ---
            updated_known = []
            for det in detections:
                if det["tier"] != "green":
                    continue

                tracker = find_matching_tracker(known_trackers, det["box"])
                if tracker and tracker["id"] == det["id"]:
                    tracker["box"] = det["box"]
                    tracker["streak"] += 1
                    tracker["missed_frames"] = 0
                else:
                    tracker = {"box": det["box"], "id": det["id"], "streak": 1, "missed_frames": 0}
                updated_known.append(tracker)

                if tracker["streak"] >= CONFIRM_GREEN_FRAMES:
                    det["confirmed"] = True
                    person_id = det["id"]
                    name = det["info"]["name"]
                    person_type = det["info"]["type"]
                    shirt_color = det["shirt"] or "Unknown"

                    if person_id not in already_marked:
                        mark_attendance(attendance_path, person_id, name, person_type, shirt_color)
                        already_marked.add(person_id)
                        print(f"Attendance marked: {name} ({person_type}, ID {person_id}) - Shirt: {shirt_color}")

                    det["label"] = f"{ascii_text(name)} ({shirt_color})"

            for tracker in known_trackers:
                if tracker not in updated_known:
                    tracker["missed_frames"] += 1
                    tracker["streak"] = 0
                    if tracker["missed_frames"] <= MAX_MISSED_FRAMES:
                        updated_known.append(tracker)
            known_trackers = updated_known

            # --- Yellow and red faces are handed to the trackers below ---
            for det in detections:
                if det["tier"] == "yellow":
                    seen_uncertain_boxes.append((det["box"], det["id"], det["info"]["name"], det["sim"]))
                elif det["tier"] == "red":
                    seen_unknown_boxes.append((det["box"], det["shirt"]))

        # Draw the boxes on every frame (they update whenever a new result arrives)
        draw_detections(frame, detections)

        if new_result:
            # --- Red: unregistered person ---
            updated_unknown = []
            for box, shirt in seen_unknown_boxes:
                tracker = find_matching_tracker(unknown_trackers, box)
                if tracker:
                    tracker["box"] = box
                    tracker["frames_seen"] += 1
                    tracker["missed_frames"] = 0
                    tracker["shirt"] = shirt
                else:
                    tracker = {"box": box, "frames_seen": 1, "missed_frames": 0,
                               "alarm_count": 0, "last_alarm": None, "shirt": shirt,
                               "soft_played": False}
                updated_unknown.append(tracker)

                # Soft tone first (once), so a guard is only gently warned
                if tracker["frames_seen"] >= SOFT_TONE_FRAMES and not tracker["soft_played"]:
                    play_alert("soft")
                    tracker["soft_played"] = True
                    print("Possible unregistered person (soft tone)...")

                # Loud alarm only after the person is seen many times
                if tracker["frames_seen"] >= LOUD_ALARM_FRAMES:
                    now = datetime.now()
                    can_alarm = (tracker["last_alarm"] is None
                                 or (now - tracker["last_alarm"]).total_seconds() > ALARM_COOLDOWN_SECONDS)
                    if can_alarm:
                        play_alert("loud", tracker["alarm_count"])
                        print(f"Unregistered person detected - Shirt: {tracker['shirt'] or 'Unknown'}")
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

            # --- Yellow: uncertain, send to the guard review queue ---
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
                        log_review_entry(snapshot_path, tracker["guessed_id"],
                                         tracker["guessed_name"], tracker["confidence"])
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

        if SHOW_FPS:
            fps_frames += 1
            if fps_frames >= 20:
                fps = fps_frames / (time.time() - fps_start)
                fps_frames = 0
                fps_start = time.time()
            cv2.putText(frame, f"Video FPS: {fps:.0f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

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

    worker.stop()
    cam.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. {len(already_marked)} student(s)/visitor(s) marked present today.")
    print(f"Attendance file: {attendance_path}")


if __name__ == "__main__":
    run()
