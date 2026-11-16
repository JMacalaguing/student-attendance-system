# Student Attendance System — Full Version

Face-detection attendance system: detects student faces, logs daily
attendance, detects the color of each student's shirt, alarms and
logs unregistered faces, and adds a guard-facing web dashboard.

## What's included

| File | Purpose |
|---|---|
| `student_data.py` | Shared storage for students + visitors, visitor auto-expiry logic |
| `enroll_student.py` | Enroll a permanent student (captures 30 face photos) |
| `visitor_enroll.py` | **Feature 5** — enroll a visitor/parent with a 24h (configurable) expiry |
| `train_recognizer.py` | Trains the face model on students + active visitors; auto-cleans expired visitors first |
| `color_utils.py` | Detects the dominant color of the shirt region below a detected face |
| `alarm_utils.py` | **Feature 3** — soft-then-loud tiered beep escalation |
| `attendance_system.py` | Main camera loop: detection, attendance, 3-tier confidence (**Feature 1**), alarms, guard keypress (**Feature 4**) |
| `set_admin_password.py` | One-time setup for the webview login |
| `app.py` | **Feature 2** — Flask dashboard: live attendance, review queue, guard button, student/visitor photo management |
| `templates/`, `static/` | Webview HTML/CSS |

## Installation

### Requirements

- Python 3.10 or newer
- A webcam for face enrollment and attendance
- Windows, macOS, or Linux with a working camera driver

### Install dependencies

Run these commands from the project directory.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`opencv-contrib-python` is required (not plain `opencv-python`) — it
includes `cv2.face`, used for recognition.

After installing, enroll at least one person and train the recognizer
before starting attendance (see [Day-to-day usage](#day-to-day-usage)).

## Privacy and repository contents

This application processes face images and attendance records, which
can contain sensitive personal data. Runtime files are intentionally
excluded from version control by `.gitignore`; they are created locally
when the application runs. Do not remove those ignore rules or commit
real student/visitor photos, snapshots, credentials, trained models, or
attendance records to a repository.

## Day-to-day usage

### 1. Enroll people

```bash
python enroll_student.py      # once per permanent student
python visitor_enroll.py      # for a parent/visitor who needs temporary access
python train_recognizer.py    # re-run after any enrollment change
```

### 2. Run attendance (on the camera machine)

```bash
python attendance_system.py
```
- `q` — quit
- `g` — **Call for Guard** manually (Feature 4) — logs an entry even
  when the camera itself didn't flag anything

### 3. Run the web dashboard (Feature 2)

```bash
python set_admin_password.py   # first time only, sets login credentials
python app.py
```
Open `http://localhost:5000`. From there you can:
- See today's attendance, live
- Approve/reject **uncertain matches** sent to the review queue (Feature 1)
- Call the guard from the browser (Feature 4)
- View recent unregistered-face alerts with snapshots
- View/remove enrolled students and active visitors, including their
  photos (editable images)

By default the dashboard only listens on `127.0.0.1` (this machine
only). To let a guard's phone reach it over the school's local WiFi,
change `host="127.0.0.1"` to `host="0.0.0.0"` at the bottom of
`app.py` — **only do this on a trusted private network**, never on
open/public WiFi, since it shows student faces and names.

## How each of the 5 features works

### 1. Low-confidence reconfirm
`attendance_system.py` splits every match into three tiers based on
LBPH's confidence score (lower = better match):
- **< `CONFIDENT_THRESHOLD`** → confident match, attendance marked automatically
- **between `CONFIDENT_THRESHOLD` and `UNCERTAIN_THRESHOLD`** → uncertain,
  sent to the review queue for a human to confirm via the webview
- **> `UNCERTAIN_THRESHOLD`** → confident no-match, treated as unregistered

Tune both thresholds in `attendance_system.py` based on your camera
and lighting during testing.

### 2. Live webview + CSV + editable images
`app.py` reads the same CSV files `attendance_system.py` writes —
no separate database. Student and visitor photos are viewable and
deletable from the "Students & Visitors" page. Login is required
(set via `set_admin_password.py`, stored as a hash, never plain text).

### 3. Tiered beeping
`alarm_utils.py`: the first confirmed sighting of an unregistered
face is a soft alert; if the *same* face is confirmed again (it's
still there, or keeps coming back), it escalates to a loud alert.
Swap the bodies of `soft_alert()`/`loud_alert()` for GPIO buzzer
code if you move this to a Raspberry Pi.

### 4. Call for Guard
Two ways to trigger it: press `g` in the camera window, or click
"Call Guard Now" in the webview (with an optional note). Both write
to `attendance/guard_calls.csv` with a timestamp.

### 5. Visitor/parent temporary enrollment
`visitor_enroll.py` registers a face the same way as a student, but
stores an `expires_at` timestamp and assigns IDs starting at 9000 so
they never collide with student IDs. `cleanup_expired_visitors()` in
`student_data.py` deletes expired visitors' photos and label entries
— it runs automatically before training and every time
`attendance_system.py` starts, so expired biometric data doesn't
linger. This matches the data-minimization principle expected under
privacy laws (including the Philippines' Data Privacy Act) for
systems collecting biometric data.

## What gets created at runtime

```
data/
  student_labels.json           # {id: name}
  visitor_labels.json           # {id: {name, note, enrolled_at, expires_at}}
  student_images/<id>/*.jpg
  visitor_images/<id>/*.jpg
  trainer/trainer.yml           # trained recognition model
  admin_credentials.json        # webview login (hashed password)
attendance/
  attendance_2026-08-21.csv     # student_id, name, type, time, shirt_color
  unregistered_log.csv          # every unregistered-face alarm
  review_queue.csv              # uncertain matches awaiting guard decision
  guard_calls.csv               # every manual guard call
  daily_summary.csv             # one row/day: had an unregistered student? YES/NO
unregistered_snapshots/         # photo taken at each alarm
review_snapshots/               # photo taken at each uncertain-match flag
```

## Things worth writing up in your paper (and testing before demo day)

- **Privacy/consent**: this collects biometric data from minors.
  Worth a short section on how long data is kept (permanent students:
  until manually removed; visitors: max 24h by default), who can see
  it (password-protected dashboard only), and that students/parents
  are informed. The visitor auto-expiry is your strongest talking
  point here — lead with it.
- **Threshold tuning**: `CONFIDENT_THRESHOLD` / `UNCERTAIN_THRESHOLD`
  need real-world testing in your actual classroom lighting before
  the demo — LBPH confidence scales differently per camera/lighting
  setup.
- **Backup/failure behavior**: what happens if the machine loses
  power or network mid-day? Right now, attendance already written to
  CSV survives (it's written incrementally, not held in memory), but
  there's no cloud backup — worth a line in your paper on this as a
  known limitation or future work.
- **Escalation beyond loud beep**: currently the loud alert is the
  end of the automated chain — what the guard does next (a checklist,
  who they notify) is outside this system's scope and worth defining
  briefly in your documentation even if not implemented in code.
