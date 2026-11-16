"""
app.py
-------
Feature 2: Live webview - a small local Flask dashboard that shows
today's attendance (from the CSVs attendance_system.py writes),
lets a guard/teacher review uncertain matches (feature 1), call the
guard (feature 4), and manage enrolled students/visitors including
deleting their photos (feature 5's manual counterpart to auto-expiry).

Login-protected, supports MULTIPLE admin accounts (see
set_admin_password.py) - nobody on the same network can view student
faces or attendance without valid credentials.

Setup:
    python set_admin_password.py     (first time only, and any time
                                       you want to add/change an account)
    python app.py

Then open http://localhost:5000 in a browser ON THE SAME MACHINE.
To make it reachable from other devices on the school's local
network, run with host="0.0.0.0" (see bottom of this file) - only
do this on a trusted, private network.
"""

import os
import csv
import json
import secrets
from datetime import date
from functools import wraps

# Flask is installed in the application's environment; suppress an editor
# diagnostic when the editor is using a different Python interpreter.
from flask import (  # pyright: ignore[reportMissingImports]
    Flask, render_template, request, redirect, url_for,
    session, send_from_directory, flash, jsonify
)
from werkzeug.security import check_password_hash  # pyright: ignore[reportMissingImports]

import student_data
import train_recognizer

ATTENDANCE_DIR = "attendance"
CREDENTIALS_FILE = "data/admin_credentials.json"
SECRET_KEY_FILE = "data/flask_secret_key.txt"
SNAPSHOT_DIR = "unregistered_snapshots"
REVIEW_SNAPSHOT_DIR = "review_snapshots"
REVIEW_QUEUE_FILE = os.path.join(ATTENDANCE_DIR, "review_queue.csv")
GUARD_CALLS_FILE = os.path.join(ATTENDANCE_DIR, "guard_calls.csv")


def get_secret_key():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(SECRET_KEY_FILE):
        with open(SECRET_KEY_FILE, "r") as f:
            return f.read().strip()
    key = secrets.token_hex(32)
    with open(SECRET_KEY_FILE, "w") as f:
        f.write(key)
    return key


app = Flask(__name__)
app.secret_key = get_secret_key()


# ---------- Auth ----------

def load_admins():
    """Returns the list of admin accounts, migrating the old single-account format if present."""
    if not os.path.exists(CREDENTIALS_FILE):
        return []

    with open(CREDENTIALS_FILE, "r") as f:
        data = json.load(f)

    if "admins" in data:
        return data["admins"]

    if "username" in data and "password_hash" in data:
        return [{"username": data["username"], "password_hash": data["password_hash"]}]

    return []


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    admins = load_admins()

    if not admins:
        flash("No admin accounts set up yet. Run 'python set_admin_password.py' first.", "error")
        return render_template("login.html")

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        account = next((a for a in admins if a["username"] == username), None)

        if account and check_password_hash(account["password_hash"], password):
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        flash("Incorrect username or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------- Helpers ----------

def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def today_attendance_path():
    return os.path.join(ATTENDANCE_DIR, f"attendance_{date.today().isoformat()}.csv")


# ---------- Dashboard ----------

@app.route("/")
@login_required
def dashboard():
    attendance_rows = read_csv_rows(today_attendance_path())
    unregistered_rows = list(reversed(read_csv_rows(os.path.join(ATTENDANCE_DIR, "unregistered_log.csv"))))[:20]
    review_rows = [r for r in read_csv_rows(REVIEW_QUEUE_FILE) if r["status"] == "pending"]
    guard_calls = list(reversed(read_csv_rows(GUARD_CALLS_FILE)))[:10]

    # Normalize to forward slashes before taking the basename, so this
    # works correctly no matter what separator wrote the path (Windows
    # writes "review_snapshots\\file.jpg"; just splitting on '/' alone
    # left the whole backslash path intact and broke every image on
    # Windows specifically).
    for row in unregistered_rows:
        row["snapshot_filename"] = row["snapshot"].replace("\\", "/").split("/")[-1]
    for row in review_rows:
        row["snapshot_filename"] = row["snapshot"].replace("\\", "/").split("/")[-1]

    return render_template(
        "dashboard.html",
        today=date.today().isoformat(),
        attendance_rows=attendance_rows,
        unregistered_rows=unregistered_rows,
        review_rows=review_rows,
        guard_calls=guard_calls,
        initial_guard_call_count=len(read_csv_rows(GUARD_CALLS_FILE)),
        initial_unregistered_count=len(read_csv_rows(os.path.join(ATTENDANCE_DIR, "unregistered_log.csv"))),
    )


@app.route("/api/alert-counts")
@login_required
def api_alert_counts():
    """
    Lightweight JSON endpoint the dashboard polls to detect NEW guard
    calls or unregistered-face alerts while the page is open, so it
    can play a sound - this is what makes 'Call for Guard' actually
    notify someone: whoever has this dashboard open in a browser tab
    hears it immediately.
    """
    guard_call_count = len(read_csv_rows(GUARD_CALLS_FILE))
    unregistered_count = len(read_csv_rows(os.path.join(ATTENDANCE_DIR, "unregistered_log.csv")))
    review_pending_count = len([r for r in read_csv_rows(REVIEW_QUEUE_FILE) if r["status"] == "pending"])
    return jsonify({
        "guard_calls": guard_call_count,
        "unregistered": unregistered_count,
        "review_pending": review_pending_count,
    })


@app.route("/call-guard", methods=["POST"])
@login_required
def call_guard():
    note = request.form.get("note", "")
    from datetime import datetime
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    file_exists = os.path.exists(GUARD_CALLS_FILE)
    with open(GUARD_CALLS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "time", "source", "note"])
        writer.writerow([date.today().isoformat(), datetime.now().strftime("%H:%M:%S"),
                          f"webview ({session.get('username')})", note])
    flash("Guard called.", "success")
    return redirect(url_for("dashboard"))


# ---------- Feature 1: Review queue (approve/reject uncertain matches) ----------

@app.route("/review/<entry_id>/<action>", methods=["POST"])
@login_required
def review_action(entry_id, action):
    rows = read_csv_rows(REVIEW_QUEUE_FILE)
    fieldnames = ["entry_id", "date", "time", "guessed_id", "guessed_name", "confidence", "snapshot", "status"]
    target = None

    for row in rows:
        if row["entry_id"] == entry_id:
            target = row
            if action == "approve":
                row["status"] = "approved"
            elif action == "reject":
                row["status"] = "rejected"
            break

    if target is None:
        flash("Review entry not found.", "error")
        return redirect(url_for("dashboard"))

    write_csv_rows(REVIEW_QUEUE_FILE, fieldnames, rows)

    if action == "approve":
        att_path = today_attendance_path()
        all_labels = student_data.load_all_labels()
        info = all_labels.get(target["guessed_id"])
        person_type = info["type"] if info else "unknown"

        already_marked = set()
        if os.path.exists(att_path):
            already_marked = {r["student_id"] for r in read_csv_rows(att_path)}

        if target["guessed_id"] not in already_marked:
            file_exists = os.path.exists(att_path)
            with open(att_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["student_id", "name", "type", "time", "shirt_color"])
                from datetime import datetime
                writer.writerow([target["guessed_id"], target["guessed_name"], person_type,
                                  datetime.now().strftime("%H:%M:%S"), "N/A (guard-confirmed)"])
        flash(f"Approved: {target['guessed_name']} marked present.", "success")
    else:
        flash(f"Rejected: {target['guessed_name']} was NOT marked present.", "success")

    return redirect(url_for("dashboard"))


# ---------- Students / Visitors management (editable images) ----------

@app.route("/students")
@login_required
def students():
    student_labels = student_data.load_student_labels()
    visitor_labels = student_data.load_visitor_labels()

    student_list = []
    for sid, name in student_labels.items():
        folder = os.path.join(student_data.STUDENT_IMAGES_DIR, sid)
        photos = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
        student_list.append({"id": sid, "name": name, "photo_count": len(photos),
                              "thumb": photos[0] if photos else None})

    visitor_list = []
    for vid, entry in visitor_labels.items():
        folder = os.path.join(student_data.VISITOR_IMAGES_DIR, vid)
        photos = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
        visitor_list.append({"id": vid, "name": entry["name"], "note": entry.get("note", ""),
                              "expires_at": entry.get("expires_at", ""), "photo_count": len(photos),
                              "thumb": photos[0] if photos else None})

    return render_template("students.html", students=student_list, visitors=visitor_list)


@app.route("/students/<student_id>/delete", methods=["POST"])
@login_required
def delete_student(student_id):
    student_data.delete_student(student_id)
    train_recognizer.train()
    flash("Student removed and model retrained.", "success")
    return redirect(url_for("students"))


@app.route("/visitors/<visitor_id>/revoke", methods=["POST"])
@login_required
def revoke_visitor(visitor_id):
    labels = student_data.load_visitor_labels()
    if visitor_id in labels:
        del labels[visitor_id]
        student_data.save_visitor_labels(labels)
        import shutil
        folder = os.path.join(student_data.VISITOR_IMAGES_DIR, visitor_id)
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        train_recognizer.train()
        flash("Visitor access revoked and model retrained.", "success")
    return redirect(url_for("students"))


# ---------- Serving images ----------

@app.route("/snapshot/<path:filename>")
@login_required
def serve_unregistered_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)


@app.route("/review-snapshot/<path:filename>")
@login_required
def serve_review_snapshot(filename):
    return send_from_directory(REVIEW_SNAPSHOT_DIR, filename)


@app.route("/student-photo/<student_id>/<path:filename>")
@login_required
def serve_student_photo(student_id, filename):
    return send_from_directory(os.path.join(student_data.STUDENT_IMAGES_DIR, student_id), filename)


@app.route("/visitor-photo/<visitor_id>/<path:filename>")
@login_required
def serve_visitor_photo(visitor_id, filename):
    return send_from_directory(os.path.join(student_data.VISITOR_IMAGES_DIR, visitor_id), filename)


if __name__ == "__main__":
    # Keep the dashboard local by default. Change to host="0.0.0.0"
    # to allow other devices on the SAME
    # trusted local network to connect (e.g. a guard's phone) -
    # only do this on a private school WiFi, never on the open internet.
    app.run(host="127.0.0.1", port=5000, debug=False)
