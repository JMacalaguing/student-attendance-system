#!/usr/bin/env python3
"""
launch_system.py
------------------
One-click launcher for presentation/demo day. Starts the web
dashboard (app.py) quietly in the background, waits a moment for it
to come up, then starts the main camera system (attendance_system.py)
in fullscreen. When the camera window is closed ('q'), the dashboard
is automatically stopped too - so there's only ever ONE thing to
start and ONE thing to stop, instead of juggling two terminals.

Usage:
    python3 launch_system.py

To make this a literal double-click icon on the Raspberry Pi
desktop, see the accompanying SecureCheck.desktop file.
"""

import subprocess
import sys
import os
import time


def main():
    python_exe = sys.executable  # whichever python this launcher itself is run with

    # Always work from THIS script's own folder, regardless of where
    # it was launched from (double-click icons often start in the
    # home folder instead of the project folder, which would
    # otherwise make app.py / attendance_system.py "not found").
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    print(f"Working directory: {project_dir}")
    print("Starting SecureCheck...")
    print("(dashboard starting in the background, camera window opening shortly)")

    dashboard_process = subprocess.Popen(
        [python_exe, "app.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=project_dir,
    )

    # Give Flask a moment to bind to its port before anyone tries to load it
    time.sleep(2)

    result = None
    try:
        # Runs in the foreground - the launcher waits here until the
        # camera window is closed with 'q'.
        result = subprocess.run([python_exe, "attendance_system.py"], cwd=project_dir)
    finally:
        print("Shutting down dashboard...")
        dashboard_process.terminate()
        try:
            dashboard_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            dashboard_process.kill()
        print("SecureCheck stopped.")

    # A non-zero exit code (e.g. a missing file, a camera error) doesn't
    # raise a Python exception on its own - check for it explicitly so a
    # double-clicked terminal window doesn't vanish before you can read
    # what went wrong.
    if result is not None and result.returncode != 0:
        print(f"\nattendance_system.py exited with an error (code {result.returncode}).")
        input("Press Enter to close this window...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # If anything crashes, print it clearly and wait for a
        # keypress before closing - so double-clicking this file
        # doesn't make the terminal window vanish before you can
        # read what went wrong.
        print(f"\nERROR: {e}")
        input("\nPress Enter to close this window...")
