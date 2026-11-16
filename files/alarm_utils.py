"""
alarm_utils.py
---------------
Feature 3: Tiered beeping escalation.

A single unrecognized face should not blast a loud alarm the first
time it's seen — that gets ignored fast if it happens whenever
lighting shifts (alarm fatigue). Instead:
  - 1st confirmed sighting of an unknown face -> soft alert
  - 2nd+ confirmed sighting of the SAME unknown face (i.e. it's
    still there / keeps coming back) -> loud alert, and it stays
    loud for further repeats

This module only handles the *sound*. attendance_system.py is
responsible for tracking how many times a given face has already
been alarmed on (see `alarm_count` on each tracker).

SOUND OUTPUT: uses Windows' built-in speaker beep (winsound.Beep) on
Windows, which is audible and reliable. Falls back to the terminal
bell character on other operating systems (Mac/Linux), where its
audibility depends on terminal settings.

If you wire up a physical buzzer (e.g. on a Raspberry Pi via GPIO),
replace the bodies of soft_alert()/loud_alert() with your GPIO
trigger code — everything that calls this module stays the same.
"""

import sys
import time

IS_WINDOWS = sys.platform.startswith("win")

if IS_WINDOWS:
    import winsound


def _beep(frequency_hz, duration_ms):
    if IS_WINDOWS:
        try:
            winsound.Beep(frequency_hz, duration_ms)
            return
        except RuntimeError:
            pass  # some environments (e.g. no sound device) can still fail - fall through
    # Non-Windows fallback: terminal bell. Audibility depends on the
    # terminal's own settings (some terminals mute this by default).
    print("\a", end="", flush=True)
    time.sleep(duration_ms / 1000)


def soft_alert():
    print("[ALERT - soft] Unrecognized face detected. Monitoring...")
    _beep(700, 200)  # one short, medium-pitch beep


def loud_alert():
    print("[ALERT - LOUD] Unrecognized face detected repeatedly! Guard attention needed.")
    for _ in range(3):
        _beep(1000, 250)  # three sharp, higher-pitched beeps
        time.sleep(0.1)


def escalate(alarm_count):
    """
    Call this each time a tracked unknown face has been confirmed
    again. alarm_count is how many times THIS face has already been
    alarmed on (0 the first time this call fires).
    """
    if alarm_count <= 0:
        soft_alert()
    else:
        loud_alert()
