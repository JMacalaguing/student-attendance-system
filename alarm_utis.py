"""
alarm_utils.py
--------------
Tiered beeping escalation.
  - 1st confirmed sighting of an unknown face -> soft alert
  - 2nd+ sighting of the SAME unknown face -> loud alert (stays loud)

This module only handles the sound. attendance_system.py tracks how many
times a face has been alarmed on (alarm_count).
"""

import array
from importlib import import_module
import math
import sys
import time

IS_WINDOWS = sys.platform.startswith("win")

if IS_WINDOWS:
    import winsound

_mixer_ready = False
_tone_cache = {}


def _make_tone(frequency_hz, duration_ms):
    pygame = import_module("pygame")

    sample_rate = 44100
    n = int(sample_rate * duration_ms / 1000)
    samples = array.array(
        "h",
        (
            int(32767 * 0.8 * math.sin(2 * math.pi * frequency_hz * i / sample_rate))
            for i in range(n)
        ),
    )
    return pygame.mixer.Sound(buffer=samples.tobytes())


def _beep(frequency_hz, duration_ms):
    global _mixer_ready

    if IS_WINDOWS:
        try:
            winsound.Beep(frequency_hz, duration_ms)
            return
        except (AttributeError, RuntimeError):
            pass

    try:
        pygame = import_module("pygame")

        if not _mixer_ready:
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
            _mixer_ready = True

        key = (frequency_hz, duration_ms)
        if key not in _tone_cache:
            _tone_cache[key] = _make_tone(frequency_hz, duration_ms)

        _tone_cache[key].play()
        time.sleep(duration_ms / 1000)
        return
    except Exception as e:
        print(f"[alarm] pygame audio failed: {e}")

    print("\a", end="", flush=True)
    time.sleep(duration_ms / 1000)


def soft_alert():
    print("[ALERT - soft] Unrecognized face detected. Monitoring ...")
    _beep(700, 200)  # one short, medium-pitch beep


def loud_alert():
    print("[ALERT - loud] Unrecognized face detected repeatedly! Guard attention needed.")
    _beep(1000, 250)  # three sharp, higher-pitched beeps
    time.sleep(0.1)


def escalate(alarm_count):
    """
    Call this each time a tracked unknown face has been confirmed again.
    alarm_count is how many times THIS face has already been alarmed on
    (0 on the first time this call fires).
    """
    if alarm_count <= 0:
        soft_alert()
    else:
        loud_alert()

