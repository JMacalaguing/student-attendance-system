import time, math, array

_ready = False


def _beep(freq, ms):
    global _ready
    try:
        import pygame
        if not _ready:
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
            _ready = True
        n = int(44100 * ms / 1000)
        data = array.array("h", (int(26000 * math.sin(2 * math.pi * freq * i / 44100)) for i in range(n)))
        pygame.mixer.Sound(buffer=data.tobytes()).play()
    except Exception as e:
        print("[alarm] audio failed:", e)
    time.sleep(ms / 1000)


def soft_alert():
    print("[ALERT - soft] Unrecognized face detected. Monitoring...")
    _beep(700, 200)


def loud_alert():
    print("[ALERT - LOUD] Unrecognized face detected repeatedly!")
    for _ in range(3):
        _beep(1000, 250)
        time.sleep(0.1)


def escalate(alarm_count):
    if alarm_count <= 0:
        soft_alert()
    else:
        loud_alert()
