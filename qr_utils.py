"""
qr_utils.py
------------
Generates a QR code that links directly to the live web dashboard,
so a guard, teacher, or panelist can scan it with their phone and
immediately see attendance records and logs - no need to type an
IP address manually.

The QR code encodes http://<this device's local IP>:5000, so
anyone scanning it must be on the SAME local network (same WiFi)
as the Raspberry Pi to actually load the page - exactly like the
"scan and connect" flow tested earlier with a phone on the same
WiFi.

Usage:
    from qr_utils import get_dashboard_qr_overlay
    qr_image = get_dashboard_qr_overlay(size=150)
    # qr_image is a BGR OpenCV-compatible numpy array, ready to
    # overlay onto a video frame.
"""

import socket
import numpy as np

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False

DASHBOARD_PORT = 5000
_cached_qr_image = None
_cached_url = None


def get_local_ip():
    """
    Finds this device's local network IP address (e.g. 192.168.1.5)
    without needing any external internet access - it briefly opens
    a UDP socket to a public address purely to ask the OS which
    local interface/IP would be used, without sending any real data.
    Falls back to '127.0.0.1' if this fails (e.g. no network at all).
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def get_dashboard_url():
    ip = get_local_ip()
    return f"http://{ip}:{DASHBOARD_PORT}"


def get_dashboard_qr_overlay(size=150):
    """
    Returns a small BGR image (numpy array, size x size) containing
    a QR code linking to the live dashboard, with a thin white
    border so it's visible against any camera background. Cached
    after first call so it isn't regenerated every frame.

    Returns None if the qrcode library isn't installed, or if a
    local IP could not be determined - callers should skip drawing
    the overlay in that case rather than crash.
    """
    global _cached_qr_image, _cached_url

    if not QRCODE_AVAILABLE:
        return None

    url = get_dashboard_url()

    if _cached_qr_image is not None and _cached_url == url and _cached_qr_image.shape[0] == size:
        return _cached_qr_image

    qr = qrcode.QRCode(box_size=4, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    pil_img = pil_img.resize((size, size))

    # Convert PIL (RGB) to an OpenCV-style BGR numpy array
    rgb_array = np.array(pil_img)
    bgr_array = rgb_array[:, :, ::-1].copy()

    _cached_qr_image = bgr_array
    _cached_url = url
    return bgr_array


def overlay_qr_on_frame(frame, size=150, margin=15):
    """
    Draws the dashboard QR code onto the bottom-right corner of a
    video frame (a BGR numpy array from OpenCV), along with a small
    'Scan for live logs' label above it. Does nothing (returns the
    frame unchanged) if the QR code couldn't be generated.
    """
    import cv2

    qr_image = get_dashboard_qr_overlay(size=size)
    if qr_image is None:
        return frame

    frame_h, frame_w = frame.shape[:2]
    x2 = frame_w - margin
    x1 = x2 - size
    y2 = frame_h - margin
    y1 = y2 - size

    if x1 < 0 or y1 < 0:
        return frame  # frame too small for the overlay - skip rather than crash

    # White backing rectangle so the QR reads cleanly against busy backgrounds
    cv2.rectangle(frame, (x1 - 6, y1 - 26), (x2 + 6, y2 + 6), (255, 255, 255), -1)
    cv2.putText(frame, "Scan for live logs", (x1 - 2, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    frame[y1:y2, x1:x2] = qr_image

    return frame
