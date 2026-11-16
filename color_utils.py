"""
color_utils.py
---------------
Classifies the dominant color of an image region (used for detecting
the color of a student's upper-body clothing).

Approach: convert the region to HSV, take the median Hue/Saturation/
Value, then map that to a human-readable color name using simple
threshold rules. This is intentionally simple (no ML) so it runs
fast on every frame for 10+ students at once.
"""

import importlib

# Load OpenCV dynamically so environments without its type stubs do not flag
# this module during static analysis. OpenCV is still required at runtime.
cv2 = importlib.import_module("cv2")
# Load NumPy dynamically for environments where static analysis cannot resolve
# the package, while still requiring it at runtime.
np = importlib.import_module("numpy")

# (name, hue_min, hue_max) - hue ranges in OpenCV's 0-179 scale
COLOR_RANGES = [
    ("Red", 0, 10),
    ("Orange", 11, 20),
    ("Yellow", 21, 33),
    ("Green", 34, 82),
    ("Cyan", 83, 100),
    ("Blue", 101, 130),
    ("Purple", 131, 150),
    ("Pink", 151, 169),
    ("Red", 170, 179),  # red wraps around
]


def get_shirt_region(frame, face_box, offset_ratio=0.3, height_ratio=0.9):
    """
    Given a face bounding box (x, y, w, h), estimate the region
    corresponding to the upper body / shirt directly below the face.
    Returns the cropped region, or None if it falls outside the frame.
    """
    x, y, w, h = face_box
    frame_h, frame_w = frame.shape[:2]

    # Widen the box a bit sideways (shoulders are wider than the face)
    pad_x = int(w * offset_ratio)
    shirt_x1 = max(0, x - pad_x)
    shirt_x2 = min(frame_w, x + w + pad_x)

    # Start just below the chin, extend down toward the chest
    shirt_y1 = min(frame_h, y + h + int(h * 0.15))
    shirt_y2 = min(frame_h, shirt_y1 + int(h * height_ratio))

    if shirt_y2 <= shirt_y1 or shirt_x2 <= shirt_x1:
        return None

    return (shirt_x1, shirt_y1, shirt_x2, shirt_y2)


def classify_color(hsv_pixel):
    h, s, v = hsv_pixel

    # Low saturation = grayscale (white/gray/black), regardless of hue
    if v < 50:
        return "Black"
    if s < 40:
        if v > 190:
            return "White"
        return "Gray"

    for name, h_min, h_max in COLOR_RANGES:
        if h_min <= h <= h_max:
            return name

    return "Unknown"


def get_dominant_shirt_color(frame, face_box):
    """
    Returns the color name (str) of the clothing region below the
    given face box, or None if the region couldn't be sampled.
    """
    region_coords = get_shirt_region(frame, face_box)
    if region_coords is None:
        return None

    x1, y1, x2, y2 = region_coords
    region = frame[y1:y2, x1:x2]

    if region.size == 0:
        return None

    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # Median is more robust to noise/shadows than mean
    median_hsv = np.median(hsv_region.reshape(-1, 3), axis=0)

    return classify_color(median_hsv)
