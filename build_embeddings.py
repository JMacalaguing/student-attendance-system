"""
build_embeddings.py
-------------------
Builds InsightFace face embeddings from COLOR photos (phone photos are fine).
This replaces train_recognizer.py for the new InsightFace system.

Expected folder layout (one folder per student). Either style works:
    students data/
        JUAN DELA CRUZ/      (name only -> ID is assigned automatically)
        MARIA SANTOS/
        5_Pedro Reyes/       (or ID_Name -> you choose the ID)
        ...

What it does for each photo:
  - finds the face with SCRFD, skips bad photos (no face, too small, low confidence)
  - computes a 512-number embedding (ArcFace)
  - keeps a copy of the photo in data/student_images/<id>/
Then saves everything to data/embeddings.npz and updates the student labels file.

Usage:
    python build_embeddings.py
"""

import os
import cv2
import numpy as np
from importlib import import_module
from insightface.app import FaceAnalysis

if __package__:
    student_data = import_module(f"{__package__}.student_data")
else:
    student_data = import_module("student_data")

DATA_DIR = student_data.DATA_DIR
STUDENT_IMAGES_DIR = student_data.STUDENT_IMAGES_DIR
load_student_labels = student_data.load_student_labels
save_student_labels = student_data.save_student_labels

EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npz")
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_SIDE = 1280        # shrink huge phone photos (faster, still plenty of detail)
MIN_FACE_PX = 80       # skip faces smaller than this
MIN_DET_SCORE = 0.6    # skip uncertain detections

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def clean_path(p):
    return p.strip().strip('"').strip("'")


def read_image(path):
    """Reads a color image. Works even if the path has special characters (e.g. ñ)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def get_embedding(img):
    """Returns (embedding, resized_img) for the biggest good face, else (None, None)."""
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    faces = app.get(img)
    if not faces:
        return None, None

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    fw = face.bbox[2] - face.bbox[0]
    fh = face.bbox[3] - face.bbox[1]
    if min(fw, fh) < MIN_FACE_PX or face.det_score < MIN_DET_SCORE:
        return None, None

    return face.normed_embedding, img


def parse_folder_name(sub, name_to_id, used_ids):
    """
    Folder can be named  "5_Juan Dela Cruz"  (you choose the ID)
    or just              "JUAN DELA CRUZ"     (ID is assigned automatically,
                                               and reused if the name was enrolled before).
    """
    if "_" in sub:
        left, right = sub.split("_", 1)
        if left.strip().isdigit():
            return left.strip(), right.strip()

    name = sub.strip()
    if name in name_to_id:
        return name_to_id[name], name

    next_id = 1
    while str(next_id) in used_ids:
        next_id += 1
    return str(next_id), name


def process_student(student_id, student_name, folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(VALID_EXT))
    if not files:
        print("  No photos found (iPhone HEIC photos must be converted to JPG).")
        return []

    out_dir = os.path.join(STUDENT_IMAGES_DIR, student_id)
    os.makedirs(out_dir, exist_ok=True)

    embeddings = []
    for f in files:
        img = read_image(os.path.join(folder, f))
        if img is None:
            print(f"  [cannot read] {f}")
            continue

        emb, small = get_embedding(img)
        if emb is None:
            print(f"  [no good face] {f}")
            continue

        embeddings.append(emb)
        cv2.imwrite(os.path.join(out_dir, f"{len(embeddings)}.jpg"), small)

    if not embeddings and not os.listdir(out_dir):
        os.rmdir(out_dir)

    print(f"  {len(embeddings)}/{len(files)} photos used for {student_name} (ID {student_id}).")
    if 0 < len(embeddings) < 3:
        print("  WARNING: fewer than 3 usable photos. Add more for better accuracy.")
    return embeddings


def main():
    parent = clean_path(input("Enter the main folder (contains ID_Full Name folders): "))
    if not os.path.isdir(parent):
        print("Folder not found.")
        return

    new_embs, new_ids = [], []
    labels = load_student_labels()
    name_to_id = {name: sid for sid, name in labels.items()}
    used_ids = set(labels.keys())

    for sub in sorted(os.listdir(parent)):
        sub_path = os.path.join(parent, sub)
        if not os.path.isdir(sub_path):
            continue

        student_id, student_name = parse_folder_name(sub, name_to_id, used_ids)
        used_ids.add(student_id)
        name_to_id[student_name] = student_id

        print(f"Processing {student_name} (ID {student_id}) ...")
        embs = process_student(student_id, student_name, sub_path)
        if embs:
            new_embs.extend(embs)
            new_ids.extend([int(student_id)] * len(embs))
            labels[student_id] = student_name

    if not new_embs:
        print("\nNothing was enrolled.")
        return

    new_embs = np.array(new_embs, dtype=np.float32)
    new_ids = np.array(new_ids, dtype=np.int32)

    # Merge with existing embeddings (re-imported students are replaced)
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(EMBEDDINGS_FILE):
        old = np.load(EMBEDDINGS_FILE)
        keep = ~np.isin(old["ids"], new_ids)
        new_embs = np.vstack([old["embeddings"][keep], new_embs])
        new_ids = np.concatenate([old["ids"][keep], new_ids])

    np.savez(EMBEDDINGS_FILE, embeddings=new_embs, ids=new_ids)
    save_student_labels(labels)

    # Save a readable ID list so you know which ID belongs to which student
    id_list_file = os.path.join(DATA_DIR, "student_id_list.csv")
    with open(id_list_file, "w", encoding="utf-8-sig") as f:
        f.write("id,name\n")
        for sid in sorted(labels.keys(), key=int):
            f.write(f'{sid},"{labels[sid]}"\n')
    print(f"ID list saved to {id_list_file}")

    print(f"\nDone. {len(set(new_ids.tolist()))} students, {len(new_ids)} embeddings saved to {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
