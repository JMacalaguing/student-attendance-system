"""
clean_labels.py
---------------
Finds students who are in the labels file but have NO face embeddings
(leftover entries from old data). Shows them first, then removes them
from the labels file ONLY if you type yes.

It does not touch your photos or embeddings.

Usage (from the same folder as student_data.py):
    python clean_labels.py
"""

import os
import numpy as np
from importlib import import_module

if __package__:
    student_data = import_module(f"{__package__}.student_data")
else:
    student_data = import_module("student_data")

embeddings_file = os.path.join(student_data.DATA_DIR, "embeddings.npz")
if not os.path.exists(embeddings_file):
    print("embeddings.npz not found. Run build_embeddings.py first.")
    raise SystemExit

enrolled_ids = {str(i) for i in np.load(embeddings_file)["ids"].tolist()}
labels = student_data.load_student_labels()

ghosts = {sid: name for sid, name in labels.items() if sid not in enrolled_ids}

print(f"Labels in file      : {len(labels)}")
print(f"With embeddings     : {len(enrolled_ids)}")
print(f"Without embeddings  : {len(ghosts)}")

if not ghosts:
    print("Nothing to clean.")
    raise SystemExit

print("\nThese labels have no face data:")
for sid in sorted(ghosts, key=int):
    print(f"  ID {sid}: {ghosts[sid]}")

answer = input("\nRemove these from the labels file? (yes/no): ").strip().lower()
if answer == "yes":
    for sid in ghosts:
        del labels[sid]
    student_data.save_student_labels(labels)
    print(f"Removed {len(ghosts)} labels. {len(labels)} students remain.")
else:
    print("Nothing changed.")
