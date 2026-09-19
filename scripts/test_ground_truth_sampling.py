"""
Test ground truth parser and clip generator using official CSV annotations.
"""
import os
import glob
from pathlib import Path
import pandas as pd
from collections import Counter

FALLS_CSV = "metadata/urfall-cam0-falls.csv"
ADLS_CSV = "metadata/urfall-cam0-adls.csv"
DATA_DIR = "Training data"

def load_ground_truth():
    df_falls = pd.read_csv(FALLS_CSV, header=None, names=['seq', 'frame', 'label', 'HW', 'MM', 'BBO', 'MaxStdXZ', 'HHmax', 'H', 'D', 'P40'])
    df_adls = pd.read_csv(ADLS_CSV, header=None, names=['seq', 'frame', 'label', 'HW', 'MM', 'BBO', 'MaxStdXZ', 'HHmax', 'H', 'D', 'P40'])

    # Build map: (seq_clean, frame_num) -> label
    # note: seq is 'fall-01', 'adl-01'
    gt_map = {}
    for _, row in df_falls.iterrows():
        gt_map[(row['seq'], int(row['frame']))] = int(row['label'])
    for _, row in df_adls.iterrows():
        gt_map[(row['seq'], int(row['frame']))] = int(row['label'])
    return gt_map

def parse_frame_num(filename):
    # Extracts number from fall-01-cam0-rgb-025.png -> 25
    stem = Path(filename).stem
    parts = stem.split('-')
    return int(parts[-1])

def get_seq_base(dir_name):
    # e.g. fall-01-cam0-rgb -> fall-01
    parts = dir_name.split('-')
    return f"{parts[0]}-{parts[1]}"

def test_sampling(window_size=8, strides=[1, 2], step=4):
    gt_map = load_ground_truth()
    seq_dirs = sorted([d for d in os.listdir(DATA_DIR) if (d.startswith("fall-") or d.startswith("adl-"))])
    print(f"Found {len(seq_dirs)} sequence directories in {DATA_DIR}.")

    clips = []
    for seq_name in seq_dirs:
        seq_path = os.path.join(DATA_DIR, seq_name)
        frames = sorted(glob.glob(os.path.join(seq_path, "*.png")))
        if len(frames) < window_size:
            continue
        base_name = get_seq_base(seq_name)
        is_adl_seq = base_name.startswith("adl")

        # Map each frame to its GT label
        frame_labels = []
        for f in frames:
            f_num = parse_frame_num(f)
            # Default to -1 if frame missing in GT
            lbl = gt_map.get((base_name, f_num), -1)
            frame_labels.append(lbl)

        # Multi-stride temporal sampling
        for stride in strides:
            span = (window_size - 1) * stride + 1
            for start_idx in range(0, len(frames) - span + 1, step):
                clip_indices = [start_idx + i * stride for i in range(window_size)]
                clip_lbls = [frame_labels[i] for i in clip_indices]

                if is_adl_seq:
                    # In ADL sequences, every action (including lying on sofa/bed) is Non-Fall (0)
                    clip_label = 0
                else:
                    # In Fall sequences:
                    # If the clip covers falling transition (0) or mostly fallen posture (1)
                    has_transition = any(l == 0 for l in clip_lbls)
                    fallen_count = sum(1 for l in clip_lbls if l == 1)
                    if has_transition or fallen_count >= 3:
                        clip_label = 1
                    else:
                        clip_label = 0

                clips.append((seq_name, clip_indices, clip_label))

    counts = Counter(c[2] for c in clips)
    print(f"Generated {len(clips)} clips: {counts[0]} Non-Fall (Class 0), {counts[1]} Fall (Class 1)")
    print(f"Class ratio: {counts[1] / (counts[0] + 1e-6):.2f}")

if __name__ == "__main__":
    test_sampling()
