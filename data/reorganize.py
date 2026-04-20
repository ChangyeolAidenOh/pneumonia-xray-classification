"""
Reorganize the Kaggle Chest X-ray dataset from 2-class (NORMAL/PNEUMONIA)
to 3-class (NORMAL/BACTERIAL/VIRAL) by splitting PNEUMONIA based on filename.

Source: https://data.mendeley.com/datasets/rscbjbr9sj/2
"""

import os
import shutil
import argparse


def reorganize(src_root, dst_root):
    os.makedirs(dst_root, exist_ok=True)

    for split in ["train", "test"]:
        for cls in ["NORMAL", "BACTERIAL", "VIRAL"]:
            os.makedirs(os.path.join(dst_root, split, cls), exist_ok=True)

    for split in ["train", "test"]:
        # Copy NORMAL
        normal_src = os.path.join(src_root, split, "NORMAL")
        normal_dst = os.path.join(dst_root, split, "NORMAL")
        for fname in os.listdir(normal_src):
            shutil.copy(os.path.join(normal_src, fname),
                        os.path.join(normal_dst, fname))

        # Split PNEUMONIA into BACTERIAL / VIRAL
        pneumonia_src = os.path.join(src_root, split, "PNEUMONIA")
        for fname in os.listdir(pneumonia_src):
            if "bacteria" in fname.lower():
                dst = os.path.join(dst_root, split, "BACTERIAL", fname)
            elif "virus" in fname.lower():
                dst = os.path.join(dst_root, split, "VIRAL", fname)
            else:
                continue
            shutil.copy(os.path.join(pneumonia_src, fname), dst)

    for split in ["train", "test"]:
        counts = {}
        for cls in ["NORMAL", "BACTERIAL", "VIRAL"]:
            path = os.path.join(dst_root, split, cls)
            counts[cls] = len(os.listdir(path))
        print(f"{split}: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to original chest_xray directory")
    parser.add_argument("--dst", required=True, help="Output path for 3-class dataset")
    args = parser.parse_args()
    reorganize(args.src, args.dst)
