"""Preprocess depth files: convert from dense .npy (cm) to sparse .npz (meters).

Renames from camera_qRrq9DqY_vpCC_XXXX_depth.npy to TTTTT_CC.npz
where TTTTT = 5-digit timestep, CC = 2-digit camera id.

Filename index mapping: index = camera_index * 1003 + timestep_index
"""

import argparse
import os
import re

import numpy as np
import sparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--input_dir", type=str, default="depths",
                        help="Subdirectory with raw .npy depth files")
    parser.add_argument("--output_dir", type=str, default="depths_processed",
                        help="Output subdirectory for processed .npz files")
    parser.add_argument("--frames_per_camera", type=int, default=1003,
                        help="Total frames per camera (used for index→timestep mapping)")
    args = parser.parse_args()

    input_dir = os.path.join(args.data_root, args.input_dir)
    output_dir = os.path.join(args.data_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    pattern = re.compile(r"camera_\w+_vp(\d+)_(\d+)_depth\.npy")
    files = sorted(os.listdir(input_dir))

    converted = 0
    skipped = 0
    for fname in files:
        m = pattern.match(fname)
        if not m:
            skipped += 1
            continue

        camera_index = int(m.group(1))
        file_index = int(m.group(2))
        timestep_index = file_index - camera_index * args.frames_per_camera

        if timestep_index < 0:
            print(f"Warning: negative timestep for {fname}, skipping")
            skipped += 1
            continue

        # Load, convert cm → m
        depth_cm = np.load(os.path.join(input_dir, fname))
        depth_m = depth_cm / 100.0

        # Save as scipy sparse .npz
        out_name = f"{timestep_index:05d}_{camera_index:02d}.npz"
        depth_sparse = sparse.COO.from_numpy(depth_m.astype(np.float32))
        sparse.save_npz(os.path.join(output_dir, out_name), depth_sparse)
        converted += 1

        if converted % 1000 == 0:
            print(f"  Processed {converted} files...")

    print(f"Done: {converted} converted, {skipped} skipped")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
