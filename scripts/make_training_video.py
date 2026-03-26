#!/usr/bin/env python3
"""Create a video from training frames listed in a transforms JSON file.

Reads the transforms JSON, filters for a single camera, sorts by timestep,
symlinks frames into a temp directory with sequential numbering, and encodes
an H.264 MP4 via ffmpeg.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description="Create video from training frames")
    parser.add_argument(
        "--transforms", required=True,
        help="Path to transforms_train.json (or any transforms JSON)",
    )
    parser.add_argument(
        "--camera_index", type=int, default=0,
        help="Camera index to select (default: 0)",
    )
    parser.add_argument("--fps", type=int, default=25, help="Output frame rate (default: 25)")
    parser.add_argument(
        "--output", default=None,
        help="Output MP4 path. Defaults to <transforms_dir>/train_cam<XX>_video.mp4",
    )
    args = parser.parse_args()

    transforms_path = os.path.abspath(args.transforms)
    transforms_dir = os.path.dirname(transforms_path)

    with open(transforms_path) as f:
        data = json.load(f)

    frames = [fr for fr in data["frames"] if fr["camera_index"] == args.camera_index]
    if not frames:
        print(f"Error: no frames found for camera_index={args.camera_index}", file=sys.stderr)
        sys.exit(1)

    frames.sort(key=lambda fr: fr["timestep_index"])
    print(f"Selected {len(frames)} frames for camera {args.camera_index} "
          f"(timesteps {frames[0]['timestep_index']}-{frames[-1]['timestep_index']})")

    output_path = args.output or os.path.join(
        transforms_dir, f"train_cam{args.camera_index:02d}_video.mp4"
    )
    output_path = os.path.abspath(output_path)

    tmpdir = tempfile.mkdtemp(prefix="training_video_")
    try:
        for i, fr in enumerate(frames):
            src = os.path.normpath(os.path.join(transforms_dir, fr["file_path"]))
            if not os.path.isfile(src):
                print(f"Warning: missing frame {src}", file=sys.stderr)
                continue
            dst = os.path.join(tmpdir, f"{i:05d}.png")
            os.symlink(src, dst)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", os.path.join(tmpdir, "%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path,
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"Video saved to {output_path}")
    finally:
        for name in os.listdir(tmpdir):
            os.unlink(os.path.join(tmpdir, name))
        os.rmdir(tmpdir)


if __name__ == "__main__":
    main()
