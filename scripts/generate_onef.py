"""Generate transforms_onef.json for canonical hair training.

Extracts all camera views for a single timestep from transforms_train.json.
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--timestep", type=int, default=0, help="Which timestep to use as canonical")
    args = parser.parse_args()

    train_path = os.path.join(args.data_root, "transforms_train.json")
    with open(train_path) as f:
        data = json.load(f)

    # Filter frames for the target timestep
    onef_frames = [fr for fr in data["frames"] if fr["timestep_index"] == args.timestep]
    assert len(onef_frames) > 0, f"No frames found for timestep {args.timestep}"

    camera_indices = sorted(set(fr["camera_index"] for fr in onef_frames))

    onef = {
        "cx": data["cx"],
        "cy": data["cy"],
        "fl_x": data["fl_x"],
        "fl_y": data["fl_y"],
        "h": data["h"],
        "w": data["w"],
        "camera_angle_x": data["camera_angle_x"],
        "camera_angle_y": data["camera_angle_y"],
        "frames": onef_frames,
        "timestep_indices": [args.timestep],
        "camera_indices": camera_indices,
    }

    out_path = os.path.join(args.data_root, "transforms_onef.json")
    with open(out_path, "w") as f:
        json.dump(onef, f, indent=2)

    print(f"Wrote {out_path}: {len(onef_frames)} frames, "
          f"timestep={args.timestep}, cameras={camera_indices}")


if __name__ == "__main__":
    main()
