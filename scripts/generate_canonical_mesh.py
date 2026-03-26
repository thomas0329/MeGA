"""Generate canonical FLAME mesh .obj from FLAME parameters.

Loads FlameHead with subdivision, runs a forward pass with the given
FLAME params, and exports the resulting mesh as an .obj file.

Usage:
    python scripts/generate_canonical_mesh.py \
        --flame_param subject_seq_whiteBg/flame_param/00000.npz \
        --output subject_seq_whiteBg/canonical_mesh.obj
"""

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from networks.meshface.flame2023.flame import FlameHead


FLAME_CFG = {
    "flame.model_path": "face-data/flame2023/flame2023.pkl",
    "flame.lmk_embedding_path": "face-data/flame2023/landmark_embedding_with_eyes.npy",
    "flame.parts_path": "face-data/flame/FLAME_masks_mouth.pkl",
    "flame.ignore_faces": "face-data/flame/lower_neck_bottom_face_idcs.npy",
    "flame.n_shape": 300,
    "flame.n_expr": 100,
    "flame.n_pose": 15,
    "flame.subdivision": 1,
    "flame.subdivision.mouth": 0,
    "flame.add_teeth": True,
    "flame.enable_offsets": True,
    "flame.move_eyes": False,
    "flame.offsets_ignore_parts": [],
}


def write_obj(filepath, verts, faces):
    with open(filepath, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    print(f"Wrote {filepath}: {len(verts)} vertices, {len(faces)} faces")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flame_param", type=str, required=True,
                        help="Path to FLAME parameter .npz file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .obj path")
    args = parser.parse_args()

    flame = FlameHead(FLAME_CFG).cuda()

    params = dict(np.load(args.flame_param, allow_pickle=True))
    shape = torch.from_numpy(params["shape"][None] if params["shape"].ndim == 1
                             else params["shape"]).float().cuda()
    expr = torch.from_numpy(params["expr"] if params["expr"].ndim == 2
                            else params["expr"][None]).float().cuda()
    rotation = torch.from_numpy(params["rotation"]).float().cuda()
    neck = torch.from_numpy(params["neck_pose"]).float().cuda()
    jaw = torch.from_numpy(params["jaw_pose"]).float().cuda()
    eyes = torch.from_numpy(params["eyes_pose"]).float().cuda()
    translation = torch.from_numpy(params["translation"]).float().cuda()

    with torch.no_grad():
        verts, _ = flame(
            shape=shape, expr=expr, rotation=rotation,
            neck=neck, jaw=jaw, eyes=eyes, translation=translation,
            zero_centered_at_root_node=False, return_landmarks=True,
            static_offset=None,
        )

    verts_np = verts[0].cpu().numpy()
    faces_np = flame.faces.cpu().numpy() + 1  # OBJ is 1-indexed

    write_obj(args.output, verts_np, faces_np)


if __name__ == "__main__":
    main()
