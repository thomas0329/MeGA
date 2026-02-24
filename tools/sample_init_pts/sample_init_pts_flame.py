"""
Sample init_pts_150000.npy using the runtime FlameHead model.

This uses the actual subdivided FLAME mesh (~16,428 vertices) with correct
scalp indices, instead of a pre-exported base OBJ (~5,100 vertices) where
the runtime scalp indices don't match.
"""

import sys
import os

# Run from repo root so FlameHead's asset paths resolve correctly
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ".")

import numpy as np
import point_cloud_utils as pcu
import torch

from networks.meshface.flame2023.flame import FlameHead


def faces_of_verts(vert_idcs, all_faces):
    """Return faces where ALL vertices are in vert_idcs."""
    vert_set = set(vert_idcs.tolist()) if hasattr(vert_idcs, "tolist") else set(vert_idcs)
    mask = np.array([all(int(v) in vert_set for v in f) for f in all_faces])
    return all_faces[mask]


def write_obj(filepath, verts, tris=None):
    with open(filepath, "w") as fw:
        for v in verts:
            fw.write(f"v {v[0]} {v[1]} {v[2]}\n")
        if tris is not None:
            for t in tris:
                fw.write(f"f {t[0]} {t[1]} {t[2]}\n")
    print(f"Saved {filepath}")


def main():
    # ---- Config matching static_hair.yaml ----
    config = {
        "flame.subdivision": 1,
        "flame.subdivision.mouth": 0,
        "flame.add_teeth": True,
        "flame.model_path": "face-data/flame2023/flame2023.pkl",
        "flame.lmk_embedding_path": "face-data/flame2023/landmark_embedding_with_eyes.npy",
        "flame.parts_path": "face-data/flame/FLAME_masks_mouth.pkl",
        "flame.ignore_faces": "face-data/flame/lower_neck_bottom_face_idcs.npy",
        "flame.n_shape": 300,
        "flame.n_expr": 100,
        "flame.n_pose": 15,
    }

    # ---- Instantiate FlameHead ----
    print("Building FlameHead model...")
    model = FlameHead(config)
    model.eval()

    print(f"  v_template: {model.v_template.shape}")  # expect ~(16428, 3) + teeth
    print(f"  faces: {model.faces.shape}")

    # ---- Get runtime scalp indices (correct for subdivided mesh) ----
    scalp_idcs = model._parts["scalp"]
    print(f"  scalp indices: {scalp_idcs.shape[0]} vertices, max index={scalp_idcs.max().item()}")

    # ---- Forward pass with canonical FLAME params ----
    flame_param_path = "subject_seq_whiteBg/flame_param/00000.npz"
    print(f"Loading canonical FLAME params from {flame_param_path}")
    fp = np.load(flame_param_path)

    with torch.no_grad():
        shape = torch.tensor(fp["shape"]).unsqueeze(0).float()       # (1, 300)
        expr = torch.tensor(fp["expr"]).float()                       # (1, 100)
        rotation = torch.tensor(fp["rotation"]).float()               # (1, 3)
        neck = torch.tensor(fp["neck_pose"]).float()                  # (1, 3)
        jaw = torch.tensor(fp["jaw_pose"]).float()                    # (1, 3)
        eyes = torch.tensor(fp["eyes_pose"]).float()                  # (1, 6)
        translation = torch.tensor(fp["translation"]).float()         # (1, 3)

        verts_posed, landmarks = model(
            shape, expr, rotation, neck, jaw, eyes, translation,
            return_landmarks=True,
        )
    verts = verts_posed[0].numpy()  # (V, 3)
    faces = model.faces.numpy()     # (F, 3)
    print(f"  Posed vertices: {verts.shape}, faces: {faces.shape}")

    # ---- Sample init points on scalp faces ----
    scalp_idcs_np = scalp_idcs.numpy()
    scalp_faces = faces_of_verts(scalp_idcs_np, faces)
    print(f"  Scalp faces: {scalp_faces.shape[0]}")

    num_surf = 50000
    num_off = 100000
    sigma = 0.02

    # Compute vertex normals
    vert_normals = pcu.estimate_mesh_vertex_normals(verts, faces)

    # Sample on-surface points
    f_i, bc = pcu.sample_mesh_random(verts, scalp_faces, num_samples=num_surf)
    surf_pts = pcu.interpolate_barycentric_coords(scalp_faces, f_i, bc, verts)
    surf_normals = pcu.interpolate_barycentric_coords(scalp_faces, f_i, bc, vert_normals)
    print(f"  Surface points: {surf_pts.shape}")

    # Sample off-surface points along normals
    rnd_idx = np.random.randint(0, surf_pts.shape[0], num_off)
    off_pts = surf_pts[rnd_idx] + surf_normals[rnd_idx] * np.random.rand(num_off, 3) * sigma
    print(f"  Off-surface points: {off_pts.shape}")

    # Concatenate
    sampled_pts = np.concatenate([surf_pts, off_pts], axis=0)
    print(f"  Total: {sampled_pts.shape}")

    # ---- Save ----
    outpath = "subject_seq_whiteBg/init_pts_150000.npy"
    np.save(outpath, sampled_pts)
    print(f"Saved {outpath}")

    # Also save OBJ for visualization
    obj_path = "tools/sample_init_pts/init_pts_150000_flame.obj"
    write_obj(obj_path, sampled_pts)

    # Save the scalp mesh for verification
    scalp_obj_path = "tools/sample_init_pts/scalp_mesh_flame.obj"
    write_obj(scalp_obj_path, verts, scalp_faces + 1)  # OBJ is 1-indexed


if __name__ == "__main__":
    main()
