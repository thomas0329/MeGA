"""Generate parsing labels using SegFace (AAAI 2025, SOTA on CelebAMask-HQ).

Runs SegFace inference on all images, then remaps output labels from SegFace's
ordering to the standard CelebAMask-HQ ordering expected by MeGA.

Usage:
    python scripts/generate_parsing.py \
        --data_root subject_seq_whiteBg_multi-frame \
        --batch_size 8
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

# Patch missing torchvision imports (mega env has older torchvision without swin_v2)
import torchvision.models as _tv_models
for _name in ["swin_v2_b", "swin_v2_s", "swin_v2_t"]:
    if not hasattr(_tv_models, _name):
        setattr(_tv_models, _name, None)

# Import SegFaceCeleb directly (avoid importing LaPa/Helen)
import importlib.util
_segface_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "submodules", "SegFace")
sys.path.insert(0, _segface_root)
from network.models.segface_celeb import SegFaceCeleb

# SegFace output ordering:
#   0:bg, 1:neck, 2:skin, 3:cloth, 4:l_ear, 5:r_ear, 6:l_brow, 7:r_brow,
#   8:l_eye, 9:r_eye, 10:nose, 11:mouth, 12:l_lip, 13:u_lip, 14:hair,
#   15:eye_g, 16:hat, 17:ear_r, 18:neck_l
#
# MeGA expected ordering (standard CelebAMask-HQ):
#   0:bg, 1:skin, 2:nose, 3:eye_g, 4:l_eye, 5:r_eye, 6:l_brow, 7:r_brow,
#   8:l_ear, 9:r_ear, 10:mouth, 11:u_lip, 12:l_lip, 13:hair, 14:hat,
#   15:ear_r, 16:neck_l, 17:neck, 18:cloth
SEGFACE_TO_MEGA = np.array(
    [0, 17, 1, 18, 8, 9, 6, 7, 4, 5, 2, 10, 12, 11, 13, 3, 14, 15, 16],
    dtype=np.uint8,
)

WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "submodules", "SegFace", "weights", "swinb_celeba_512", "model_299.pt",
)


def load_model():
    """Load SegFace Swin-B model with pretrained CelebAMask-HQ weights."""
    model = SegFaceCeleb(512, "swin_base").cuda()
    model.eval()
    ckpt = torch.load(WEIGHTS_PATH, map_location="cuda")
    model.load_state_dict(ckpt["state_dict_backbone"])
    print(f"Loaded SegFace weights from {WEIGHTS_PATH}")
    return model


def make_transform():
    """Build the preprocessing transform matching SegFace training."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    img_dir = os.path.join(args.data_root, "images")
    out_dir = os.path.join(args.data_root, "parsing")
    os.makedirs(out_dir, exist_ok=True)

    # Collect all images, skip already-processed ones
    all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    to_process = []
    for img_path in all_imgs:
        name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, f"{name}_labels.png")
        if not os.path.exists(out_path):
            to_process.append((img_path, name, out_path))

    print(f"Total images: {len(all_imgs)}, to process: {len(to_process)}, "
          f"skipping: {len(all_imgs) - len(to_process)}")

    if not to_process:
        print("Nothing to do.")
        return

    model = load_model()
    transform = make_transform()

    # Process in batches
    for i in tqdm(range(0, len(to_process), args.batch_size),
                  desc="Generating parsing labels"):
        batch_items = to_process[i:i + args.batch_size]
        batch_tensors = []

        for img_path, _, _ in batch_items:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (512, 512):
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            batch_tensors.append(transform(img))

        batch = torch.stack(batch_tensors).cuda()

        with torch.no_grad():
            seg_output = model(batch, {}, torch.zeros(len(batch_items), dtype=torch.long).cuda())
            seg_output = F.interpolate(seg_output, size=(512, 512),
                                       mode="bilinear", align_corners=False)
            preds = seg_output.argmax(dim=1).cpu().numpy()  # (B, H, W)

        for j, (_, name, out_path) in enumerate(batch_items):
            labels = SEGFACE_TO_MEGA[preds[j]]
            cv2.imwrite(out_path, labels)

    print(f"Done. Output: {out_dir}/ ({len(to_process)} files)")


if __name__ == "__main__":
    main()
