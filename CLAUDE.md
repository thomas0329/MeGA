# MeGA: Hybrid Mesh-Gaussian Head Avatar (CVPR 2025)

Hybrid head avatar combining a FLAME-based mesh face branch with a 3D Gaussian Splatting hair branch. The mesh renders the face via nvdiffrast, Gaussians model hair via diff-gauss, and an occlusion-aware depth-comparison fuses both at render time.

Paper: https://arxiv.org/abs/2404.19026

## Environment

- Python 3.9, conda env named `mega`
- PyTorch 1.12.1, CUDA 11.3, PyTorch3D
- numpy must be pinned to 1.23.1 (`pip install numpy==1.23.1`) after other deps
- Custom CUDA submodules installed via pip: `submodules/diff-gauss`, `submodules/nvdiffrast`, `submodules/simple-knn`
- Other deps: opencv-python, lpips, kornia, tensorboard, sparse, trimesh, roma, chumpy, ninja
- Setup: `conda activate mega && ./create_env.sh`

## Data Layout

Data lives under a root like `nersemble/preprocess/<SUBJECT_ID>/...`. Each subject folder contains:
- `transforms_{train,val,test,onef}.json` — camera/frame metadata
- `flame_param/` — per-frame FLAME parameters (.npz)
- `images/`, `parsing/`, `depth/` — GT images, face parsing labels, depth maps
- `init_pts_150000.npy` — initial Gaussian point cloud for hair

FLAME model files go in `face-data/flame2023/` and `face-data/flame/`.

## Training Pipeline

Training is a **two-step** process:

### Step 1: Train canonical hair (HairTrainer)
```bash
bash ./scripts/train_hair.sh
```
- Config: `configs/nersemble/<ID>/hair.yaml`
- Key: `pipe.neutral_hair: True` selects `HairTrainer` and uses `split="onef"` (single-frame canonical)
- `gs.enable_reset: True` — saves `checkpoint_reset.pth` on opacity reset
- Runs for many epochs (e.g. 8000) with `data.load_images: True`
- Output: `$WORKSPACE/$VERSION/checkpoint_reset.pth`

### Step 2: Train full avatar (JointTrainer)
```bash
bash ./scripts/train_full.sh
```
- Config: `configs/nersemble/<ID>/full.yaml`
- Key: `pipe.neutral_hair: False` selects `JointTrainer`
- `gs.pretrain` must point to Step 1's `checkpoint_reset.pth`
- Two training stages controlled by `training.stages: [head, joint]` and `training.stages_epoch: [50]`:
  - **head stage** (epochs 1-50): freezes all, unfreezes `head_*` params. Learns face mesh geometry/texture. Loss: `training.head_stage_loss: [head]`
  - **joint stage** (epochs 51+): freezes all, unfreezes `hair` + `head_tex*` params. Loads pretrained canonical hair, learns deformation field + refines head texture. Loss: `training.joint_stage_loss: [hair, joint]`
- `data.load_images: False` — images loaded on-the-fly
- Output: `$WORKSPACE/$VERSION/checkpoint_latest.pth`

### Config Overrides
`train.py` accepts `--extra_config '{"key": value}'` to override any YAML key at launch. The merged config is saved as `params_tmp.yaml` beside the original.

### Bald subjects
Subject 218 is auto-detected as bald (`config["bald"]` set from config path). Bald subjects skip the `gs.pretrain` requirement.

## Evaluation & Rendering

### Compute metrics
```bash
bash ./scripts/metrics.sh
```
Runs `metrics.py --checkpoint <path> --split {test,val,train,onef}`. Renders all frames, then computes PSNR, SSIM, LPIPS (AlexNet), and depth error (mm). Results saved to `<checkpoint_dir>/<split>_eval/metrics.txt`.

Flags: `--skip_render` (metrics only), `--skip_metric` (render only), `--time` (profile), `-d` (debug: save hair masks).

### Render sequences
```bash
bash ./scripts/render.sh
```
Uses `funny_demo/render_funny.py`. Requires `--checkpoint`, `--motionfile` (transforms JSON), `--name`.

### Generate video
```bash
bash ./scripts/img2video.sh <renders_dir>
```
Converts rendered frames to MP4 via ffmpeg.

## Architecture

### Entry point: `train.py`
- Selects trainer class: `HairTrainer` if `pipe.neutral_hair` else `JointTrainer`
- Loads `NeRSembleData` dataset, creates DataLoaders
- Seed fixed at 42

### Face branch: `networks/meshface/facewrapper.py` → `MeshFaceWrapper`
Models:
- `neural_texture` — learnable UV texture map (1024x1024, 12ch)
- `dynamic_texture` — expression-conditioned offset (`DynamicDecoder`)
- `view_texture` — view-conditioned offset (`ViewDecoder`)
- `disp_decoder` — displacement map + eye transforms (`DispDecoder`)
- `pe` — positional encoding (LPE or PE)
- `head_mlp` — pixel decoder: neural features → RGB (`PixelDecoder`)
- `FlameHead` — FLAME 2023 mesh model with subdivision, teeth, offsets
- `NVDiffRenderer` — nvdiffrast-based rasterizer

Render pipeline: neural texture → FLAME mesh with learned offsets → rasterize → sample texture at UV coords → decode to RGB.

### Hair branch: `networks/gshair/hairwrapper.py` → `GSHairWrapper`
Models:
- `canonical_gs` — `GaussianModel` initialized from `init_pts` point cloud
- `deform_mlp` — expression-conditioned MLP producing per-Gaussian offsets for xyz, rotation, scaling, opacity, features_dc

Render pipeline: canonical Gaussians → rigid transform (from scalp registration) → non-rigid deform (MLP conditioned on expression) → diff-gauss splatting.

AIAP (As-Isometric-As-Possible) regularization on deformed Gaussians when `gs.enable_aiap: True`.

### Fusion: `fuse()` method in both trainers
Depth-based occlusion test: compare hair depth vs head depth per pixel. Hair pixels closer to camera blend over face. During validation, morphological erosion+dilation cleans the mask boundary.

### Data: `dataset/nersemble.py` → `NeRSembleData`
When `pipe.neutral_hair: True`, forces `split="onef"` to train on a single canonical frame.

## Loss Functions

### Face losses (`MeshFaceWrapper.compute_losses`)
- `rgb.head` — L2 on rendered face vs GT (hair region masked out)
- `ssim` — 1 - SSIM on face
- `depth.head` — L1 depth with threshold filtering
- `normal.head` — L2 on depth-derived normals
- `mesh.laplacian` — cotangent Laplacian smoothing
- `mesh.normal` — mesh normal consistency
- `mesh.edges` — edge length regularization
- `mesh.verts_scale` — scalp shrinkage toward center (prevents scalp poking through hair)
- DiPho loss: 3x weighted L2+SSIM on basic (static) texture rendering

### Hair losses (`GSHairWrapper.compute_losses`)
- `rgb.hair` — L2 on rendered hair vs GT (non-hair masked to white)
- `ssim` — 1 - SSIM on hair
- `silh.hair` — chamfer-distance-weighted silhouette loss
- `silh.solid_hair` — penalizes transparent Gaussians inside eroded hair mask
- Per-attribute regularization on deformation offsets (rotation, scaling, opacity, features_dc)
- `aiap.xyz`, `aiap.cov` — AIAP losses

### Joint losses (computed in trainer)
- `rgb` — L2 on fused render vs full GT image
- `ssim` — 1 - SSIM on fused render
- `rgb.head` + `ssim` on whole-head render (JointTrainer only)

Which losses are active depends on the current stage via `training.<stage>_stage_loss`.

## Configuration Reference

Configs in `configs/nersemble/<ID>/hair.yaml` and `full.yaml`. Key parameters:

| Key | Description |
|-----|-------------|
| `pipe.neutral_hair` | `True` → HairTrainer, `False` → JointTrainer |
| `gs.pretrain` | Path to pretrained hair checkpoint (required for full training) |
| `gs.init_pts` | Path to initial point cloud .npy |
| `training.stages` | List of stage names, e.g. `[head, joint]` |
| `training.stages_epoch` | Epoch boundaries between stages |
| `training.<stage>_stage_loss` | Which loss groups are active per stage |
| `training.epochs` | Total epochs |
| `training.eval_interval` | Steps between validation runs |
| `training.visual_interval` | Steps between visualization saves |
| `flame.optimize_params` | Whether to jointly optimize FLAME params |
| `flame.subdivision` | Mesh subdivision level |
| `gs.enable_aiap` | Enable AIAP regularization |
| `gs.densify_until_iter` | Stop densification after this step |
| `gs.enable_reset` | Enable opacity reset (saves checkpoint_reset.pth) |
| `training.lambda_*` | Loss weights (see Loss Functions section) |

## Editing Features

### Hair alteration
```bash
bash ./scripts/alter_hair.sh
```
Swaps one subject's hair onto another's head via `funny_demo/render_funny.py --hair <hair_ckpt>`. Uses `--with_pose` for same-identity, omit for cross-identity.

### Texture painting
```bash
bash ./scripts/paint.sh
```
Optimizes neural texture to match 2D painted images via `funny_demo/paint_opt.py`. Uses a `painting` stage that freezes everything except `head_tex_basic` and `head_tex_mlp`.

## Checkpoints

Saved at `$WORKSPACE/$VERSION/`:
- `checkpoint_latest.pth` — saved every 2000 steps
- `checkpoint_reset.pth` — saved on opacity reset (hair training)
- `checkpoint_<stage>_it<step>.pth` — saved at stage transitions
- `checkpoint_<step>.pth` — saved at eval intervals
- `flame_params.npz` — optimized FLAME parameters
- `params.yaml` — config snapshot
- `training.log` — full training log
- TensorBoard logs in the same directory

## Custom CUDA Submodules

| Submodule | Purpose |
|-----------|---------|
| `submodules/diff-gauss` | Differentiable Gaussian splatting rasterizer |
| `submodules/nvdiffrast` | NVIDIA differentiable rasterization for mesh rendering |
| `submodules/simple-knn` | Fast KNN for Gaussian neighbor queries (AIAP loss) |

Install all three: `pip install submodules/diff-gauss submodules/nvdiffrast submodules/simple-knn`

## Key File Paths

| File | Purpose |
|------|---------|
| `train.py` | Training entry point |
| `metrics.py` | Evaluation entry point |
| `joint_trainer.py` | Full avatar trainer (face + hair) |
| `hair_trainer.py` | Hair-only canonical trainer |
| `networks/meshface/facewrapper.py` | Face branch wrapper |
| `networks/gshair/hairwrapper.py` | Hair branch wrapper |
| `networks/meshface/flame2023/flame.py` | FLAME head model |
| `networks/gshair/gs/gaussian_model.py` | Gaussian model with densification |
| `networks/gshair/gs/deformation.py` | Expression-conditioned deformation MLP |
| `networks/meshface/mesh_renderer/mesh_renderer.py` | nvdiffrast mesh renderer |
| `networks/meshface/textures/neural_texture.py` | Learnable UV texture |
| `networks/meshface/neural2rgb/pixel_decoder.py` | Neural features → RGB decoder |
| `dataset/nersemble.py` | NeRSemble dataset loader |
| `dataset/cameras.py` | Camera class |
| `utils.py` | Utilities (SSIM, AIAP loss, checkpoint restore, etc.) |
| `funny_demo/render_funny.py` | Rendering/editing demo script |
| `funny_demo/paint_opt.py` | Texture painting optimization |
