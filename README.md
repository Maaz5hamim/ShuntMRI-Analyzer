# ShuntMRI-Analyzer

A deep-learning pipeline for analyzing brain MRI scans of patients with **ventriculo-peritoneal (VP) shunts** — devices implanted to treat hydrocephalus. The shunt hardware creates susceptibility artifacts in MRI that obscure the catheter path. This project automates three tasks from raw MRI to a clinically useful catheter length estimate.

---

## Tasks

| Task | Output | Description |
|------|--------|-------------|
| **A** | `subj*_taskA.nii.gz` | Binary mask of susceptibility **artifact** caused by the shunt hardware |
| **B** | `subj*_taskB.nii.gz` | Binary mask of the **catheter** path through brain tissue |
| **C** | `taskC.csv` | Estimated **total catheter length** in millimeters (visible + hidden under artifact) |

---

## Pipeline Architecture

```
Raw T1 MRI (.nii.gz)
        │
        ▼
┌─────────────────────────┐
│     Preprocessing        │
│  - N4 bias correction    │
│  - Register to MNI 152   │  (ANTs SyN — affine + deformable warp)
│  - Min-max normalisation │
│  - Unsharp mask enhance  │
└──────────┬──────────────┘
           │  registered image (MNI space)
           ▼
┌──────────────────────────────────────────┐
│         Task A — Artifact Segmentation    │
│                                           │
│  Stage 1: Heuristic Anchor Detection     │
│    AxialAnchor — scan all axial slices,  │
│    find best slice where shunt artifact  │
│    "dents" the brain convex hull         │
│            │                             │
│  Stage 2: 3-D Tracker                   │
│    Tracker — propagate artifact mask     │
│    up and down from anchor slice using   │
│    search-radius region growing          │
│            │                             │
│  Stage 3: U-Net Refinement (MONAI UNet)  │
│    Input: [registered image | coarse     │
│    artifact estimate] — 2-channel 96³   │
│    volume cropped around ROI             │
│    Output: refined artifact mask         │
│            │                             │
│  Inverse warp back to subject space      │
└──────────┬───────────────────────────────┘
           │  artifact mask (subject space)
           ▼
┌──────────────────────────────────────────┐
│         Task B — Catheter Segmentation   │
│                                           │
│  Custom 3-D U-Net (sliding-window        │
│  inference, 64³ patches, stride 32)      │
│  Input: registered T1 image              │
│  Output: binary catheter mask            │
└──────────┬───────────────────────────────┘
           │  catheter mask + artifact mask
           ▼
┌──────────────────────────────────────────┐
│      Task C — Length Estimation          │
│                                           │
│  1. Binary closing to fill catheter gaps │
│  2. Spine extraction via MST on point    │
│     cloud (networkx), endpoints = graph  │
│     diameter                             │
│  3. Artifact-side tip detection →        │
│     entry point where catheter enters    │
│     artifact region                      │
│  4. Spline smoothing (scipy splprep)     │
│     → arc length of visible segment      │
│  5. Feature extraction (curvature,       │
│     artifact volume, extent, etc.)       │
│  6. ML regressor (scikit-learn, joblib)  │
│     predicts hidden length under artifact│
│  Output: visible_len + hidden_est (mm)  │
└──────────────────────────────────────────┘
```

---

## Project Structure

```
ShuntMRI-Analyzer/
├── preprocess.py                  # ANTs registration + N4 + normalisation
├── predict.py                     # Full inference entry point
│
├── artifact_segmentation/         # Task A
│   ├── anchor.py                  # AxialAnchor — heuristic slice detection
│   ├── tracker.py                 # Tracker — 3-D artifact propagation
│   └── pipeline.py                # Batch preprocessing + U-Net dataset builder
│
├── catheter_segmentation/         # Task B
│   └── unet.py                    # UNet3D — custom 3-D U-Net architecture
│
├── catheter_length_estimation/    # Task C
│   └── estimate_length.py         # Spine extraction + ML length estimator
│
├── models/                        # Trained weights (Git LFS)
│   ├── artifact_seg_model.pth     # MONAI UNet — artifact segmentation
│   ├── catheter_seg_model.pth     # UNet3D — catheter segmentation
│   ├── catheter_hidden_model.pkl  # scikit-learn regressor — hidden length
│   └── checkpoints/
│       └── epoch25.pth            # Intermediate training checkpoint
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_artifact_segmentation.ipynb
│   ├── 03b_artifact_exploration.ipynb
│   ├── 04_catheter_segmentation.ipynb
│   ├── 04b_catheter_seg_v2.ipynb
│   ├── 05_length_estimation.ipynb
│   └── 06_full_pipeline.ipynb
│
├── templates/
│   └── mni_icbm152_nlin_asym_09c/   # MNI ICBM 152 atlas (ANTs template)
│       └── mni_icbm152_t1_tal_nlin_asym_09c.nii
│
├── data/
│   ├── Train/
│   │   └── subj{001..N}/
│   │       ├── subj{ID}_image.nii.gz        # T1 MRI
│   │       ├── subj{ID}_artifact.nii.gz     # Ground truth artifact mask
│   │       ├── subj{ID}_catheter.nii.gz     # Ground truth catheter mask
│   │       ├── reg_0GenericAffine.mat       # ANTs affine transform
│   │       └── reg_1{Inverse}Warp.nii.gz   # ANTs deformable warp
│   └── Test/
│       └── subj{001..N}/
│           ├── subj{ID}_image.nii.gz
│           ├── reg_0GenericAffine.mat
│           └── reg_1{Inverse}Warp.nii.gz
│
├── output/                        # Inference outputs
│   ├── subj{ID}_taskA.nii.gz
│   ├── subj{ID}_taskB.nii.gz
│   └── taskC.csv
│
└── requirements.txt
```

---

## Models

### Task A — Artifact Segmentation (MONAI UNet)

A two-stage approach:

**Stage 1 — Heuristic anchor (`anchor.py`)**

For every axial slice, `AxialAnchor`:
1. Otsu-thresholds + morphologically cleans the slice to extract the brain boundary
2. Computes the convex hull and identifies "dents" — regions inside the hull but outside the brain
3. Scores each dent by `area × circularity` (shunt artifacts are compact round voids)
4. Returns the slice with the highest-scoring dent as the anchor

**Stage 2 — 3-D tracker (`tracker.py`)**

`Tracker` walks outward (superior and inferior) from the anchor slice using a configurable `search_radius`, propagating the artifact mask through the volume and smoothing the result.

**Stage 3 — MONAI UNet refinement**

Architecture: 3-D U-Net (`channels=(32,64,128,256)`, 3 strides, 2 residual units per block)
- Input: 2-channel 96³ crop — `[MRI | coarse artifact estimate]`
- Output: refined binary artifact mask
- Inference: crop around tracker ROI → resize to 96³ → sigmoid threshold 0.5 → resize back → paste into full volume
- Inverse ANTs warp returns mask to subject space

### Task B — Catheter Segmentation (UNet3D)

Custom lightweight 3-D U-Net (`unet.py`):

```
Encoder:  1 → b → 2b → 4b → 8b   (b = base_channels = 16)
Bottleneck:       8b → 16b
Decoder:  16b → 8b → 4b → 2b → b → 1
```

Each encoder/decoder block: `Conv3d → InstanceNorm3d → ReLU → Conv3d → InstanceNorm3d → ReLU`

Inference: sliding-window over the full registered volume (64³ patches, stride 32), predictions averaged in overlapping regions.

### Task C — Length Estimation

The catheter is partially obscured by the susceptibility artifact, so its full length cannot be read directly from the segmentation mask. The estimator:

1. **Spine extraction** — converts the 3-D mask to physical-space coordinates (mm), downsamples to 2 mm steps, builds a distance-based adjacency graph, extracts the minimum spanning tree, and finds the longest path (graph diameter) as the catheter spine
2. **Entry point** — identifies where the spine crosses into the artifact region to separate visible from hidden segments
3. **Feature engineering** — visible length, curvature stats, artifact volume, artifact extent along catheter direction, endpoint separation
4. **ML regressor** — a scikit-learn model (serialised via joblib) predicts the hidden segment length; total = visible + predicted hidden

---

## Setup

```bash
pip install -r requirements.txt
```

> **ANTs**: `antspyx` is the Python binding for ANTs registration. On some systems it must be installed separately:
> ```bash
> pip install antspyx
> ```

Model weights use **Git LFS** — make sure `git-lfs` is installed and run `git lfs pull` after cloning.

---

## Running Inference

```bash
python predict.py
```

The script reads from `data/Train/` by default (edit `Config` at the top of `predict.py` to point at `data/Test/`). For each subject it:

1. Registers the T1 image to MNI space (saves ANTs transforms to `data/Train/subj{ID}/`)
2. Runs the heuristic anchor + tracker to produce a coarse artifact estimate
3. Refines with the artifact U-Net → saves `output/subj{ID}_taskA.nii.gz`
4. *(Commented out)* Segments the catheter → saves `output/subj{ID}_taskB.nii.gz`
5. *(Commented out)* Estimates catheter length → appends to `output/taskC.csv`

To enable Task B and C, uncomment the relevant blocks in `predict.py`.

---

## Training

Training code lives in the notebooks:

| Notebook | Task |
|----------|------|
| `notebooks/03_artifact_segmentation.ipynb` | Task A — U-Net training |
| `notebooks/04_catheter_segmentation.ipynb` | Task B — U-Net training |
| `notebooks/05_length_estimation.ipynb` | Task C — feature extraction + regressor training |

`artifact_segmentation/pipeline.py` handles batch preprocessing and dataset creation (preprocessing all training subjects and packaging them as 4-D NIfTI volumes for U-Net training):

```bash
cd artifact_segmentation
python pipeline.py
```

---

## Data Format

Each training subject directory contains:

| File | Description |
|------|-------------|
| `subj{ID}_image.nii.gz` | T1-weighted MRI (variable FOV and spacing) |
| `subj{ID}_artifact.nii.gz` | Binary mask — susceptibility artifact |
| `subj{ID}_catheter.nii.gz` | Binary mask — catheter path |
| `reg_0GenericAffine.mat` | ANTs affine transform (subject → MNI) |
| `reg_1Warp.nii.gz` | ANTs forward deformable warp |
| `reg_1InverseWarp.nii.gz` | ANTs inverse warp (MNI → subject) |

Test subjects have the image and registration files only (no ground truth masks).

---

## Dependencies

| Package | Role |
|---------|------|
| `antspyx` | MRI registration (affine + SyN), N4 bias correction |
| `SimpleITK` | N4 bias field correction (fallback implementation) |
| `nibabel` | NIfTI file I/O |
| `torch` | Deep learning inference and training |
| `monai` | MONAI UNet architecture (Task A refinement) |
| `scikit-image` | Brain contour extraction, morphological ops |
| `scipy` | Spline smoothing, distance transforms, graph tools |
| `networkx` | MST-based catheter spine extraction |
| `numpy` | Array operations |
| `tqdm` | Progress bars |
| `joblib` | Model serialisation (Task C regressor) |
| `pandas` | CSV output for Task C results |
| `matplotlib` | Visualisation in notebooks |
