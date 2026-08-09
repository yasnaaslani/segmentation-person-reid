# Segmentation Person ReID on OAK-D

A minimal demo showing **person re-identification from segmentation-only pixels**.

Instead of feeding a full rectangular detector crop into Person ReID, this project:

```text
person segmentation mask
        ↓
mask-derived tight crop
        ↓
background and neighboring pixels set to black
        ↓
128×256 Person ReID input
        ↓
normalized embedding
        ↓
cosine similarity
```

Click a visible person's actual segmentation mask to enroll one reference embedding. The demo then computes the similarity of every currently visible segmented person to that reference.

This is deliberately **not a tracker** and does not contain the larger project's face bootstrap, embedding bank, overlap state machine, motion path, depth-following, or drone logic.

## Why segmentation-masked ReID?

A detector bbox can include background or pixels from a nearby person, especially in crowds. Masking keeps the ReID input focused on the visible pixels assigned to that person instance.

## Features

- OAK-D RGB input
- YOLOv8n-seg person instances
- Mask-IoU / containment duplicate suppression
- Mask-derived ReID crops — detector bbox is not used to decide ReID crop geometry
- Black background outside the person silhouette
- On-device Person ReID inference
- L2-normalized embeddings
- Cosine similarity
- Mouse-based reference enrollment using the actual person mask
- Startup warm-up to suppress noisy early detections

## Repository layout

```text
segmentation-person-reid/
├── README.md
├── requirements.txt
├── .gitignore
├── models/
│   └── README.md
├── assets/
└── src/
    ├── main.py
    ├── reid.py
    └── yolov8_seg.py
```

## Install

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py \
  --seg-model models/yolov8n-seg-416_openvino_2022.1_4shave.blob \
  --reid-model models/person-reidentification-retail-0277.blob
```

### Controls

- **Left click on a person's mask**: enroll/replace the reference
- **R**: clear the reference
- **Q / Esc**: quit

After enrollment, each visible person is labeled with `sim=<cosine similarity>`.

## Important scope note

This demo intentionally uses a **single reference embedding**. It does not implement a multi-view embedding bank. That makes it useful for demonstrating masked ReID without exposing the architecture of a larger target-tracking system.

## Practical note

A single body embedding is not expected to be equally strong from every viewpoint. Front/back/side viewpoint robustness belongs in a separate multi-view ReID-bank project rather than this minimal demo.

## Model files

See `models/README.md`. Model binaries are excluded from Git.
