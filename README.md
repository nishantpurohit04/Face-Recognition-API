# Face Recognition API

A production-style face recognition service built with **FastAPI + DeepFace (ArcFace) + MTCNN**, containerised with Docker.

## Features

| Endpoint | Description |
|---|---|
| `POST /verify` | 1:1 face verification (same person?) |
| `POST /identify` | 1:N face identification against gallery |
| `POST /register` | Register a face into the gallery |
| `GET /gallery` | List all registered identities |
| `DELETE /gallery/{name}` | Remove an identity |
| `POST /embed` | Extract raw 512-d ArcFace embedding |

---

## Quick start (Docker)

```bash
# Build & run
docker compose up --build

# API:      http://localhost:8000
# UI:       http://localhost:8000
# Swagger:  http://localhost:8000/docs
```

Gallery data persists in `./known_faces/` across restarts.

---

## Quick start (local, no Docker)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd app
uvicorn main:app --reload --port 8000
```

---

## Project structure

```
face-recognition-api/
├── app/
│   └── main.py          ← FastAPI application (all routes)
├── frontend/
│   └── index.html       ← Browser UI (served at /)
├── known_faces/         ← Gallery (auto-created, git-ignored)
│   └── <name>/
│       └── <uuid>.jpg
├── evaluate.py          ← MLflow threshold evaluation script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Pipeline

```
Input image
    │
    ▼
MTCNN (detect + align)       ← 5-landmark affine warp → 112×112 crop
    │
    ▼
ArcFace backbone             ← ResNet-100, trained with angular margin loss
    │
    ▼
512-d L2-normalised vector
    │
    ▼
Cosine distance vs gallery   ← threshold τ = 0.40 (adjustable in main.py)
    │
    ▼
Identity label / "Unknown"
```

---

## MLflow evaluation

Evaluate accuracy across different thresholds and log metrics:

```bash
pip install mlflow scikit-learn
python evaluate.py --test_dir test_pairs/ --model ArcFace
mlflow ui   # view at http://localhost:5000
```

Expected layout for `test_pairs/`:
```
test_pairs/
├── same/         ← pairs of the same person (pair_01_a.jpg + pair_01_b.jpg)
└── different/    ← pairs of different people
```

---

## Key design decisions

- **ArcFace over FaceNet** — additive angular margin loss gives a more discriminative embedding space; achieves 99.83% on LFW vs FaceNet's 99.63%.
- **MTCNN detector** — returns 5 facial landmarks used for geometric alignment before embedding. More robust than Haar cascades on tilted / small faces.
- **Cosine similarity** — preferred over Euclidean for L2-normalised embeddings (ArcFace outputs normalised vectors by design).
- **Gallery as directory** — simple, no DB dependency, easy to volume-mount. Replace with FAISS index for galleries > 10k identities.
- **Multi-image per identity** — `/register` can be called multiple times per name. Matching takes the minimum distance across all reference images.

---

## Tech stack

| Layer | Tool |
|---|---|
| API framework | FastAPI |
| Face recognition | DeepFace (ArcFace model) |
| Face detection | MTCNN |
| Serving | Uvicorn |
| Containerisation | Docker + Docker Compose |
| Experiment tracking | MLflow |

---

## Portfolio talking points

1. Implemented the full face recognition pipeline from scratch (detect → align → embed → match)
2. Chose ArcFace over FaceNet — can explain angular margin loss and why it outperforms triplet loss
3. Designed a RESTful API with clear separation of concerns (detection, embedding, matching)
4. Containerised with Docker, gallery persists via volume mount
5. MLflow integration for threshold evaluation — demonstrates MLOps awareness
