"""
Face Recognition API
Stack: FastAPI + DeepFace (ArcFace) + MTCNN
"""
 
import uuid
import shutil
import numpy as np
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager
 
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
 
from deepface import DeepFace
 
# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "ArcFace"       # alternatives: "Facenet", "VGG-Face"
DETECTOR        = "mtcnn"         # alternatives: "retinaface", "opencv"
DISTANCE_METRIC = "cosine"        # alternatives: "euclidean"
MATCH_THRESHOLD = 0.40            # cosine distance threshold (lower = stricter)
GALLERY_DIR     = Path("known_faces")
MAX_UPLOAD_MB   = 10              # reject uploads larger than this
GALLERY_DIR.mkdir(exist_ok=True)
 
 
# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n")
    print("  ✅ Server is running!")
    print("  🌐 Dashboard  →  http://localhost:8000")
    print("  🔒 Face ID    →  http://localhost:8000/face-id")
    print("  📄 API Docs   →  http://localhost:8000/docs")
    print("\n")
    yield
 
 
# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Face Recognition API",
    description="1:1 verification and 1:N identification using ArcFace + MTCNN",
    version="1.0.0",
    lifespan=lifespan,
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Serve the frontend
frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
 
 
# ── Schemas ───────────────────────────────────────────────────────────────────
class VerifyResponse(BaseModel):
    verified: bool
    distance: float
    threshold: float
    model: str
 
class IdentifyMatch(BaseModel):
    identity: str
    distance: float
    verified: bool
 
class IdentifyResponse(BaseModel):
    query_face_found: bool
    best_match: IdentifyMatch | None
    all_matches: List[IdentifyMatch]
    gallery_size: int
 
class RegisterResponse(BaseModel):
    success: bool
    identity: str
    message: str
 
class GalleryResponse(BaseModel):
    identities: List[str]
    count: int
 
class EmbeddingResponse(BaseModel):
    embedding_size: int
    embedding: List[float]
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def save_upload(file: UploadFile) -> Path:
    """
    Save an uploaded file to a temp location and return its path.
    Enforces MAX_UPLOAD_MB size limit via chunked read to prevent large uploads
    from exhausting disk or memory.
    """
    tmp_path = Path(f"/tmp/{uuid.uuid4().hex}_{file.filename}")
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with open(tmp_path, "wb") as f:
            while chunk := file.file.read(1024 * 64):  # 64 KB chunks
                written += len(chunk)
                if written > max_bytes:
                    tmp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {MAX_UPLOAD_MB} MB limit.",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")
    return tmp_path
 
 
def get_gallery_dirs() -> List[Path]:
    """
    Return only subdirectories inside GALLERY_DIR.
    Skips stray files like .DS_Store that would inflate gallery_size.
    """
    return [d for d in GALLERY_DIR.iterdir() if d.is_dir()]
 
 
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors (0 = identical, 2 = opposite)."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(1 - np.dot(a, b))
 
 
def sanitise_name(name: str) -> str:
    """Strip whitespace and reject path-traversal characters."""
    name = name.strip()
    if not name or any(c in name for c in r"/\.."):
        raise HTTPException(status_code=422, detail="Invalid identity name.")
    return name
 
 
# ── Routes ────────────────────────────────────────────────────────────────────
 
@app.get("/", include_in_schema=False)
async def root():
    index = Path("frontend/index.html")
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Face Recognition API — visit /docs for Swagger UI"}
 
 
@app.get("/face-id", include_in_schema=False)
async def face_id_page():
    """Serve the Face ID enrollment + verification UI."""
    page = Path("frontend/face-id.html")
    if page.exists():
        return FileResponse(str(page))
    return {"error": "face-id.html not found in frontend/"}
 
 
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "detector": DETECTOR}
 
 
# 1. 1:1 Verification ──────────────────────────────────────────────────────────
@app.post("/verify", response_model=VerifyResponse, tags=["Core"])
async def verify(
    image1: UploadFile = File(..., description="First face image"),
    image2: UploadFile = File(..., description="Second face image"),
):
    """
    Compare two face images and return whether they belong to the same person.
    """
    path1 = save_upload(image1)
    path2 = save_upload(image2)
    try:
        result = DeepFace.verify(
            img1_path=str(path1),
            img2_path=str(path2),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=True,
        )
        return VerifyResponse(
            verified=result["verified"],
            distance=round(result["distance"], 4),
            threshold=round(MATCH_THRESHOLD, 4),
            model=MODEL_NAME,
        )
    except ValueError:
        # Do not leak internal exception details to the client
        raise HTTPException(status_code=422, detail="Face not detected in one or both images.")
    except Exception:
        raise HTTPException(status_code=500, detail="Verification failed. Please try again.")
    finally:
        path1.unlink(missing_ok=True)
        path2.unlink(missing_ok=True)
 
 
# 2. 1:N Identification ────────────────────────────────────────────────────────
@app.post("/identify", response_model=IdentifyResponse, tags=["Core"])
async def identify(image: UploadFile = File(..., description="Query face image")):
    """
    Identify a face against the registered gallery.
    Returns best match and all candidates within threshold.
    """
    query_path = save_upload(image)
    identity_dirs = get_gallery_dirs()
 
    if not identity_dirs:
        query_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=404,
            detail="Gallery is empty. Register at least one face first via /register."
        )
 
    try:
        query_result = DeepFace.represent(
            img_path=str(query_path),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True,
        )
        query_emb = np.array(query_result[0]["embedding"])
    except ValueError:
        raise HTTPException(status_code=422, detail="No face detected in query image.")
    except Exception:
        raise HTTPException(status_code=500, detail="Embedding extraction failed.")
    finally:
        query_path.unlink(missing_ok=True)
 
    # Compare query against every gallery image; keep best (min) distance per identity
    matches: List[IdentifyMatch] = []
    for id_dir in identity_dirs:
        name = id_dir.name
        best_dist: float | None = None
 
        for img_file in id_dir.iterdir():
            if not img_file.is_file():
                continue
            try:
                ref_result = DeepFace.represent(
                    img_path=str(img_file),
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR,
                    enforce_detection=False,
                )
                ref_emb = np.array(ref_result[0]["embedding"])
                dist = cosine_distance(query_emb, ref_emb)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
            except Exception:
                continue  # skip unreadable gallery images
 
        if best_dist is not None:
            matches.append(IdentifyMatch(
                identity=name,
                distance=round(best_dist, 4),
                verified=best_dist <= MATCH_THRESHOLD,
            ))
 
    if not matches:
        return IdentifyResponse(
            query_face_found=True,
            best_match=None,
            all_matches=[],
            gallery_size=len(identity_dirs),
        )
 
    matches.sort(key=lambda m: m.distance)
    best = matches[0]
 
    return IdentifyResponse(
        query_face_found=True,
        best_match=best if best.verified else None,
        all_matches=matches[:5],
        gallery_size=len(identity_dirs),
    )
 
 
# 3. Register a face ───────────────────────────────────────────────────────────
@app.post("/register", response_model=RegisterResponse, tags=["Gallery"])
async def register(
    image: UploadFile = File(...),
    name: str = Form(..., description="Identity name (e.g. 'nishant')"),
):
    """
    Register a face into the gallery under a given identity name.
    Multiple images per identity are supported (improves matching accuracy).
    """
    name = sanitise_name(name)
    tmp_path = save_upload(image)
 
    try:
        DeepFace.represent(
            img_path=str(tmp_path),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True,
        )
    except ValueError:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="No face detected in the uploaded image.")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Could not process image.")
 
    id_dir = GALLERY_DIR / name
    id_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "").suffix or ".jpg"
    dest = id_dir / f"{uuid.uuid4().hex}{suffix}"
    shutil.move(str(tmp_path), str(dest))
 
    return RegisterResponse(
        success=True,
        identity=name,
        message=f"Face registered for '{name}'. Total images: {len(list(id_dir.iterdir()))}",
    )
 
 
# 4. Gallery management ────────────────────────────────────────────────────────
@app.get("/gallery", response_model=GalleryResponse, tags=["Gallery"])
async def list_gallery():
    """List all registered identities."""
    names = sorted(d.name for d in get_gallery_dirs())
    return GalleryResponse(identities=names, count=len(names))
 
 
@app.delete("/gallery/{name}", tags=["Gallery"])
async def delete_identity(name: str):
    """Remove an identity from the gallery."""
    name = sanitise_name(name)
    id_dir = GALLERY_DIR / name
    if not id_dir.exists() or not id_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Identity '{name}' not found.")
    shutil.rmtree(id_dir)
    return {"deleted": name}
 
 
# 5. Extract embedding ─────────────────────────────────────────────────────────
@app.post("/embed", response_model=EmbeddingResponse, tags=["Utilities"])
async def embed(image: UploadFile = File(...)):
    """
    Extract raw ArcFace embedding vector from a face image.
    Useful for building custom downstream classifiers.
    """
    tmp_path = save_upload(image)
    try:
        result = DeepFace.represent(
            img_path=str(tmp_path),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True,
        )
        emb = result[0]["embedding"]
        return EmbeddingResponse(embedding_size=len(emb), embedding=emb)
    except ValueError:
        raise HTTPException(status_code=422, detail="No face detected in the uploaded image.")
    except Exception:
        raise HTTPException(status_code=500, detail="Embedding extraction failed.")
    finally:
        tmp_path.unlink(missing_ok=True)