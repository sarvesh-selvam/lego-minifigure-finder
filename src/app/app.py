import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.inference.bundle import load_bundle
from src.inference.predictor import Predictor

# ---------------------------------------------------------------------------
# Configuration — point this at your trained bundle directory
# ---------------------------------------------------------------------------
BUNDLE_DIR = Path("artifacts/test_v1/bundle")

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model bundle once on startup, release on shutdown."""
    global predictor
    if not BUNDLE_DIR.exists():
        raise RuntimeError(
            f"Bundle not found at '{BUNDLE_DIR}'. "
            "Run the training pipeline first or update BUNDLE_DIR in app.py."
        )
    predictor = load_bundle(BUNDLE_DIR)
    print(f"Model loaded from {BUNDLE_DIR}")
    yield
    predictor = None


app = FastAPI(
    title="LEGO Minifigure Finder",
    description="Upload an image to check whether it contains a LEGO minifigure.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image file and return classification results.

    Returns:
        pred_label:    predicted class name ("minifig" or "not_minifig")
        pred_idx:      predicted class index
        positive_prob: probability of the positive (minifig) class
        is_positive:   True if positive_prob >= threshold
        probs:         full probability distribution over all classes
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    result = predictor.predict_pil(img)
    return JSONResponse(content=result)
