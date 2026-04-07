import io
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.inference.predictor import Predictor
from src.inference.bundle import load_bundle

MLFLOW_TRACKING_URI = "mlflow"
REGISTERED_MODEL = "lego-minifigure-finder"
MODEL_ALIAS = "production"

predictor: Predictor | None = None
model_info: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, model_info

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    # Resolve the production alias to a concrete version
    try:
        version = client.get_model_version_by_alias(REGISTERED_MODEL, MODEL_ALIAS)
    except Exception:
        raise RuntimeError(
            f"No '{MODEL_ALIAS}' alias found for model '{REGISTERED_MODEL}'. "
            "Run the training pipeline first."
        )

    # Download the bundle artifacts from the run associated with the registered version
    bundle_uri = f"runs:/{version.run_id}/bundle"
    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = Path(mlflow.artifacts.download_artifacts(bundle_uri, dst_path=tmp))
        predictor = load_bundle(bundle_dir)
        bundle_meta = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))

    model_info = {
        "registered_model": REGISTERED_MODEL,
        "alias": MODEL_ALIAS,
        "version": version.version,
        "run_id": version.run_id,
        "arch": bundle_meta["arch"],
        "class_names": bundle_meta["class_names"],
        "image_size": bundle_meta["image_size"],
        "threshold": bundle_meta["threshold"],
    }

    print(f"Model         : {REGISTERED_MODEL} @{MODEL_ALIAS} (version {version.version})")
    print(f"Arch          : {bundle_meta['arch']}")
    print(f"Classes       : {bundle_meta['class_names']}")
    print(f"Threshold     : {bundle_meta['threshold']}")
    yield
    predictor = None
    model_info = {}


app = FastAPI(
    title="LEGO Minifigure Finder",
    description="Upload an image to check whether it contains a LEGO minifigure.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Liveness check — reports which model version is loaded."""
    return {"status": "ok", "model_loaded": predictor is not None, **model_info}


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
