from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def find_asset(relative_path: str) -> Path:
    """Resolve an asset path across different execution environments

    (root, api subfolder, or Vercel serverless /var/task).
    """
    candidates: list[Path] = [Path.cwd() / relative_path]
    current_file = globals().get("__file__")
    if current_file:
        file_path = Path(current_file).resolve()
        candidates.append(file_path.parent / relative_path)
        candidates.append(file_path.parent.parent / relative_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


MODEL_PATH = find_asset("Emotion_detection_model_new.tflite")
CASCADE_PATH = find_asset("haarcascade_frontalface_default.xml")
PUBLIC_PATH = find_asset("public")
LABELS = ("Angry", "Happy", "Neutral", "Sad", "Surprised")
MAX_IMAGE_BYTES = 4 * 1024 * 1024

app = FastAPI(title="Mood Lens", version="1.0.0")


@lru_cache(maxsize=1)
def get_model():
    model_path = find_asset("Emotion_detection_model_new.tflite")
    if not model_path.is_file():
        raise RuntimeError(f"The TensorFlow Lite model file is missing at {model_path}.")

    Interpreter = None
    # 1. Prefer ai-edge-litert (standard on Linux / Vercel Serverless)
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        pass

    # 2. Try tflite_runtime
    if Interpreter is None:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            pass

    # 3. Fallback to tensorflow (e.g. for local Windows development)
    if Interpreter is None:
        try:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "Could not load a TFLite interpreter. "
                "Ensure ai-edge-litert, tflite-runtime, or tensorflow is installed."
            ) from exc

    interpreter = Interpreter(model_path=str(model_path), num_threads=2)
    interpreter.allocate_tensors()
    return interpreter


@lru_cache(maxsize=1)
def get_face_detector():
    cascade_path = find_asset("haarcascade_frontalface_default.xml")
    if not cascade_path.is_file():
        raise RuntimeError(f"The face detector file is missing at {cascade_path}.")
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError("The face detector file could not be loaded.")
    return detector


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def index():
    public_dir = find_asset("public")
    index_file = public_dir / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return {"status": "ok", "message": "Mood Lens API is running."}


@app.get("/api/health")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": get_model.cache_info().currsize == 1,
        "labels": list(LABELS),
    }


@app.post("/api/predict")
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    content_type = (image.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a JPG, PNG, or WEBP image.")

    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Choose an image smaller than 4 MB.")
    if not data:
        raise HTTPException(status_code=400, detail="The image is empty.")

    encoded = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="We could not read that image. Try a JPG or PNG.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = get_face_detector().detectMultiScale(
        gray, scaleFactor=1.12, minNeighbors=5, minSize=(36, 36)
    )
    if len(faces) == 0:
        raise HTTPException(
            status_code=422,
            detail="No face found. Try a clear, front-facing photo with good light.",
        )

    x, y, width, height = max(faces, key=lambda rect: rect[2] * rect[3])
    face = gray[y : y + height, x : x + width]
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    batch = (face.astype(np.float32) / 255.0).reshape(1, 48, 48, 1)

    try:
        interpreter = get_model()
        input_info = interpreter.get_input_details()[0]
        output_info = interpreter.get_output_details()[0]
        interpreter.set_tensor(input_info["index"], batch)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_info["index"])[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The model could not analyze this image.") from exc

    if len(scores) != len(LABELS):
        raise HTTPException(status_code=500, detail="The model output does not match the five emotion labels.")

    winner = int(np.argmax(scores))
    return {
        "emotion": LABELS[winner],
        "confidence": round(float(scores[winner]) * 100, 1),
        "scores": [
            {"emotion": label, "score": round(float(score) * 100, 1)}
            for label, score in zip(LABELS, scores)
        ],
        "face": {"x": int(x), "y": int(y), "width": int(width), "height": int(height)},
    }


# Mount static assets directory if available
if PUBLIC_PATH.is_dir():
    app.mount("/static", StaticFiles(directory=str(PUBLIC_PATH)), name="static")
