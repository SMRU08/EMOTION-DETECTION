from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.index import (  # noqa: E402
    CASCADE_PATH,
    LABELS,
    MAX_IMAGE_BYTES,
    MODEL_PATH,
    PUBLIC_PATH,
    app,
    get_face_detector,
    get_model,
)

__all__ = [
    "app",
    "get_model",
    "get_face_detector",
    "LABELS",
    "MAX_IMAGE_BYTES",
    "MODEL_PATH",
    "CASCADE_PATH",
    "PUBLIC_PATH",
]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
