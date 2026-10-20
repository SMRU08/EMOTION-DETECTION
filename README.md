# Mood Lens

A responsive emotion-detection demo built from the project's five-class Keras model. The browser captures a still image or accepts an upload; a FastAPI endpoint finds the largest face with the bundled Haar cascade and predicts one of five expression classes using a TensorFlow Lite model.

## Run locally on Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>. Camera capture requires localhost or HTTPS. Uploads are limited to 4 MB after browser-side resizing. Images are analyzed in memory and are not saved.

## Deploy on Vercel

Import this Git repository into Vercel with the project root set to the repository root. Vercel detects the FastAPI app in `app.py` and installs `requirements.txt`. The model and cascade files are included in the function bundle. Vercel uses the lightweight LiteRT interpreter rather than the TensorFlow training runtime.

## Convert the model

`Emotion_detection_model_new.tflite` is generated from the source H5 model by `scripts/convert_model.py`. To regenerate it, install `requirements-dev.txt` and run:

```powershell
python scripts/convert_model.py
```

The converter checks that LiteRT predictions match the Keras model on a sample input.

## Notes

The classifier was trained on 48 × 48 grayscale images and recognizes Angry, Happy, Neutral, Sad, and Surprised. This is an experimental expression classifier, not a measure of a person's actual emotional state.
