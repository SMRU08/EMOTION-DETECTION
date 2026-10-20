# Mood Lens — Facial Expression Demo

Mood Lens is a small web demo that detects one of five facial expressions from a photo. It combines a responsive browser interface with a FastAPI endpoint, OpenCV face detection, and a TensorFlow Lite model.

> **Important:** This is an experimental expression classifier. It cannot determine a person's true feelings, and its predictions may be wrong. Do not use it for medical, hiring, monitoring, or other consequential decisions.

## Features

- Take a snapshot with your camera or upload a JPG, PNG, or WEBP image.
- Detect the largest face and classify its expression as **Angry**, **Happy**, **Neutral**, **Sad**, or **Surprised**.
- View a confidence score for each class.
- Send image data in memory for prediction; the application does not save uploaded images.
- Deploy the FastAPI application and its static interface together on Vercel.

## How it works

1. The browser captures or reads an image and submits it to `POST /api/predict`.
2. OpenCV locates the largest face using the bundled Haar cascade.
3. The face is converted to 48 × 48 grayscale pixels and passed to the bundled TensorFlow Lite model.
4. The API returns the predicted class and class scores as JSON.

The classifier was trained on 48 × 48 grayscale inputs. Face detection and expression classification can both fail with poor lighting, an obstructed face, or images unlike the training data.

## Run locally on Windows

Python 3.12 is recommended. From the repository root, run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>. Camera access is available on localhost or over HTTPS. The browser resizes uploads before submission; the API also enforces a 4 MB request limit. Photos are processed in memory and are not written to disk.

## API

### `GET /api/health`

Returns a basic health response and whether the model has been loaded in the current process.

### `POST /api/predict`

Accepts a multipart form field named `image` containing an image. A successful response includes the predicted expression and per-class scores. Invalid files, oversized uploads, and images without a detectable face return an appropriate HTTP error.

Interactive API documentation is available at `/docs` when the application is running.

## Deploy to Vercel

1. Push this repository to GitHub.
2. In Vercel, import the GitHub repository and set the **Root Directory** to the directory containing `app.py` and `vercel.json` (the repository root for this project).
3. Keep the framework preset on **Other** or allow Vercel to detect the Python FastAPI application. The entry point is `app.py`; dependencies are listed in `requirements.txt`.
4. Deploy, then check `/api/health` and the site homepage.

`vercel.json` configures a 60-second function limit. Vercel runs the app with the Linux LiteRT dependency; `requirements-dev.txt` is for local Windows development. If a previous deployment returns `404: NOT_FOUND`, check that the deployment uses the correct Git repository, production branch, and root directory, then redeploy the latest commit.

## Model files

- `Emotion_detection_model_new.tflite` — inference model used by the web app.
- `Emotion_detection_model_new.h5` — original Keras model used as the conversion source.
- `haarcascade_frontalface_default.xml` — OpenCV face detector.

To regenerate the TFLite file, install `requirements-dev.txt` and run:

```powershell
python scripts/convert_model.py
```

The conversion script compares TFLite and Keras predictions on a deterministic sample before finishing.

## Repository layout

```text
app.py                                FastAPI application and prediction API
public/index.html                     Browser interface
Emotion_detection_model_new.tflite   Runtime model
haarcascade_frontalface_default.xml   Face detector
scripts/convert_model.py              Model conversion utility
requirements.txt                      Vercel/Linux runtime dependencies
requirements-dev.txt                  Local Windows development dependencies
vercel.json                           Vercel function configuration
```

## License

No license is currently included. Contact the repository owner before reusing or redistributing this project.
