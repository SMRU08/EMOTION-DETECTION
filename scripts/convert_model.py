from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "Emotion_detection_model_new.h5"
target = ROOT / "Emotion_detection_model_new.tflite"
saved_model = ROOT / ".tmp" / "emotion_saved_model"

model = load_model(source, compile=False)
model.export(str(saved_model), format="tf_saved_model")
converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
converter.optimizations = []
target.write_bytes(converter.convert())

interpreter = tf.lite.Interpreter(model_path=str(target), num_threads=1)
interpreter.allocate_tensors()
input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]
sample = np.random.default_rng(7).random((1, 48, 48, 1), dtype=np.float32)
interpreter.set_tensor(input_info["index"], sample)
interpreter.invoke()
lite_scores = interpreter.get_tensor(output_info["index"])
keras_scores = model.predict(sample, verbose=0)
np.testing.assert_allclose(lite_scores, keras_scores, rtol=1e-4, atol=1e-5)
print(f"Created {target.name} ({target.stat().st_size:,} bytes); output shape {lite_scores.shape}")
