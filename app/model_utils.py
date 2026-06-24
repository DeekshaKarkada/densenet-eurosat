import traceback
import tensorflow as tf
import numpy as np
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'Dense121_base_model.keras')

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)  # ← load at startup
print("Model loaded!")

"""
def preprocess_image(image: Image.Image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img
"""

def predict(image: Image.Image):
    img = image.convert('RGB').resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)[0]
    top3_idx = np.argsort(preds)[::-1][:3]
    return {
        "predicted_class": CLASS_NAMES[np.argmax(preds)],
        "confidence": float(np.max(preds)),
        "top3": [{"class": CLASS_NAMES[i], "confidence": float(preds[i])} for i in top3_idx],
        "all_probs": {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    }