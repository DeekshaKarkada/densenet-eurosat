import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model('models/Dense121_base_model.keras')
    return model

def preprocess_image(image: Image.Image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img

def predict(image: Image.Image):
    model = load_model()
    img   = preprocess_image(image)
    preds = model.predict(img)[0]

    top3_idx   = np.argsort(preds)[::-1][:3]
    top3       = [{"class": CLASS_NAMES[i], 
                   "confidence": float(preds[i])} for i in top3_idx]
    
    return {
        "predicted_class": CLASS_NAMES[np.argmax(preds)],
        "confidence":      float(np.max(preds)),
        "top3":            top3,
        "all_probs":       {CLASS_NAMES[i]: float(preds[i]) 
                            for i in range(len(CLASS_NAMES))}
    }