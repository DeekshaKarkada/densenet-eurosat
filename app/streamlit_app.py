import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/Dense121_base_model.keras')

def predict(image):
    img = image.convert('RGB').resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = load_model().predict(img)[0]
    top3_idx = np.argsort(preds)[::-1][:3]
    return {
        "predicted_class": CLASS_NAMES[np.argmax(preds)],
        "confidence":      float(np.max(preds)),
        "top3":            [{"class": CLASS_NAMES[i], "confidence": float(preds[i])} for i in top3_idx],
        "all_probs":       {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    }

st.set_page_config(page_title="EuroSAT Classifier", page_icon="🛰️")
st.title("🛰️ EuroSAT Land Use Classifier")
st.write("Upload a satellite image to classify its land use category.")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    with st.spinner("Classifying..."):
        result = predict(image)

    st.success(f"Predicted: **{result['predicted_class']}**")
    st.metric("Confidence", f"{result['confidence']*100:.2f}%")

    st.subheader("Top 3 Predictions")
    for item in result["top3"]:
        st.progress(item["confidence"],
                   text=f"{item['class']}: {item['confidence']*100:.1f}%")

    st.subheader("All Class Probabilities")
    probs = result["all_probs"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(list(probs.keys()), list(probs.values()))
    ax.set_xlabel("Confidence")
    ax.set_title("Class Probabilities")
    plt.tight_layout()
    st.pyplot(fig)