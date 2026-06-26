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
    return tf.keras.models.load_model('models/Dense121_finetuned_model.keras')

def predict(image):
    img = image.convert('RGB').resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = load_model().predict(img, verbose=0)[0]
    top3_idx = np.argsort(preds)[::-1][:3]
    return {
        "predicted_class": CLASS_NAMES[np.argmax(preds)],
        "confidence": float(np.max(preds)),
        "top3": [{"class": CLASS_NAMES[i], "confidence": float(preds[i])} for i in top3_idx],
        "all_probs": {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    }

st.set_page_config(page_title="EuroSAT Classifier", page_icon="🛰️", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🛰️ EuroSAT Classifier")
    st.markdown("---")
    st.header("About")
    st.write("Classifies satellite images into 10 land-use categories using DenseNet121.")
    st.markdown("---")
    st.header("Model Info")
    st.markdown("**Backbone:** DenseNet121")
    st.markdown("**Dataset:** EuroSAT RGB")
    st.markdown("**Val Accuracy:** 93.3%")
    st.markdown("**Classes:** 10")
    st.markdown("---")
    st.header("Land Use Classes")
    for name in CLASS_NAMES:
        st.markdown(f"• {name}")

# Main
st.title("🛰️ EuroSAT Land Use Classifier")
st.write("Upload a satellite image to classify its land use category.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])

    with col1:
        #st.image(image, caption="Uploaded Image", use_column_width=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Classifying..."):
            result = predict(image)

        confidence = result['confidence'] * 100

        if confidence >= 80:
            st.success(f"**{result['predicted_class']}**")
        elif confidence >= 60:
            st.warning(f"**{result['predicted_class']}**")
        else:
            st.error(f"**{result['predicted_class']}**")

        st.metric("Confidence", f"{confidence:.2f}%")

        st.subheader("Top 3 Predictions")
        for item in result["top3"]:
            st.progress(item["confidence"],
                       text=f"{item['class']}: {item['confidence']*100:.1f}%")

    st.markdown("---")
    st.subheader("All Class Probabilities")
    probs = result["all_probs"]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(list(probs.keys()), list(probs.values()), color='steelblue')
    bars[list(probs.keys()).index(result['predicted_class'])].set_color('green')
    ax.set_xlabel("Confidence")
    ax.set_title("Class Probabilities")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("Upload a satellite image to get started.")
    st.markdown("---")
    st.subheader("Supported Land Use Classes")
    cols = st.columns(5)
    for i, name in enumerate(CLASS_NAMES):
        cols[i % 5].info(name)