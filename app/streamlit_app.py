import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt

API_URL = "https://densenet-eurosat.onrender.com/predict"

st.set_page_config(page_title="EuroSAT Classifier", page_icon="🛰️")
st.title("🛰️ EuroSAT Land Use Classifier")
st.write("Upload a satellite image to classify its land use category.")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    with st.spinner("Classifying..."):
        response = requests.post(
            API_URL,
            files={"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
        )

    if response.status_code == 200:
        result = response.json()

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
    else:
        st.error(f"Prediction failed: {response.status_code}")