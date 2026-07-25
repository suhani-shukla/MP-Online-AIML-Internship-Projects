"""
app.py
Streamlit app for interactive CIFAR-10 image classification.
Upload an image and get a predicted class with confidence scores.
"""

import os

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from predict import predict_array, load_artifacts


st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)


@st.cache_resource
def get_model_and_classes():
    """Loads the model and class names once, cached across reruns."""
    return load_artifacts()


def main():
    st.title("🖼️ CIFAR-10 Image Classifier")
    st.write(
        "Upload an image and the model will classify it into one of "
        "10 categories: airplane, automobile, bird, cat, deer, dog, "
        "frog, horse, ship, or truck."
    )

    model, class_names = get_model_and_classes()

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Classifying..."):
            image_array = np.array(image)
            result = predict_array(image_array, model=model, class_names=class_names)

        st.success(f"**Predicted Class:** {result['predicted_class'].capitalize()}")
        st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

        st.subheader("Prediction Probabilities")
        probs = result["all_probabilities"]
        sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

        fig, ax = plt.subplots(figsize=(8, 4))
        labels = list(sorted_probs.keys())
        values = list(sorted_probs.values())
        colors = ["#2ecc71" if l == result["predicted_class"] else "#3498db" for l in labels]

        ax.barh(labels, values, color=colors)
        ax.set_xlabel("Probability")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        plt.tight_layout()

        st.pyplot(fig)

        with st.expander("Raw probability values"):
            st.json(sorted_probs)
    else:
        st.info("👆 Upload an image to get started.")

    st.markdown("---")
    st.caption("Model: CNN trained on CIFAR-10 | Built with TensorFlow + Streamlit")


if __name__ == "__main__":
    main()