import numpy as np
import json
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Cancer Type Classifier", layout="centered")

@st.cache_resource
def load_data():
    bundle = joblib.load("outputs/models/deployment_bundle.joblib")
    with open("outputs/models/app_ui_data.json") as f:
        ui_data = json.load(f)
    return bundle, ui_data

bundle, ui_data = load_data()
model = bundle["model"]
gene_order = bundle["gene_order"]
mean_ = bundle["mean_"]
scale_ = bundle["scale_"]
label_classes = bundle["label_classes"]
slider_info = ui_data["slider_info"]
examples = ui_data["examples"]

st.title("Cancer Type Classifier")
st.caption(
    "Predicts cancer type (BRCA, COAD, KIRC, LUAD, PRAD) from RNA-Seq gene "
    "expression levels across 25 key genes."
)

# ---- Session state init ----
for gene in gene_order:
    if gene not in st.session_state:
        st.session_state[gene] = slider_info[gene]["default"]

# ---- Load example patient ----
st.subheader("Quick test: load a real patient")
col1, col2 = st.columns([2, 1])
with col1:
    chosen_class = st.selectbox("Class:", options=list(examples.keys()))
with col2:
    patient_options = [ex["sample_id"] for ex in examples[chosen_class]]
    chosen_patient = st.selectbox("Patient:", options=patient_options)

if st.button("Load this patient's values"):
    patient_data = next(
        ex for ex in examples[chosen_class] if ex["sample_id"] == chosen_patient
    )
    for gene, val in patient_data["values"].items():
        st.session_state[gene] = val
    st.success(f"Loaded patient (true class: {chosen_class}) — scroll down to Predict")

st.divider()

# ---- Sliders ----
st.subheader("Gene expression levels")

for i, gene in enumerate(gene_order, start=1):
    info = slider_info[gene]
    st.slider(
        f"Gene {i}",
        min_value=float(info["min"]),
        max_value=float(info["max"]),
        step=(info["max"] - info["min"]) / 200 if info["max"] > info["min"] else 0.01,
        key=gene,
        help=f"Dataset ID: {gene}",
    )

st.divider()

if st.button("Predict", type="primary"):
    raw = np.array([st.session_state[g] for g in gene_order], dtype=float)
    log_vals = np.log2(raw + 1)
    scaled = (log_vals - mean_) / scale_
    X = scaled.reshape(1, -1)

    pred_idx = model.predict(X)[0]
    pred_label = label_classes[pred_idx]
    proba = model.predict_proba(X)[0]

    st.success(f"Predicted cancer type: **{pred_label}**")

    proba_df = pd.DataFrame(
        {"Cancer type": label_classes, "Confidence": proba}
    ).set_index("Cancer type")
    st.bar_chart(proba_df)

    st.caption(
        "This is a student project demo trained on a public benchmark "
        "dataset and is not a diagnostic tool."
    )
