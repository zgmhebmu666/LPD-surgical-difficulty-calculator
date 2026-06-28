# -*- coding: utf-8 -*-
"""
Streamlit web calculator for the LPD surgical difficulty prediction model.
Deploy this folder to Streamlit Community Cloud, a hospital intranet server,
or any environment that supports Streamlit.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model_calibrated.joblib"
METADATA_PATH = BASE_DIR / "calculator_metadata.json"

st.set_page_config(
    page_title="LPD Surgical Difficulty Calculator",
    page_icon="🧮",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

model = load_model()
meta = load_metadata()

st.title("LPD Surgical Difficulty Prediction Calculator")
st.caption("Machine-learning model for individualized risk estimation")

st.info(
    "This web calculator is intended for research and auxiliary risk assessment only. "
    "It does not replace clinical judgement or multidisciplinary decision-making."
)

st.markdown(
    f"""
**Best model:** {meta['best_model_display']}  
**Validation AUC:** {meta['validation_auc']:.3f} (95% CI {meta['validation_auc_ci_low']:.3f}–{meta['validation_auc_ci_high']:.3f})  
**Classification threshold:** {meta['threshold_train_youden']:.3f} based on the training-set Youden index
"""
)

values = {}
st.subheader("Input predictors")

for item in meta["variables"]:
    name = item["name"]
    label = item.get("display_name", name)
    help_text = f"Original variable name: {name}"

    if item["type"] == "numeric":
        min_value = float(item["min"])
        max_value = float(item["max"])
        default = float(item["default"])
        step = 1.0
        data_range = max_value - min_value
        if data_range <= 10:
            step = 0.1
        if data_range <= 1:
            step = 0.01
        values[name] = st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=min(max(default, min_value), max_value),
            step=step,
            help=help_text,
        )
    else:
        options = item["categories"]
        default = item.get("default", options[0])
        option_labels = item.get("option_labels", {})

        def format_option(x):
            return option_labels.get(str(x), str(x))

        default_index = 0
        for i, opt in enumerate(options):
            if str(opt) == str(default):
                default_index = i
                break
        values[name] = st.selectbox(
            label,
            options=options,
            index=default_index,
            format_func=format_option,
            help=help_text,
        )

input_df = pd.DataFrame([values], columns=meta["selected_variables"])

st.divider()
if st.button("Calculate predicted risk", type="primary"):
    probability = float(model.predict_proba(input_df)[:, 1][0])
    threshold = float(meta["threshold_train_youden"])
    classification = "High-risk / predicted difficult" if probability >= threshold else "Low-risk / predicted non-difficult"

    st.metric("Predicted probability of surgical difficulty", f"{probability * 100:.1f}%")
    st.write(f"**Risk classification:** {classification}")
    st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("Model input used for prediction"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption(
    "The calculator was generated automatically from the final calibrated model. "
    "Please validate the model externally before routine clinical use."
)
