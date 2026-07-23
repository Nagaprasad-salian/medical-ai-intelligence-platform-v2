"""
Streamlit frontend for the Advanced AI Medical Intelligence Platform.
Talks to the FastAPI backend over HTTP.
"""

import os
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Medical Intelligence Platform", layout="wide")

st.title("🩺 Advanced AI Medical Intelligence Platform")
st.caption("Deep Learning + Explainable AI (Grad-CAM) + LLM-generated reports for chest X-ray screening.")

st.warning(
    "⚠️ This is a demo/educational tool. It is **not** a certified medical device and "
    "must never be used for real clinical diagnosis. Always consult a licensed physician."
)

tab_predict, tab_history = st.tabs(["🔍 New Prediction", "📜 History"])

with tab_predict:
    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        with col1:
            st.subheader("Uploaded Image")
            st.image(Image.open(uploaded_file), use_container_width=True)

        if st.button("Run Analysis", type="primary"):
            with st.spinner("Analyzing image..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                try:
                    resp = requests.post(f"{API_URL}/predict", files=files, timeout=60)
                    resp.raise_for_status()
                    result = resp.json()

                    with col2:
                        st.subheader("Grad-CAM Explanation")
                        img_url = f"{API_URL}{result['gradcam_image_url']}"
                        st.image(img_url, use_container_width=True)

                    st.subheader("Prediction Result")
                    m1, m2 = st.columns(2)
                    m1.metric("Predicted Class", result["predicted_class"])
                    m2.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    st.caption(f"Model focus region: {result['focus_region']}")

                    st.subheader("AI-Generated Report")
                    st.info(result["llm_report"])

                    st.caption(result["disclaimer"])

                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")

with tab_history:
    st.subheader("Prediction History")
    if st.button("Refresh"):
        st.rerun()

    try:
        resp = requests.get(f"{API_URL}/history", timeout=30)
        resp.raise_for_status()
        records = resp.json()

        if not records:
            st.info("No predictions yet. Run one from the 'New Prediction' tab.")
        else:
            for r in records:
                with st.expander(f"#{r['id']} — {r['predicted_class']} ({r['confidence']*100:.1f}%) — {r['created_at']}"):
                    st.write(f"**File:** {r['image_filename']}")
                    st.write(f"**Focus region:** {r['focus_region']}")
                    if r["gradcam_path"]:
                        st.image(f"{API_URL}/gradcam/{r['gradcam_path']}", width=300)
                    st.write("**Report:**")
                    st.write(r["llm_report"])
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch history: {e}")
