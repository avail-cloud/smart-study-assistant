import streamlit as st
import requests
from PIL import Image
import os
import time
import pandas as pd

# ===== Configuration =====
try:
    # Read API_BASE from secrets, fallback to default if not found
    API_BASE = st.secrets["general"]["API_BASE"]  
except KeyError:
    st.error("API_BASE is missing from secrets.toml. Using default value.")
    API_BASE = "http://localhost:8000"  # Default value if not found in secrets

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "ms", "metrics.csv")

# Streamlit app configuration
st.set_page_config(page_title="Smart Study Assistant", layout="centered")
st.title("Smart Study Assistant — Streamlit UI")

st.markdown(
    f"This UI talks to the FastAPI microservice (ms) running at `{API_BASE}`. Make sure you started `ms/app.py` before calling features."
)

# Tabs for different functionalities
tabs = st.tabs(["Ask (Text)", "Explain", "Summarize", "Upload Image (OCR)", "Image → QA", "Auto-Grade", "Metrics"])

# ---------- Helper functions ----------
def call_api_json(endpoint: str, payload: dict, timeout=30):
    url = f"{API_BASE}{endpoint}"
    try:
        t0 = time.time()
        r = requests.post(url, json=payload, timeout=timeout)
        latency = time.time() - t0
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def call_api_file(endpoint: str, file_bytes: bytes, filename: str, fields: dict = None, timeout=60):
    url = f"{API_BASE}{endpoint}"
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    data = fields or {}
    try:
        t0 = time.time()
        r = requests.post(url, files=files, data=data, timeout=timeout)
        latency = time.time() - t0
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# ---------- Tab: Ask (Text) ----------
with tabs[0]:
    st.header("Ask a Study Question")
    q = st.text_area("Question", height=120, placeholder="e.g., Explain DNS in simple terms")
    ctx = st.text_area("Optional context (paste notes)", height=120)
    if st.button("Get Answer"):
        if not q.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Contacting microservice..."):
                payload = {"question": q, "context": ctx}
                res = call_api_json("/qa", payload)
            if res.get("error"):
                st.error(res["error"])
            else:
                st.subheader("Answer")
                st.write(res.get("text") or res.get("answer") or "No text returned.")
                st.caption(f"Latency (ms): {round((res.get('latency') or 0)*1000,1)}")

# ---------- Tab: Explain ----------
with tabs[1]:
    st.header("Explain a Topic")
    topic = st.text_input("Topic to explain", placeholder="e.g., Virtual Memory")
    if st.button("Explain"):
        if not topic.strip():
            st.warning("Enter a topic.")
        else:
            with st.spinner("Generating explanation..."):
                res = call_api_json("/explain", {"topic": topic})
            if res.get("error"):
                st.error(res["error"])
            else:
                st.write(res.get("text", "No response"))

# ---------- Tab: Summarize ----------
with tabs[2]:
    st.header("Summarize Text")
    long_text = st.text_area("Paste long text here", height=220)
    if st.button("Summarize"):
        if not long_text.strip():
            st.warning("Paste text to summarize.")
        else:
            with st.spinner("Summarizing..."):
                res = call_api_json("/summarize", {"text": long_text})
            if res.get("error"):
                st.error(res["error"])
            else:
                st.write(res.get("text", "No summary returned"))

# ---------- Tab: Upload Image (OCR) ----------
with tabs[3]:
    st.header("Upload Image — OCR (extract text)")
    uploaded = st.file_uploader("Upload an image (png, jpg) for OCR", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image preview", use_column_width=True)
        if st.button("Extract Text"):
            with st.spinner("Uploading and extracting..."):
                file_bytes = uploaded.getvalue()
                res = call_api_file("/upload_image", file_bytes, uploaded.name)
            if res.get("error"):
                st.error(res["error"])
            else:
                st.subheader("Extracted Text")
                st.text_area("OCR Output", res.get("extracted_text", ""), height=240)

# ---------- Tab: Image → QA ----------
with tabs[4]:
    st.header("Image → QA (OCR + Ask)")
    uploaded2 = st.file_uploader("Upload image for OCR + QA", type=["png", "jpg", "jpeg"], key="imgqa")
    question_for_image = st.text_input("Question about image (optional)", placeholder="e.g., Summarize this")
    if uploaded2:
        st.image(Image.open(uploaded2), caption="Image preview", use_column_width=True)
        if st.button("Ask about image"):
            with st.spinner("OCR then asking..."):
                res = call_api_file("/image_qa", uploaded2.getvalue(), uploaded2.name, fields={"question": question_for_image})
            if res.get("error"):
                st.error(res["error"])
            else:
                st.subheader("Extracted Text")
                st.text_area("OCR text", res.get("extracted_text", ""), height=180)
                st.subheader("Model Answer")
                st.write(res.get("text", ""))

# ---------- Tab: Auto-grade ----------
with tabs[5]:
    st.header("Auto-grade Short Answer")
    grade_q = st.text_input("Question to grade", placeholder="What is cloud computing?")
    grade_ref = st.text_area("Reference Answer (model answer)", height=120)
    grade_student = st.text_area("Student's answer", height=120)
    if st.button("Grade Answer"):
        if not (grade_q.strip() and grade_ref.strip() and grade_student.strip()):
            st.warning("Fill question, reference, and student answer.")
        else:
            with st.spinner("Grading..."):
                payload = {"question": grade_q, "reference": grade_ref, "student_answer": grade_student}
                res = call_api_json("/grade", payload)
            if res.get("error"):
                st.error(res["error"])
            else:
                st.subheader("Grade & Feedback")
                st.write(res.get("text", ""))

# ---------- Tab: Metrics ----------
with tabs[6]:
    st.header("Metrics (from ms/metrics.csv)")
    st.markdown("The microservice logs metrics to `ms/metrics.csv`. Click below to load and preview.")
    if st.button("Load metrics file"):
        try:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "ms", "metrics.csv"))
            st.dataframe(df.tail(30))
        except Exception as e:
            st.error(f"Could not load metrics.csv: {e}")
    st.markdown("If you don't see metrics, ensure the microservice has been called and `metrics.csv` exists.")

# Footer
st.markdown("---")
st.caption("Smart Study Assistant UI — connect this to the FastAPI microservice running at localhost.")
