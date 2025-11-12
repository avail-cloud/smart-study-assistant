# app.py
"""
Smart Study Assistant - FastAPI microservice (Hugging Face router backend)
Put this file in ms/ and create ms/.env with HF_TOKEN and TESSERACT_CMD values.

Endpoints:
  GET  /ping
  POST /qa          -> {"question": "...", "context": "..."} returns {"text", "latency", "tokens"}
  POST /explain     -> {"topic": "..."}
  POST /summarize   -> {"text": "..."}
  POST /upload_image-> multipart form file -> {"extracted_text": "..."}
  POST /image_qa    -> multipart file + question form field -> OCR + QA
  POST /grade       -> {"question","reference","student_answer"} -> grade via model
"""

import os, time, json, csv, datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pytesseract
import requests
import uvicorn
from dotenv import load_dotenv

# -------------------- load .env from ms/ folder --------------------
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face router token
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
DEFAULT_HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")  # safe default

# configure tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# -------------------- app & CORS --------------------
app = FastAPI(title="Smart Study Assistant - MS (HF router)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev: allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

METRICS_FILE = str(BASE_DIR / "metrics.csv")
METRICS_HEADER = ["timestamp_utc", "endpoint", "latency_s", "model", "tokens_total", "notes"]

def ensure_metrics_file():
    if not os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(METRICS_HEADER)

def log_metrics(endpoint: str, latency: float, model: str, tokens_total: Optional[int], notes: str = ""):
    ensure_metrics_file()
    row = [
        datetime.datetime.utcnow().isoformat(),
        endpoint,
        f"{latency:.4f}",
        model or "",
        tokens_total if tokens_total is not None else "",
        notes or ""
    ]
    with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# -------------------- HF router HTTP wrapper --------------------
ROUTER_URL = "https://router.huggingface.co/hf-inference"

def hf_inference(prompt: str,
                 model: str = DEFAULT_HF_MODEL,
                 max_output_tokens: Optional[int] = None,
                 max_tokens: Optional[int] = None,
                 temperature: float = 0.6) -> dict:
    """
    Call HF router endpoint. Accepts max_tokens (compat) mapped to max_new_tokens.
    Returns {"text": str, "latency": float, "error": optional}
    """
    hf_token = HF_TOKEN
    if not hf_token:
        return {"text": "Hugging Face call failed: HF_TOKEN missing from ms/.env. Create a token on https://huggingface.co/settings/tokens", "latency": 0.0, "tokens_total": None}

    # map param names
    if max_output_tokens is None and max_tokens is not None:
        max_output_tokens = max_tokens
    if max_output_tokens is None:
        max_output_tokens = 256

    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "inputs": prompt,
        "parameters": {"max_new_tokens": int(max_output_tokens), "temperature": float(temperature)}
    }

    start = time.time()
    try:
        resp = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=60)
        latency = time.time() - start
    except Exception as e:
        return {"text": f"Hugging Face call failed (network): {e}", "latency": 0.0, "tokens_total": None}

    if resp.status_code != 200:
        # attempt to surface server error
        try:
            err = resp.json()
            err_msg = err.get("error") or json.dumps(err)
        except Exception:
            err_msg = resp.text
        return {"text": f"Hugging Face call failed: {resp.status_code} - {err_msg}", "latency": latency, "tokens_total": None}

    # parse success
    try:
        data = resp.json()
    except Exception:
        return {"text": "Hugging Face call failed: could not parse JSON response", "latency": latency, "tokens_total": None}

    # extract generated text robustly from common shapes
    text = ""
    try:
        if isinstance(data, dict):
            if "generated_text" in data:
                text = data["generated_text"]
            elif "text" in data:
                text = data["text"]
            elif "outputs" in data and isinstance(data["outputs"], list):
                first = data["outputs"][0]
                if isinstance(first, dict):
                    text = first.get("generated_text") or first.get("text") or str(first)
                else:
                    text = str(first)
            elif "results" in data and isinstance(data["results"], list):
                first = data["results"][0]
                if isinstance(first, dict):
                    text = first.get("generated_text") or first.get("text") or str(first)
                else:
                    text = str(first)
            else:
                # sometimes the router returns list shape although content is dict
                # fallback to stringify
                text = json.dumps(data)
        elif isinstance(data, list):
            first = data[0]
            if isinstance(first, dict):
                text = first.get("generated_text") or first.get("text") or str(first)
            else:
                text = str(first)
        else:
            text = str(data)
    except Exception:
        text = str(data)

    return {"text": text, "latency": latency, "tokens_total": None}

# -------------------- LLM wrapper (compat with max_tokens) --------------------
def call_openai_chat(prompt: str,
                     model: str = DEFAULT_HF_MODEL,
                     max_output_tokens: Optional[int] = None,
                     max_tokens: Optional[int] = None,
                     temperature: float = 0.6) -> dict:
    """
    Compatibility wrapper used by endpoints. Delegates to hf_inference.
    """
    return hf_inference(prompt=prompt, model=model, max_output_tokens=max_output_tokens, max_tokens=max_tokens, temperature=temperature)

# -------------------- Request models --------------------
class QARequest(BaseModel):
    question: str
    context: Optional[str] = None

# -------------------- Endpoints --------------------
@app.get("/ping")
def ping():
    return {"status": "ok", "time": time.time()}

@app.post("/qa")
def qa(req: QARequest):
    ctx = f"\n\nContext:\n{req.context}" if req.context else ""
    prompt = f"You are an expert tutor. Answer concisely and give a short explanation.\nQuestion: {req.question}{ctx}"
    out = call_openai_chat(prompt, max_tokens=300)
    log_metrics("qa", out.get("latency", 0), DEFAULT_HF_MODEL, out.get("tokens_total"), "")
    return {"text": out.get("text"), "latency": out.get("latency"), "tokens": out.get("tokens_total")}

@app.post("/explain")
def explain(body: dict):
    topic = body.get("topic", "")
    prompt = f"Explain the topic '{topic}' in simple steps with a short example and one-line summary."
    out = call_openai_chat(prompt, max_tokens=350)
    log_metrics("explain", out.get("latency", 0), DEFAULT_HF_MODEL, out.get("tokens_total"), "")
    return {"text": out.get("text"), "latency": out.get("latency"), "tokens": out.get("tokens_total")}

@app.post("/summarize")
def summarize(body: dict):
    text = body.get("text", "")
    prompt = f"Summarize the following text into 3 concise bullet points:\n\n{text}"
    out = call_openai_chat(prompt, max_tokens=220)
    log_metrics("summarize", out.get("latency", 0), DEFAULT_HF_MODEL, out.get("tokens_total"), "")
    return {"text": out.get("text"), "latency": out.get("latency"), "tokens": out.get("tokens_total")}

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    tmp = BASE_DIR / f"tmp_{int(time.time())}.png"
    with open(tmp, "wb") as f:
        f.write(contents)
    try:
        img = Image.open(tmp).convert("RGB")
        extracted = pytesseract.image_to_string(img)
    except Exception as e:
        return {"error": str(e)}
    log_metrics("upload_image", 0.0, "pytesseract", None, "ocr_extracted")
    return {"extracted_text": extracted}

@app.post("/image_qa")
async def image_qa(file: UploadFile = File(...), question: str = Form(...)):
    contents = await file.read()
    tmp = BASE_DIR / f"tmp_{int(time.time())}.png"
    with open(tmp, "wb") as f:
        f.write(contents)
    try:
        img = Image.open(tmp).convert("RGB")
        extracted = pytesseract.image_to_string(img)
    except Exception as e:
        return {"error": str(e)}
    prompt = f"Use the following extracted text as context, then answer the question with reasoning and a short summary.\n\nContext:\n{extracted}\n\nQuestion: {question}"
    out = call_openai_chat(prompt, max_tokens=350)
    log_metrics("image_qa", out.get("latency", 0), DEFAULT_HF_MODEL + " + pytesseract", out.get("tokens_total"), "")
    return {"extracted_text": extracted, "text": out.get("text"), "latency": out.get("latency"), "tokens": out.get("tokens_total")}

@app.post("/grade")
def grade(body: dict):
    q = body.get("question", "")
    ref = body.get("reference", "")
    stud = body.get("student_answer", "")
    prompt = (
        "You are an expert grader for short answers. "
        "Given the question, a reference answer, and a student answer, "
        "provide a numeric score from 0 to 5 (integer) and brief feedback. "
        f"Question: {q}\nReference: {ref}\nStudent: {stud}\n\n"
        "Return in this exact format:\nScore: <0-5>\nFeedback: <one sentence>"
    )
    out = call_openai_chat(prompt, max_tokens=180)
    log_metrics("grade", out.get("latency", 0), DEFAULT_HF_MODEL, out.get("tokens_total"), "")
    return {"text": out.get("text"), "latency": out.get("latency"), "tokens": out.get("tokens_total")}

# -------------------- startup --------------------
ensure_metrics_file()

if __name__ == "__main__":
    # Run without reload for stability on Windows (avoid possible multiprocess issues)
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
