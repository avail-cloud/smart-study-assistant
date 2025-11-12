# app.py
"""
Smart Study Assistant - FastAPI microservice (updated for openai>=1.0.0 and .env)
Endpoints:
  POST /qa, /explain, /summarize, /upload_image, /image_qa, /grade
Logs to metrics.csv
"""

import os
import time
import csv
import json
import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import pytesseract
import uvicorn

# load .env
from dotenv import load_dotenv
from pathlib import Path
# ensure we load the .env located in the same folder as this file (ms/.env)
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)  # loads variables from .env into environment

# OpenAI new client
from openai import OpenAI

# Read env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found in environment. Ensure ms/.env contains it and restart.")
# configure pytesseract exe path (Windows)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# create OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI app and CORS
app = FastAPI(title="Smart Study Assistant - MS (updated)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

METRICS_FILE = "metrics.csv"
METRICS_HEADER = ["timestamp_utc","endpoint","latency_s","model","tokens_total","notes"]

def ensure_metrics_file():
    if not os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(METRICS_HEADER)

def log_metrics(endpoint, latency, model, tokens_total, notes=""):
    ensure_metrics_file()
    row = [
        datetime.datetime.utcnow().isoformat(),
        endpoint,
        f"{latency:.4f}",
        model or "",
        tokens_total or "",
        notes
    ]
    with open(METRICS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def call_openai_chat(prompt, model="gpt-4o-mini", max_output_tokens: Optional[int] = None,
                     max_tokens: Optional[int] = None, temperature=0.2):
    """
    Compatibility wrapper for OpenAI Responses API.
    Accepts either `max_output_tokens` (new) or `max_tokens` (old callers).
    Returns dict: {'text', 'latency', 'tokens_total'}
    """
    # Decide which token param to send
    if max_output_tokens is None and max_tokens is not None:
        max_output_tokens = max_tokens

    # default if still None
    if max_output_tokens is None:
        max_output_tokens = 256

    start = time.time()
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )
    except Exception as e:
        latency = time.time() - start
        return {"text": f"LLM call failed: {e}", "latency": latency, "tokens_total": None, "error": str(e)}

    latency = time.time() - start

    # Extract text
    text = ""
    try:
        if hasattr(resp, "output_text") and resp.output_text:
            text = resp.output_text
        else:
            out = getattr(resp, "output", None) or (resp.get("output", None) if isinstance(resp, dict) else None)
            if out:
                if isinstance(out, list):
                    for block in out:
                        if isinstance(block, dict):
                            content = block.get("content", [])
                            for piece in content:
                                if isinstance(piece, dict):
                                    txt = piece.get("text") or piece.get("content") or ""
                                    text += txt
                                else:
                                    text += str(piece)
                        else:
                            text += str(block)
                else:
                    text = str(out)
            else:
                choices = resp.get("choices", None) if isinstance(resp, dict) else getattr(resp, "choices", None)
                if choices:
                    c = choices[0]
                    msg = c.get("message", {}) if isinstance(c, dict) else None
                    if msg:
                        text = msg.get("content","") if isinstance(msg, dict) else str(msg)
                    else:
                        text = str(c)
    except Exception:
        text = str(resp)

    # tokens/usage
    tokens_total = None
    try:
        usage = resp.get("usage", None) if isinstance(resp, dict) else getattr(resp, "usage", None)
        if usage:
            tokens_total = usage.get("total_tokens", None) if isinstance(usage, dict) else None
    except Exception:
        tokens_total = None

    return {"text": text, "latency": latency, "tokens_total": tokens_total}

# ---- Request models ----
class QARequest(BaseModel):
    question: str
    context: Optional[str] = None

# ---- Endpoints ----
@app.get("/ping")
def ping():
    return {"status":"ok","time": time.time()}

@app.post("/qa")
def qa(req: QARequest):
    ctx = f"\n\nContext:\n{req.context}" if req.context else ""
    prompt = f"You are an expert tutor. Answer concisely and give a short explanation.\nQuestion: {req.question}{ctx}"
    out = call_openai_chat(prompt, max_tokens=300)
    log_metrics("qa", out.get("latency", 0), "gpt-4o-mini", out.get("tokens_total"), "")
    return {"text": out.get("text"), "latency": out.get("latency"), "tokens": out.get("tokens_total")}

@app.post("/explain")
def explain(body: dict):
    topic = body.get("topic", "")
    prompt = f"Explain the topic '{topic}' in simple steps with a short example and one-line summary."
    out = call_openai_chat(prompt, max_tokens=350)
    log_metrics("explain", out.get("latency",0), "gpt-4o-mini", out.get("tokens_total"), "")
    return {"text": out.get("text"), "latency": out.get("latency"), "tokens": out.get("tokens_total")}

@app.post("/summarize")
def summarize(body: dict):
    text = body.get("text", "")
    prompt = f"Summarize the following text into 3 concise bullet points:\n\n{text}"
    out = call_openai_chat(prompt, max_tokens=220)
    log_metrics("summarize", out.get("latency",0), "gpt-4o-mini", out.get("tokens_total"), "")
    return {"text": out.get("text"), "latency": out.get("latency"), "tokens": out.get("tokens_total")}

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    tmp = f"tmp_{int(time.time())}.png"
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
    tmp = f"tmp_{int(time.time())}.png"
    with open(tmp, "wb") as f:
        f.write(contents)
    try:
        img = Image.open(tmp).convert("RGB")
        extracted = pytesseract.image_to_string(img)
    except Exception as e:
        return {"error": str(e)}
    prompt = f"Use the following extracted text as context, then answer the question with reasoning and a short summary.\n\nContext:\n{extracted}\n\nQuestion: {question}"
    out = call_openai_chat(prompt, max_tokens=350)
    log_metrics("image_qa", out.get("latency",0), "gpt-4o-mini + pytesseract", out.get("tokens_total"), "")
    return {"extracted_text": extracted, "text": out.get("text"), "latency": out.get("latency"), "tokens": out.get("tokens_total")}

@app.post("/grade")
def grade(body: dict):
    q = body.get("question","")
    ref = body.get("reference","")
    stud = body.get("student_answer","")
    prompt = (
        "You are an expert grader for short answers. "
        "Given the question, a reference answer, and a student answer, "
        "provide a numeric score from 0 to 5 (integer) and brief feedback. "
        f"Question: {q}\nReference: {ref}\nStudent: {stud}\n\n"
        "Return in this exact format:\nScore: <0-5>\nFeedback: <one sentence>"
    )
    out = call_openai_chat(prompt, max_tokens=180)
    log_metrics("grade", out.get("latency",0), "gpt-4o-mini", out.get("tokens_total"), "")
    return {"text": out.get("text"), "latency": out.get("latency"), "tokens": out.get("tokens_total")}

# Ensure metrics file exists
ensure_metrics_file()

# Run
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
