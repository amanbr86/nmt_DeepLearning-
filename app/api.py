"""FastAPI service.   Run from the project root:  uvicorn app.api:app --host 0.0.0.0 --port 8000
   curl -X POST localhost:8000/translate -H 'Content-Type: application/json' -d '{"text": "I am going to the university."}'"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import Translator

MODEL = os.getenv("NMT_MODEL", "attention")
CKPT_DIR = os.getenv("NMT_CKPT_DIR", "checkpoints")
BEAM = int(os.getenv("NMT_BEAM", "1"))
state = {}


@asynccontextmanager
async def lifespan(app):
    state["translator"] = Translator(MODEL, CKPT_DIR, beam=BEAM)   # load once at startup
    yield
    state.clear()


app = FastAPI(title="English → Amharic NMT", lifespan=lifespan)


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    translation: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "beam": BEAM}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' must not be empty")
    if len(text) > 1000:
        raise HTTPException(status_code=400, detail="'text' too long (max 1000 characters)")
    return {"translation": state["translator"].translate(text)}
