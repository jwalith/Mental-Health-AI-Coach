#!/usr/bin/env python3
"""
RAG comparison – Gemini 1.5‑Flash vs DeepSeek‑LoRA
-------------------------------------------------
• FAISS dense retrieval (top‑2)
• Shared prompt (system + few‑shot + retrieved + question)
• Prints Gemini answer first, DeepSeek answer next
"""

import os, sys, textwrap, torch
from typing import List

import pandas as pd, faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          GenerationConfig, TextIteratorStreamer,
                          BitsAndBytesConfig)
from peft import PeftModel

# ── paths and small knobs ────────────────────────────────────────────────────
DATA_PATH            = "data"
FAISS_INDEX_PATH     = os.path.join(DATA_PATH, "faiss_index.bin")
EMBD_PATH            = os.path.join(DATA_PATH, "counselchat_with_embeddings.pkl")

BI_ENCODER_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_COLUMN          = "questionText"
TOP_K_FAISS          = 2

BASE_MODEL_ID        = "deepseek-ai/deepseek-llm-7b-chat"
LORA_ADAPTER_PATH    = "./lora-deepseek/final"

# ── prompt pieces ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are an empathetic AI focused on mental‑health support."
FEW_SHOT = [
    ("I feel anxious all the time. What can I do?",
     "I'm sorry you're feeling anxious. Deep‑breathing and grounding help…"),
    ("I can't sleep at night due to stress. Any suggestions?",
     "Stress can disrupt sleep. A consistent routine and relaxation aid…"),
    ("I'm struggling with work‑life balance. How do I cope?",
     "Set clear boundaries between work and personal time…"),
]
def prompt(question: str, retrieved: List[str]) -> str:
    few = "\n\n".join(f"User: {q}\nAssistant: {a}" for q, a in FEW_SHOT)
    chunks = "\n".join(f"- {c}" for c in retrieved) if retrieved else "None."
    return f"""{SYSTEM_PROMPT}

Few‑Shot Examples:
{few}

Retrieved information:
{chunks}

User's Question: {question}

Respond helpfully and empathetically."""

# ── retrieval stack ──────────────────────────────────────────────────────────
def load_retrieval():
    idx = faiss.read_index(FAISS_INDEX_PATH)
    df  = pd.read_pickle(EMBD_PATH)
    bi  = SentenceTransformer(BI_ENCODER_MODEL)
    return df, idx, bi
# ── retrieval stack ──────────────────────────────────────────────────────────
def retrieve(q: str, df, idx, bi) -> List[str]:
    # single 384‑dim row vector
    vec = bi.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    _, ids = idx.search(vec, TOP_K_FAISS)          # ids is shape (1, k)
    return df.iloc[ids[0]][TEXT_COLUMN].tolist()


# ── gemini ───────────────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
g_model = genai.GenerativeModel("gemini-1.5-flash")
def gemini_gen(p: str) -> str:
    t = g_model.generate_content(p).text.strip()
    return t

# ── deepseek‑lora ────────────────────────────────────────────────────────────
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID,
                                            quantization_config=bnb,
                                            device_map="auto",
                                            torch_dtype=torch.float16)
dsk = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
dsk.eval()
def deepseek_gen(p: str) -> str:
    gcfg = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.9, top_k=40,
                            max_new_tokens=128, num_return_sequences=1,
                            pad_token_id=tok.eos_token_id)
    ins = tok(p, return_tensors="pt").to(dsk.device)
    stream = TextIteratorStreamer(tok, skip_prompt=True)
    with torch.no_grad():
        dsk.generate(**ins, streamer=stream, **gcfg.to_dict())
    return "".join(stream).strip()

# ── chat loop ────────────────────────────────────────────────────────────────
df, idx, bi = load_retrieval()
print("Ask a question (quit to exit):")
while True:
    q = input("You: ").strip()
    if q.lower() in {"quit", "exit"}:
        break
    docs = retrieve(q, df, idx, bi)
    p = prompt(q, docs)
    print("\n━━ Gemini ━━")
    print(textwrap.fill(gemini_gen(p), 88), "\n")
    print("━━ DeepSeek‑LoRA ━━")
    print(textwrap.fill(deepseek_gen(p), 88), "\n")
