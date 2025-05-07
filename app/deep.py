#!/usr/bin/env python3
"""
RAG chatbot – DeepSeek‑7B‑chat + LoRA adapter only
-------------------------------------------------
• FAISS dense retrieval (top‑2 chunks)
• Sentence‑Transformers bi‑encoder
• DeepSeek‑LoRA generator (4‑bit)
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os, sys, textwrap, torch
from typing import List
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          GenerationConfig,
                          TextIteratorStreamer)
from peft import PeftModel
from transformers import BitsAndBytesConfig

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH            = "data"
FAISS_INDEX_PATH     = os.path.join(DATA_PATH, "faiss_index.bin")
EMBD_PATH            = os.path.join(DATA_PATH, "counselchat_with_embeddings.pkl")

BI_ENCODER_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_COLUMN          = "questionText"           # change if you prefer answerText
TOP_K_FAISS          = 2                        # nearest chunks to keep

BASE_MODEL_ID        = "deepseek-ai/deepseek-llm-7b-chat"
LORA_ADAPTER_PATH    = "./lora-deepseek/final"  # path to your trained adapter

# ── Prompts ──────────────────────────────────────────────────────────────────
# SYSTEM_PROMPT_GENERAL = """
# You are an empathetic AI focused on mental health support. Provide concise,
# supportive, evidence‑based guidance tailored to the user's emotional state.
# """
SYSTEM_PROMPT_GENERAL = """
You are an empathetic AI focused on mental health support. Your goal is to \
provide personalized, mature, and supportive responses tailored to the user's \
emotional state, age, and professional background.

Behavior Guidelines:
1. Personalization: Adapt your responses to the user's age and professional back‑
   ground (relatable support for students, nuanced advice for professionals).
2. Empathy: Detect emotional cues and respond with genuine encouragement.
3. Evidence‑Based Advice: Base guidance on established psychological research.
4. Self‑Reflection: Encourage users to explore their thoughts and emotions.
5. Positive Outlook: Acknowledge challenges while guiding toward solutions.
6. Targeted Support: Address academic pressure, career stress, etc.
7. Holistic Wellness: Promote sleep, nutrition, exercise with practical tips.
8. Inspirational Content: Uplifting stories and simple mental‑wellbeing tips.
9. Community Impact: Highlight positive societal impact of personal growth.
10. Topic Focus: Gently redirect off‑topic questions back to mental health.

Response Style:
• Concise yet impactful.
• Tone sensitive to emotion.
• No meta‑commentary.
"""

FEW_SHOT_EXAMPLES = [
    {
        "user": "I feel anxious all the time. What can I do?",
        "assistant": "I'm sorry you're feeling anxious. Deep‑breathing, grounding..."
    },
    {
        "user": "I can't sleep at night due to stress. Any suggestions?",
        "assistant": "Stress can disrupt sleep. Try a consistent routine..."
    },
    {
        "user": "I'm struggling with work‑life balance. How do I cope?",
        "assistant": "Setting clear boundaries between work and personal time helps..."
    }
]

def fewshot_block() -> str:
    return "\n\n".join(f"User: {ex['user']}\nDeepSeek Assistant: {ex['assistant']}"
                       for ex in FEW_SHOT_EXAMPLES)

PROMPT_TEMPLATE = """
{system_prompt}

Few‑Shot Examples:
{fewshot}

Retrieved information:
{retrieved}

User's Question: {question}

Respond helpfully and empathetically.
"""

# ── Load retrieval stack ─────────────────────────────────────────────────────
def load_retrieval():
    index = faiss.read_index(FAISS_INDEX_PATH)
    df    = pd.read_pickle(EMBD_PATH)
    bi    = SentenceTransformer(BI_ENCODER_MODEL)
    return df, index, bi

def retrieve_chunks(query: str, df, index, bi) -> List[str]:
    q_vec = bi.encode([query], normalize_embeddings=True).astype("float32")
    _, ids = index.search(q_vec, TOP_K_FAISS)
    return df.iloc[ids[0]][TEXT_COLUMN].tolist()

# ── Load DeepSeek with LoRA adapter ──────────────────────────────────────────
def load_deepseek():
    print("Loading DeepSeek‑LoRA …", flush=True)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
    model.eval()
    print("Model device:", next(model.parameters()).device)
    return tok, model

# ── Generation ───────────────────────────────────────────────────────────────
def deepseek_generate(prompt: str, tok, model) -> str:
    gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_new_tokens=128,
        num_return_sequences=1,
        pad_token_id=tok.eos_token_id,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True)
    with torch.no_grad():
        model.generate(**inputs, streamer=streamer, **gen_cfg.to_dict())
    return "".join(chunk for chunk in streamer).strip()

# ── Chat loop ────────────────────────────────────────────────────────────────
def chat():
    df, index, bi = load_retrieval()
    tok, model    = load_deepseek()

    print("\nAsk a mental‑health question (type 'quit' to exit):")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        docs   = retrieve_chunks(q, df, index, bi)
        prompt = PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT_GENERAL.strip(),
            fewshot=fewshot_block(),
            retrieved="\n".join(f"- {d}" for d in docs) if docs else "None.",
            question=q,
        )
        ans = deepseek_generate(prompt, tok, model)
        print("\nAssistant:\n" + textwrap.fill(ans, width=88) + "\n")

# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        chat()
    except KeyboardInterrupt:
        print("\nBye!")
