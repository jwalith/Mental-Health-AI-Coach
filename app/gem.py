#!/usr/bin/env python3
"""
RAG Mental‑Health Chatbot (Gemini + FAISS)

• Dense retrieval with Sentence‑Transformers bi‑encoder
• FAISS index for ANN search
• Cross‑encoder re‑ranking
• Gemini 1.5‑Flash for answer generation
-------------------------------------------------------
Usage:
  $ export GEMINI_API_KEY="your_key"
  $ python rag_mental_health_chatbot.py
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os
import sys
import json
import textwrap
from typing import List, Tuple

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai   # pip install google-generativeai

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH            = "data"                   # edit if you store assets elsewhere
FAISS_INDEX_PATH     = os.path.join(DATA_PATH, "faiss_index.bin")
EMBD_PATH            = os.path.join(DATA_PATH, "counselchat_with_embeddings.pkl")

# You can swap in any ST model you used when building the index
BI_ENCODER_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TEXT_COLUMN          = "questionText"
EMBEDDING_COLUMN     = "question_embedding"

TOP_K_INITIAL        = 10   # vectors from FAISS
TOP_K_RERANK         = 4    # after cross‑encoder re‑ranking
NUM_GEN_CANDIDATES   = 3    # how many diverse answers Gemini should draft

# ── System + Few‑shot Prompts (from your snippet) ─────────────────────────────
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
        "user_question": "I feel anxious all the time. What can I do?",
        "bot_response": "I'm sorry to hear that you're feeling anxious. Anxiety can be overwhelming, "
                        "but there are ways to manage it. Deep breathing exercises, grounding techniques, "
                        "and seeking support from a professional can help. Would you like to learn some "
                        "specific techniques?"
    },
    {
        "user_question": "I can't sleep at night due to stress. Any suggestions?",
        "bot_response": "Stress can make it really difficult to sleep. Have you tried creating a bedtime "
                        "routine, such as avoiding screens before bed or practicing relaxation techniques? "
                        "I'd be happy to share more strategies if you'd like."
    },
    {
        "user_question": "I'm struggling with work‑life balance. How do I cope?",
        "bot_response": "Finding work‑life balance can be tough. Start by setting clear boundaries between "
                        "work and personal time. Scheduling breaks and prioritizing self‑care can also help. "
                        "Can you tell me more about what's making it hard to balance?"
    }
]

GENERATION_PROMPT_TEMPLATE = """
{system_prompt}

Few‑Shot Examples:
{few_shot_prompt}

Retrieved information:
{retrieved_chunks}

User's Question: {query}

Please generate {num_candidates} diverse, empathetic, and helpful response candidates to the user's question. If the retrieved information is relevant, ensure the responses are consistent with it. If no relevant information was found, answer based on your general knowledge and persona as a mental health support AI.

Format your output clearly, numbering each candidate like this:

1. [First response candidate]
2. [Second response candidate]
3. [Third response candidate]
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_resources():
    """Load FAISS, embeddings DF, encoders, and Gemini model."""
    print("Loading resources … ", end="", flush=True)
    index = faiss.read_index(FAISS_INDEX_PATH)
    data  = pd.read_pickle(EMBD_PATH)           # expects columns: text, embedding (list/ndarray)
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    g_model = genai.GenerativeModel("gemini-1.5-flash")
    print("done.")
    return data, index, bi_encoder, cross_encoder, g_model


def embed_query(query: str, bi_encoder: SentenceTransformer):
    """Return a normalized vector for the query."""
    emb = bi_encoder.encode([query], normalize_embeddings=True)
    return emb.astype("float32")


def search_index(query_emb, index, k=TOP_K_INITIAL) -> Tuple[List[int], List[float]]:
    """FAISS ANN search → indices of docs and distances (smaller=closer)."""
    distances, doc_indices = index.search(query_emb, k)
    return doc_indices[0].tolist(), distances[0].tolist()


def rerank(query: str, docs: List[str], cross_encoder: CrossEncoder, top_n=TOP_K_RERANK):
    """Cross‑encoder scores → top_n highest‑scored docs."""
    pairs = [[query, d] for d in docs]
    scores = cross_encoder.predict(pairs)  # higher = more relevant
    scored = sorted(zip(docs, scores), key=lambda t: t[1], reverse=True)
    return [d for (d, s) in scored[:top_n]]


def make_fewshot_block() -> str:
    """Combine few‑shot examples for the prompt."""
    return "\n\n".join(f"User: {ex['user_question']}\nAssistant: {ex['bot_response']}"
                       for ex in FEW_SHOT_EXAMPLES)


def build_generation_prompt(query: str,
                            retrieved: List[str],
                            num_candidates: int = NUM_GEN_CANDIDATES) -> str:
    retrieved_block = "\n".join(f"- {chunk}" for chunk in retrieved) if retrieved else "None."
    return GENERATION_PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT_GENERAL.strip(),
        few_shot_prompt=make_fewshot_block(),
        retrieved_chunks=retrieved_block,
        query=query.strip(),
        num_candidates=num_candidates
    )


def generate_answers(prompt: str, g_model, **gen_kwargs) -> List[str]:
    """Call Gemini and return the numbered answers parsed as a list."""
    response = g_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=1024,
            top_p=1.0,
            top_k=40,
        )
    )
    text = response.text.strip()
    # Expect numbered list "1. … 2. … 3. …"
    answers = []
    for line in text.splitlines():
        if line.lstrip().startswith(tuple("123456789")):
            # Remove leading number + dot
            answers.append(line.lstrip().split(maxsplit=1)[1] if ". " in line else line)
    return answers or [text]   # fallback: whole text if parsing failed


# ── Interactive Loop ──────────────────────────────────────────────────────────
def chat():
    data, index, bi_encoder, cross_encoder, g_model = load_resources()
    print("\nType your question (or 'quit'): ")
    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        # ① Dense retrieval
        q_emb = embed_query(user_q, bi_encoder)
        nn_ids, _ = search_index(q_emb, index)
        retrieved_docs = data.iloc[nn_ids][TEXT_COLUMN].tolist()

        # ② Re‑rank
        reranked = rerank(user_q, retrieved_docs, cross_encoder)

        # ③ Build prompt & generate
        gen_prompt = build_generation_prompt(user_q, reranked)
        answers = generate_answers(gen_prompt, g_model)

        # ④ Display
        print("\nGemini Assistant:")
        for ans in answers:
            print(textwrap.fill(ans, width=88))
            print("-" * 88)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    missing_key = "GEMINI_API_KEY" not in os.environ
    if missing_key:
        sys.stderr.write("❌ Please set GEMINI_API_KEY before running.\n")
        sys.exit(1)
    try:
        chat()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Bye!")
