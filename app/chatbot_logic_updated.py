import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pandas as pd
import google.generativeai as genai
import os
import streamlit as st
from cryptography.fernet import Fernet

# ---------------------- Prompt Configuration ----------------------

SYSTEM_PROMPT_GENERAL = """
You are an empathetic AI focused on mental health support. Provide personalized, mature, and supportive responses tailored to the user's emotional state, age, and professional background.
"""

FEW_SHOT_EXAMPLES = [
    {
        "user_question": "I feel anxious all the time. What can I do?",
        "bot_response": "I'm sorry to hear that you're feeling anxious. Deep breathing exercises, grounding techniques, and seeking support from a professional can help."
    },
    {
        "user_question": "I can't sleep at night due to stress. Any suggestions?",
        "bot_response": "Stress can make it difficult to sleep. Try creating a bedtime routine or practicing relaxation techniques. Would you like a few?"
    }
]

GENERATION_PROMPT_TEMPLATE = """
{system_prompt}

Few-Shot Examples:
{few_shots}

Retrieved information:
{retrieved}

User's Question: {query}

Please generate thoughtful, empathetic, and helpful responses directly addressing the user's concern.
"""

def format_few_shots():
    return "\n".join(f"User: {e['user_question']}\nAssistant: {e['bot_response']}" for e in FEW_SHOT_EXAMPLES)

# ---------------------- Model Loaders ----------------------

def load_resources():
    DATA_PATH = "data"
    FAISS_INDEX_PATH = f"{DATA_PATH}/faiss_index.bin"
    EMBD_PATH = f"{DATA_PATH}/counselchat_with_embeddings.pkl"

    index = faiss.read_index(FAISS_INDEX_PATH)
    data = pd.read_pickle(EMBD_PATH)
    bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    BASE_MODEL_ID = "deepseek-ai/deepseek-llm-7b-chat"
    LORA_PATH = "./lora-deepseek/final"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    return data, index, bi_encoder, cross_encoder, tokenizer, model

# ---------------------- DeepSeek Generator ----------------------

def deepseek_generate_multiple(prompt: str, tokenizer, model, n=3):
    gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_new_tokens=256,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_cfg.to_dict())
    return [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]

# ---------------------- Gemini Re-ranker ----------------------

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
g_model = genai.GenerativeModel("gemini-1.5-flash")

def gemini_rerank(query, candidates):
    prompt = f"""
You are ranking assistant replies for a mental health chatbot.

User Question:
{query}

Candidates:
{chr(10).join([f"{i+1}. {c}" for i, c in enumerate(candidates)])}

Please rank these responses from most helpful to least helpful. 
Return only the number of the best candidate.
"""
    try:
        response = g_model.generate_content(prompt)
        top_choice = int(response.text.strip()[0]) - 1
        return candidates[top_choice]
    except:
        return candidates[0]

# ---------------------- Chat Function ----------------------

def conversation_chat(query, data, index, bi_encoder, cross_encoder, tokenizer, model, cipher):
    context_embedding = bi_encoder.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(context_embedding, 4)
    retrieved_passages = data.iloc[ids[0]]["questionText"].tolist()

    few_shots = format_few_shots()
    generation_prompt = GENERATION_PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT_GENERAL,
        few_shots=few_shots,
        retrieved="\n".join(f"- {txt}" for txt in retrieved_passages),
        query=query
    )

    candidates = deepseek_generate_multiple(generation_prompt, tokenizer, model, n=3)
    best_response = gemini_rerank(query, candidates)
    return best_response

def load_encryption_key():
    return Fernet.generate_key()

def collect_user_profile(cipher):
    if "user_profile" not in st.session_state:
        st.session_state["user_profile"] = {"name": "User", "specific_concern": "stress"}

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []