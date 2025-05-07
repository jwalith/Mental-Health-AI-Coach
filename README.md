# Mental Health AI Coach (RAG + LoRA Fine-Tuning)

This project implements a **personalized, empathetic mental health coach** using a fine-tuned version of **DeepSeek-7B**. It combines **Retrieval-Augmented Generation (RAG)**, **LoRA-based fine-tuning**, and **response reranking** with Gemini 2.0 Flash.

## 🔧 Features

- 🔍 **Semantic Retrieval**: FAISS index + bi-encoder (MiniLM) to retrieve contextually similar mental health Q&A.
- 🧠 **LLM Fine-Tuning**: DeepSeek-7B fine-tuned with LoRA (on Q_proj, V_proj) using real-world mental health conversations.
- 💡 **RAG Pipeline**:
  - Generate 3 diverse responses using DeepSeek-LoRA
  - Rerank using Gemini 2.0 Flash (or DeepSeek R1 via Ollama)
- ⚡ **4-Bit Inference**: Efficient inference using `bitsandbytes` with NF4 + double quantization on a 16GB GPU.
- 🔒 **Session Encryption**: AES-based `Fernet` encryption used for protecting user data in Streamlit sessions.

## 📦 Tech Stack

- `transformers` + `peft` + `bitsandbytes`
- `faiss`, `sentence-transformers`
- `streamlit` for UI
- `cryptography` for secure session handling
- `google.generativeai` for Gemini reranking

## 🧪 Example Use

Ask a question like:

```txt
I'm feeling overwhelmed with work and can't focus.
```

And get a grounded, personalized response pulled from real-world cases, fine-tuned with empathy.

## 🧠 Architecture

1. **Input → Embedding → FAISS Search**
2. **Retrieve Top-K Relevant Questions**
3. **Construct Prompt + Few-Shot Examples**
4. **Generate 3 Candidates via DeepSeek-7B-LoRA**
5. **Rerank with Gemini 2.0 Flash**
6. **Return Best Response to User**

## 🔐 Security

Session-level user profile and chat history are encrypted using `Fernet` (AES 128) for safe interaction and privacy preservation.

## 📂 Directory Structure

```
/app
├── chatbot_logic.py
├── main.py
├── faiss_index.bin
├── counselchat_with_embeddings.pkl
├── lora-deepseek/
└── final/
```

## 📌 Future Improvements

- Summarization-based memory compression
- Long-form context (DeepSeek-Prover-V2)
- Real-time user mood tracking and topic memory

## 🙋 Author

Jwalith, MS in Data Science  
Stony Brook University  
