import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from transformers import BitsAndBytesConfig
import os
import json
from peft import prepare_model_for_kbit_training

# Load tokenizer and model
model_id = "deepseek-ai/deepseek-llm-7b-chat"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# Load and preprocess dataset
def format_example(example):
    prompt = f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['response']}"
    return tokenizer(prompt, padding="max_length", truncation=True, max_length=512)

dataset = load_dataset("json", data_files=r"C:\Users\jwali\Downloads\AMS 691_LLM\counselchat_lora.jsonl", split="train")
dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora-deepseek",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    resume_from_checkpoint=True,  # ✅ Resume from last checkpoint
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train the model
# trainer.train()
trainer.train(resume_from_checkpoint="./lora-deepseek/checkpoint-100")

# from pathlib import Path

# def get_last_checkpoint(path="./lora-deepseek"):
#     checkpoints = list(Path(path).glob("checkpoint-*"))
#     if not checkpoints:
#         return None
#     return str(sorted(checkpoints, key=lambda x: int(x.name.split('-')[-1]))[-1])

# last_ckpt = get_last_checkpoint()
# if last_ckpt:
#     trainer.train(resume_from_checkpoint=last_ckpt)
# else:
#     trainer.train()


# Save the adapter
model.save_pretrained("./lora-deepseek/final")