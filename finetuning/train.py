import os
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(SCRIPT_DIR, "fitness-lora-final")
SYSTEM_PROMPT = "You are a fitness assistant that helps with exercises, training, and nutrition questions."

def format_example(example, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    return model, tokenizer

def train_model(model, dataset):
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.05, task_type="CAUSAL_LM")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    training_args = SFTConfig(
        output_dir=os.path.join(SCRIPT_DIR, "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=768,
    )

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    trainer.save_model(ADAPTER_PATH)
    return model
