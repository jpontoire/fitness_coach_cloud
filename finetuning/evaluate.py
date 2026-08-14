from .train import load_model, SYSTEM_PROMPT, ADAPTER_PATH
from peft import PeftModel
import os

def generate_response(model, tokenizer, question, max_new_tokens=300):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def main():
    model, tokenizer = load_model()

    if not os.path.exists(ADAPTER_PATH):
        print("No adapter found, train first with main.py")
        return

    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "is_gradient_checkpointing") and model.is_gradient_checkpointing:
        model.gradient_checkpointing_disable()

    structured_q = """Based on this exercise data, describe dumbbell bench press.

Context: Exercise: dumbbell bench press
Category: chest
Target: pectorals
Secondary muscles: triceps, shoulders
Equipment: dumbbell
Instructions: Lie flat on a bench with your feet flat on the ground..."""

    conversational_q = "How much rest should I take between sets for muscle growth?"
    off_topic_q = "What's the best programming language to learn in 2026?"

    for label, q in [("Structured", structured_q), ("Conversational", conversational_q), ("Off-topic", off_topic_q)]:
        print(f"=== {label} ===")
        print(f"Q: {q[:100]}...")
        print(f"A: {generate_response(model, tokenizer, q)}")
        print("---")

if __name__ == "__main__":
    main()
