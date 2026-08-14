from .dataset import build_structured_dataset
from .conversational_dataset import build_conversational_dataset
from .refusal_dataset import build_refusal_dataset
from .train import load_model, train_model, format_example, ADAPTER_PATH
from datasets import Dataset
from peft import PeftModel
import random, os, json
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "exercises.json")

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        exercises = json.load(f)

    model, tokenizer = load_model()

    if os.path.exists(ADAPTER_PATH):
        print("Adapter found, loading...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    else:
        print("Building dataset...")
        structured = build_structured_dataset(exercises)
        conversational = build_conversational_dataset(n=50)
        refusal = build_refusal_dataset(n=50)
        full = structured + conversational + refusal
        random.shuffle(full)

        formatted = [format_example(ex, tokenizer) for ex in full]
        dataset = Dataset.from_list(formatted)

        subprocess.run(["ollama", "stop", "llama3.1:8b"])

        model = train_model(model, dataset)

if __name__ == "__main__":
    main()
