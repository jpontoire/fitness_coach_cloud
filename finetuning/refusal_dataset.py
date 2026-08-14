import ollama
import os
import json

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama.Client(host=OLLAMA_HOST)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

refusal_answer = "I'm a fitness assistant focused on exercises, training, and nutrition. I can't help with that, but feel free to ask me anything about workouts or muscle building!"

def generate_off_topic_question():
    prompt = """Generate a single realistic question a person might ask an AI assistant, on a topic completely unrelated to fitness, exercise, or nutrition (e.g. cooking, coding, history, geography, entertainment). Return ONLY the question, nothing else."""
    response = ollama_client.generate(model="llama3.1:8b", prompt=prompt)["response"].strip()
    return response

def build_refusal_dataset(n=150, cache_path=None):
    if cache_path is None:
        cache_path = os.path.join(SCRIPT_DIR, "cache", "refusal.json")

    if os.path.exists(cache_path):
        print(f"Found cache ({cache_path}), loading...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("No cache found, generating...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    examples = []
    for i in range(n):
        question = generate_off_topic_question()
        examples.append({"question": question, "answer": refusal_answer})
        if (i + 1) % 10 == 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
            print(f"Generated and saved {i + 1}/{n}")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    return examples
