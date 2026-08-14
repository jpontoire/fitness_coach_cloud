import ollama
import os
import json

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama.Client(host=OLLAMA_HOST)

def generate_conversational_example():
    prompt = """Generate a realistic question a beginner or intermediate gym-goer might ask about fitness, training principles, nutrition, or recovery — NOT about a specific exercise's instructions. Then provide a natural, conversational answer (not a structured list).

Format your response exactly as:
Q: <question>
A: <answer>"""
    response = ollama_client.generate(model="llama3.1:8b", prompt=prompt)["response"]
    return response

def parse_example(raw_response):
    lines = raw_response.strip().split("\n")
    question, answer = None, None
    for line in lines:
        if line.startswith("Q:"):
            question = line[2:].strip()
        elif line.startswith("A:"):
            answer = line[2:].strip()
    return question, answer

def build_conversational_dataset(n=150, cache_path="cache/conversational.json"):
    if os.path.exists(cache_path):
        print(f"Found cache ({cache_path}), loading...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("No cache found, generating...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    examples = []
    for i in range(n):
        raw = generate_conversational_example()
        question, answer = parse_example(raw)
        if question and answer:
            examples.append({"question": question, "answer": answer})

        if (i + 1) % 10 == 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
            print(f"Generated and saved {i + 1}/{n}")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    return examples
