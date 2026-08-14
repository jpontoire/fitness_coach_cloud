import os
import ollama

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama.Client(host=OLLAMA_HOST)

def router_node(state):
    prompt = f"""Classify this question into exactly one category:
- exercise_lookup: asking about a specific exercise (how to do it, what muscles it targets)
- program_request: asking for a workout plan/program with multiple exercises
- conversational: general fitness/nutrition/training question, not about a specific exercise
- off_topic: unrelated to fitness

Question: {state['question']}

Answer with only the category name."""

    response = ollama_client.generate(model="llama3.1:8b", prompt=prompt)["response"].strip().lower()

    valid_intents = ["exercise_lookup", "program_request", "conversational", "off_topic"]
    intent = next((i for i in valid_intents if i in response), "conversational")

    return {"intent": intent}
