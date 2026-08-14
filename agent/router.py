import os
from openai import OpenAI

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

def router_node(state):
    prompt = f"""Classify this question into exactly one category:
- exercise_lookup: asking about a specific exercise (how to do it, what muscles it targets)
- program_request: asking for a workout plan/program with multiple exercises
- conversational: general fitness/nutrition/training question, not about a specific exercise
- off_topic: unrelated to fitness

Question: {state['question']}

Answer with only the category name."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    response = response.choices[0].message.content.strip()

    valid_intents = ["exercise_lookup", "program_request", "conversational", "off_topic"]
    intent = next((i for i in valid_intents if i in response), "conversational")

    return {"intent": intent}
