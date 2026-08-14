from rag.retrieval import retrieve

import re
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from openai import OpenAI

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

DEFAULT_MUSCLE_GROUPS = ["chest", "back", "upper legs", "shoulders", "waist"]
PPL_GROUPS = {
    "push": ["chest", "shoulders", "upper arms"],   # triceps is in "upper arms"
    "pull": ["back", "upper arms", "lower arms"],    # biceps is in "upper arms", anterior biceps is in "lower arms"
    "legs": ["upper legs", "lower legs"],
}

class ProgramRequest(BaseModel):
    muscles: List[str]
    equipment: List[str]
    preferences: Optional[str] = None

class ProgramExercise(BaseModel):
    name: str
    sets: int
    reps: str
    target_muscle: str

class WorkoutProgram(BaseModel):
    exercises: List[ProgramExercise]
    notes: Optional[str] = None


def extract_program_params(question):
    prompt = f"""Extract structured information from this workout request.
If the request mentions a "push", "pull", or "legs" day/split, map it to muscle groups as follows:
- push → chest, shoulders, upper arms
- pull → back, upper arms, lower arms
- legs → upper legs, lower legs
Otherwise, use these muscle group values when applicable: chest, back, shoulders, upper arms, lower arms, upper legs, lower legs, waist, cardio, neck.
Use these equipment values when applicable: dumbbell, barbell, body weight, cable, band, kettlebell, machine, other.
If not specified, use an empty list for muscles/equipment.
Also extract any specific preferences or constraints mentioned (e.g. "avoid squats", "focus on isolation exercises") as a short free-text note. If none, use null.
Request: {question}
Respond with ONLY a JSON object like this: {{"muscles": ["chest"], "equipment": ["dumbbell"], "preferences": null}}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    response = response.choices[0].message.content.strip()

    match = re.search(r'\{.*?\}', response, re.DOTALL)
    if not match:
        print(f"No JSON found in response: {response}")
        return ProgramRequest(muscles=[], equipment=[])

    json_str = match.group()

    try:
        data = json.loads(json_str)
        validated = ProgramRequest(**data)
        return validated
    except Exception as e:
        print(f"Extraction failed: {e}, JSON extracted: {json_str}")
        return ProgramRequest(muscles=[], equipment=[])

def retrieve_for_program(muscles, equipment, collection, embedding_model, k_per_muscle=5):
    target_muscles = muscles if muscles else DEFAULT_MUSCLE_GROUPS

    where = {"equipment": {"$in": equipment}} if equipment else None

    all_candidates = {}
    for muscle in target_muscles:
        query = f"{muscle} exercise"

        results = retrieve(query, collection, embedding_model, k=k_per_muscle, where=where)

        candidates = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            candidates.append({"text": doc, "name": meta["name"]})

        all_candidates[muscle] = candidates

    return all_candidates

def extract_first_json_object(text):
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def generate_program(candidates, preferences, max_retries=2):
    candidates_text = ""
    for muscle, exercises in candidates.items():
        candidates_text += f"\n{muscle.upper()}:\n"
        for ex in exercises:
            candidates_text += f"- {ex['name']}\n"

    preferences_text = f"\nUser preferences/constraints: {preferences}" if preferences else ""

    prompt = f"""Based on the following available exercises, create a structured workout program.
    Select 2-3 exercises per muscle group. Respect any user preferences or constraints.{preferences_text}
    Available exercises:{candidates_text}
    Respond with ONLY a JSON object matching this exact structure. You need to write an actual descriptive title, not placeholder text. Also, you must include valid exercise names and target muscles, do not copy and paste the exact following structure without making any changes:
    {{
        "title": "Upper and Lower Legs Workout",
        "exercises": [
        {{"name": "...", "sets": 3, "reps": "8-12", "target_muscle": "..."}}
        ],
        "notes": "..."
    }}
    IMPORTANT: You MUST include between 2 and 3 exercises for EACH muscle group listed above, not just one exercise total."""

    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        response = response.choices[0].message.content

        json_str = extract_first_json_object(response)
        if json_str:
            try:
                data = json.loads(json_str)
                program = WorkoutProgram(**data)
                return program
            except Exception as e:
                print(f"Try {attempt + 1} failed: {e}")
                print(f"Raw response: {response}")
                continue
        else:
            print(f"No JSON found. Raw response: {response}")

    return None
