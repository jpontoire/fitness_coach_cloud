import json

def load_exercises(path):
    with open(path, "r", encoding="utf-8") as f:
        exercises = json.load(f)
    return exercises

def build_chunk(exercise):
    text = f"""Exercise: {exercise['name']}
Category: {exercise['category']} ({exercise['body_part']})
Target muscle: {exercise['target']}
Secondary muscles: {', '.join(exercise['secondary_muscles'])}
Equipment: {exercise['equipment']}
Instructions: {exercise['instructions']['en']}"""
    return text

def build_metadata(exercise):
    return {
        "id": exercise["id"],
        "name": exercise["name"],
        "category": exercise["category"],
        "equipment": exercise["equipment"],
        "target": exercise["target"],
    }

def build_dataset(exercises):
    chunks = []
    metadatas = []
    for exercise in exercises:
        chunks.append(build_chunk(exercise))
        metadatas.append(build_metadata(exercise))
    return chunks, metadatas
