def build_structured_example(exercise):
    context = f"""Exercise: {exercise['name']}
Category: {exercise['category']}
Target: {exercise['target']}
Secondary muscles: {', '.join(exercise['secondary_muscles'])}
Equipment: {exercise['equipment']}
Instructions: {exercise['instructions']['en']}"""

    question = f"Based on this exercise data, describe {exercise['name']}.\n\nContext: {context}"
    answer = context

    return {"question": question, "answer": answer}

def build_structured_dataset(exercises):
    return [build_structured_example(ex) for ex in exercises]
