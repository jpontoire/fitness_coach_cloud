import os
from .program import extract_program_params, retrieve_for_program, generate_program
from rag.chunking import load_exercises, build_dataset
from rag.indexing import create_embeddings, get_collection
from rag.retrieval import retrieve
from finetuning.train import load_model, SYSTEM_PROMPT, ADAPTER_PATH
from peft import PeftModel
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "exercises.json")
EMBEDDINGS_PATH = os.path.join(SCRIPT_DIR, "..", "data", "embeddings")

REFUSAL_ANSWER = "I'm a fitness assistant focused on exercises, training, and nutrition. I can't help with that, but feel free to ask me anything about workouts or muscle building!"

def load_resources():
    exercises = load_exercises(DATA_PATH)
    chunks, metadatas = build_dataset(exercises)
    chunks, metadatas, embeddings = create_embeddings(chunks, metadatas, EMBEDDINGS_PATH)
    collection = get_collection(chunks, metadatas, embeddings)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    model, tokenizer = load_model()
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "is_gradient_checkpointing") and model.is_gradient_checkpointing:
        model.gradient_checkpointing_disable()

    return {
        "collection": collection,
        "embedding_model": embedding_model,
        "model": model,
        "tokenizer": tokenizer,
    }

def generate_response(model, tokenizer, question, max_new_tokens=400):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def rag_node(state, resources):
    where = None
    if state.get("equipment"):
        where = {"equipment": {"$in": state["equipment"]}}

    results = retrieve(state["question"], resources["collection"], resources["embedding_model"], k=5, where=where)
    context = "\n\n---\n\n".join(
        doc for doc in results["documents"][0]
    )
    return {"context": context}

def generate_node(state, resources):
    if state.get("context"):
        question = f"Based on this exercise data, answer the question: {state['question']}\n\nContext: {state['context']}"
    else:
        question = state["question"]

    answer = generate_response(resources["model"], resources["tokenizer"], question)
    return {"answer": answer}

def refusal_node(state, resources):
    return {"answer": REFUSAL_ANSWER}

def program_node(state, resources):
    params = extract_program_params(state["question"])

    equipment = state.get("equipment") or params.equipment

    candidates = retrieve_for_program(
        params.muscles, equipment,
        resources["collection"], resources["embedding_model"]
    )

    program = generate_program(candidates, params.preferences)

    if program is None:
        return {"answer": "Sorry, I couldn't generate a valid workout program. Could you try rephrasing your request?"}

    answer_lines = []
    for ex in program.exercises:
        answer_lines.append(f"- {ex.name}: {ex.sets} sets x {ex.reps} reps ({ex.target_muscle})")
    if program.notes:
        answer_lines.append(f"\nNotes: {program.notes}")

    return {"answer": "\n".join(answer_lines)}
