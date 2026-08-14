from .chunking import load_exercises, build_dataset
from .indexing import create_embeddings, get_collection
from .retrieval import retrieve
import numpy as np
from sentence_transformers import SentenceTransformer
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "exercises.json")
EMBEDDINGS_PATH = os.path.join(SCRIPT_DIR, "..", "data", "embeddings")

def main():
    exercises = load_exercises(DATA_PATH)
    chunks, metadatas = build_dataset(exercises)
    chunks, metadatas, embeddings = create_embeddings(chunks, metadatas, EMBEDDINGS_PATH)
    collection = get_collection(chunks, metadatas, embeddings)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    results = retrieve("chest exercise with dumbells", collection, model)
    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        print(f"[{meta['name']}] (distance: {dist:.4f})")
        print(doc[:150])
        print("---")

if __name__ == "__main__":
    main()
