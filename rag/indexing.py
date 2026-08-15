import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

def create_embeddings(chunks, metadatas, output_path):
    if not os.path.exists(output_path) or not os.path.exists(f"{output_path}/embeddings.npy"):
        os.makedirs(output_path, exist_ok=True)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(chunks, show_progress_bar=True)
        np.save(f"{output_path}/embeddings.npy", embeddings)
        with open(f"{output_path}/metadata.json", "w") as f:
            json.dump({"chunks": chunks, "metadatas": metadatas}, f)
        return chunks, metadatas, embeddings
    else:
        embeddings = np.load(f"{output_path}/embeddings.npy")
        with open(f"{output_path}/metadata.json", "r") as f:
            data = json.load(f)
        return data["chunks"], data["metadatas"], embeddings

def get_collection(chunks, metadatas, embeddings, db_path="./chromadb"):
    chroma_host = os.environ.get("CHROMA_HOST")

    if chroma_host:
        if chroma_host.startswith("https://"):
            hostname = chroma_host.replace("https://", "")
            client = chromadb.HttpClient(host=hostname, ssl=True, port=443)
        elif chroma_host.startswith("http://"):
            hostname = chroma_host.replace("http://", "")
            client = chromadb.HttpClient(host=hostname, ssl=False)
        else:
            chroma_port = int(os.environ.get("CHROMA_PORT", 8000))
            client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    else:
        client = chromadb.PersistentClient(path=db_path)

    collection = client.get_or_create_collection(name="exercises")

    if collection.count() != len(chunks):
        if collection.count() > 0:
            client.delete_collection(name="exercises")
            collection = client.create_collection(name="exercises")
        embeddings = np.asarray(embeddings)
        collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
    return collection
