def retrieve(query, collection, model, k=10, where=None):
    query_embedding = model.encode(query)
    query_kwargs = {
        "query_embeddings": query_embedding.tolist(),
        "n_results": k,
    }
    if where:
        query_kwargs["where"] = where
    results = collection.query(**query_kwargs)
    return results
