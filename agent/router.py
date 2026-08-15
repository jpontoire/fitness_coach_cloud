from .llm import get_client_and_model

def router_node(state):
    client, model = get_client_and_model(state.get("provider"), state.get("api_key"))

    prompt = f"""Determine if this question is about a specific video game quest and requires looking up detailed quest information, or if it's a general question that can be answered directly.
Question: {state['question']}
Answer with only one word: "RAG" or "DIRECT"."""
    # (garde ton vrai prompt existant, celui-ci est un exemple générique)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = response.choices[0].message.content.strip()

    needs_rag = "RAG" in response_text.upper()
    return {"needs_rag": needs_rag}
