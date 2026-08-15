from openai import OpenAI

PROVIDER_CONFIG = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}

def get_client_and_model(provider, api_key):
    provider = (provider or "groq").lower()
    config = PROVIDER_CONFIG.get(provider, PROVIDER_CONFIG["groq"])

    client = OpenAI(api_key=api_key, base_url=config["base_url"])
    return client, config["model"]
