import os
from dotenv import find_dotenv, load_dotenv

# Otomatis mencari file .env di folder mana pun di dalam proyek
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


def get_api_key(provider: str) -> str:
    """Retrieve an API key from the environment."""
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    env_var = key_map.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")

    key = os.getenv(env_var)
    if not key:
        raise EnvironmentError(
            f"{env_var} is not set. Add it to your .env file."
        )
    return key


# --- Testing ---
if __name__ == "__main__":
    print(f"[DEBUG] Path .env ditemukan di: {dotenv_path}")

    try:
        anthropic_key = get_api_key("anthropic")
        print(f"Anthropic Key terdeteksi: {anthropic_key[:6]}...")
    except EnvironmentError as e:
        print(f"[Peringatan] {e}")

    try:
        get_api_key("unknown_provider")
    except ValueError as e:
        print(f"[Error] {e}")