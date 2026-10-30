import json
import os

from dotenv import load_dotenv
from openai import APIError, OpenAI, RateLimitError

load_dotenv()

MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY tidak ditemukan di .env")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

try:
    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system", 
                "content": """Extract entities. Return JSON with this schema:
{"people": [string], "organizations": [string], "locations": [string]}"""
            },
            {
                "role": "user", 
                "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla."
            }
        ]
    )

    # Cek jika OpenRouter merespons dengan format error yang tidak standar
    if not hasattr(response, 'choices') or not response.choices:
        print("Error: API OpenRouter sedang down atau kelebihan beban (rate-limited).")
    else:
        content = response.choices[0].message.content
        if content:
            result = json.loads(content)
            print(json.dumps(result, indent=2))
        else:
            print("Error: API returned an empty response (None)")

except RateLimitError as e:
    print("\n[PERINGATAN] Anda telah kehabisan limit harian gratis OpenRouter (atau OpenAI).")
    print("Silakan tunggu besok agar limit di-reset, atau gunakan layanan berbayar.")
    print("Pesan Asli:", str(e))
except APIError as e:
    print("\n[PERINGATAN] Terjadi kesalahan pada API penyedia.")
    print("Pesan Asli:", str(e))