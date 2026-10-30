from openai import OpenAI, RateLimitError, APIError
import os, json
from dotenv import load_dotenv

load_dotenv()

# Gunakan OpenRouter lagi (karena OpenAI API Anda tidak memiliki saldo)
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

try:
    response = client.chat.completions.create(
        model="openrouter/free",
        response_format={"type": "json_object"}, # enforces valid JSON
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