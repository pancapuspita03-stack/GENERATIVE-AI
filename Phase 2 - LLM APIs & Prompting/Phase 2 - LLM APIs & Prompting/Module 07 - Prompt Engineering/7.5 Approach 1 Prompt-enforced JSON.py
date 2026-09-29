import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

SYSTEM = """You are a data extractor. Extract information and
return ONLY a JSON object.
No markdown, no explanation, no code fences. Raw JSON only.
Schema:
{
"company": string,
"founded": integer or null,
"products": [string],
"headquarters": string or null,
"is_public": boolean
}"""

texts = [
    "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude AI models and is headquartered in San Francisco. It is a private company.",
    "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San Francisco, it remains private despite a major Microsoft investment.",
]

def extract_company_info(text: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model="openrouter/free",
            max_tokens=512,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text}
            ],
        )

        # Cek jika OpenRouter merespons dengan format error yang tidak standar (NoneType choices)
        if not hasattr(resp, 'choices') or not resp.choices:
            return {"error": "OpenRouter API is down or rate-limited."}

        content = resp.choices[0].message.content
        if content is None:
            return {"error": "API returned an empty response (None)"}

        raw = content.strip()

        # Strip any accidental markdown fences
        raw = (
            raw.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        
        return json.loads(raw)
        
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw": raw}
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}


for text in texts:
    info = extract_company_info(text)
    print(json.dumps(info, indent=2))
    print()