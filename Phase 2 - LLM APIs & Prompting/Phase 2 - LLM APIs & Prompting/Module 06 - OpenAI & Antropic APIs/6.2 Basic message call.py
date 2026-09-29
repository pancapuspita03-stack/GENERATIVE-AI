import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Dialihkan ke OpenRouter karena saldo Anthropic habis
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

message = client.chat.completions.create(
    model="openrouter/free",
    max_tokens=512,
    messages=[
        {"role": "system", "content": "You are a concise technical writer. Answer in plain English, no jargon."},
        {"role": "user", "content": "Explain what a vector database does."}
    ]
)

print(message.choices[0].message.content)

# Usage stats (OpenAI format)
print(f"Input tokens: {message.usage.prompt_tokens}")
print(f"Output tokens: {message.usage.completion_tokens}")