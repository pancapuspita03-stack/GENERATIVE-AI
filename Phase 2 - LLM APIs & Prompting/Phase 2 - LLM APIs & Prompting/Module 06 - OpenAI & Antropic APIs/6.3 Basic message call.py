from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Dialihkan ke OpenRouter karena limit OpenAI habis
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

response = client.chat.completions.create(
    model="openrouter/free",
    max_tokens=1024,
    messages=[
        {
            "role": "system",
            "content": "You are a concise technical assistant."
        },
        {
            "role": "user",
            "content": "What is the difference between RAG and fine-tuning?"
        }
    ]
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")