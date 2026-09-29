import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
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