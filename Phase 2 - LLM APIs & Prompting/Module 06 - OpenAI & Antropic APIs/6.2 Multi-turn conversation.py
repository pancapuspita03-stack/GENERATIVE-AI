import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Menggunakan OpenRouter sebagai jembatan
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

def chat(system: str) -> None:
    history = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            break

        history.append({
            "role": "user",
            "content": user_input
        })

        # Memanggil Claude lewat OpenRouter
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "system", "content": system},
                *history
            ]
        )

        assistant_text = response.choices[0].message.content

        history.append({
            "role": "assistant",
            "content": assistant_text
        })

        print(f"Claude: {assistant_text}\n")

chat(system="You are a helpful Python tutor.")