import ollama

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

        response = ollama.chat(
            model="qwen3:8b",
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                *history
            ]
        )

        assistant_text = response["message"]["content"]

        history.append({
            "role": "assistant",
            "content": assistant_text
        })

        print(f"Qwen: {assistant_text}\n")


chat(system="You are a helpful Python tutor.")