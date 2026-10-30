import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def stream_to_file(prompt: str, output_path: str):
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )

    print(f"Writing response to file '{output_path}' in real-time...\n")
    with open(output_path, "w", encoding="utf-8") as file:
        response = client.chat.completions.create(
            model="openrouter/free",
            max_tokens=512,
            stream=True,
            messages=[{"role": "user", "content": prompt}]
        )
        for chunk in response:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                file.write(text)
                file.flush()
                print(text, end="", flush=True)

    print(f"\n\nResponse saved to: {output_path}")

# Example Exercise 4
if __name__ == "__main__":
    stream_to_file("Explain retrieval-augmented generation.", "exercise_4_output.txt")
