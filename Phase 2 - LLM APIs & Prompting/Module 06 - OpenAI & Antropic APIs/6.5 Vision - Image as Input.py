import os
import base64
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Menggunakan OpenRouter
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# Option A: URL (fastest)
def describe_image_url(url: str) -> str:
    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",  # Ganti dengan model gratis yang mendukung vision di OpenRouter jika perlu
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe what you see in this image in detail."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url
                    }
                }
            ]
        }]
    )

    return response.choices[0].message.content


# Option B: base64 (for local files)
def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.standard_b64encode(data).decode()
    ext = Path(path).suffix.lstrip(".").lower()
    media_type = f"image/{ext}"  # image/png, image/jpeg, image/webp, image/gif

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{b64}"
                    }
                }
            ]
        }]
    )

    return response.choices[0].message.content


# Usage:
if __name__ == "__main__":
    print("Sedang menganalisis gambar dari Wikipedia...")
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Sunrise_over_the_sea.jpg/1280px-Sunrise_over_the_sea.jpg"
    text = describe_image_url(image_url)
    print(text)