import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Menggunakan OpenRouter karena saldo Anthropic habis
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

print("Streaming respons...\n")

# Memanggil API dengan mode stream=True
response = client.chat.completions.create(
    model="openrouter/free",
    max_tokens=512,
    messages=[{"role": "user", "content": "List 5 use cases for vector databases."}],
    stream=True,
    stream_options={"include_usage": True} # Agar kita bisa mendapatkan info token di akhir
)

input_tokens = 0
output_tokens = 0

# Menampilkan teks sedikit demi sedikit (streaming) saat AI mengetik
for chunk in response:
    if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
        
    # Mengambil informasi jumlah token (biasanya dikirim di data chunk terakhir)
    if hasattr(chunk, 'usage') and chunk.usage:
        input_tokens = chunk.usage.prompt_tokens
        output_tokens = chunk.usage.completion_tokens

print() # Baris baru setelah teks selesai

# Menampilkan total token
print(f"\nTotal tokens: {input_tokens + output_tokens}")