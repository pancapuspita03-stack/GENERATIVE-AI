import openai
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Exercise 1
# Retry on Rate Limit with Exponential Backoff
# ============================================================

def retry_on_rate_limit(
    client,
    messages,
    max_retries=5,
    system="",
    max_tokens=512
):
    """
    Mengulang panggilan API saat terjadi RateLimitError dengan jeda waktu eksponensial.
    """
    api_messages = []
    if system:
        api_messages.append({"role": "system", "content": system})
    api_messages += messages

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="google/gemini-2.5-flash",
                max_tokens=max_tokens,
                messages=api_messages,
            )
            return response

        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt
            print(f"Batas rate limit tercapai. Mencoba kembali dalam {wait_time} detik...")
            time.sleep(wait_time)

    raise RuntimeError("Batas maksimum percobaan ulang telah terlampaui.")


class MockRateLimitClient:
    def __init__(self, failures_before_success=3):
        self.failures_before_success = failures_before_success
        self.attempts = 0

    class Chat:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self.Completions(parent)

        class Completions:
            def __init__(self, parent):
                self.parent = parent

            def create(self, **kwargs):
                self.parent.attempts += 1
                if self.parent.attempts <= self.parent.failures_before_success:
                    import httpx
                    err_response = httpx.Response(429, request=httpx.Request("POST", "http://test"))
                    raise openai.RateLimitError(
                        message="Mock rate limit", 
                        response=err_response, 
                        body=None
                    )

                return MockResponse()

    @property
    def chat(self):
        return self.Chat(self)


class MockResponse:
    class Choice:
        class Message:
            content = "Respon mock setelah berhasil mencoba ulang."
        message = Message()
    
    choices = [Choice()]
    
    class Usage:
        prompt_tokens = 10
        completion_tokens = 20

    usage = Usage()


# ============================================================
# Exercise 2
# Token Budget Manager
# ============================================================
class BudgetExceeded(Exception):
    pass

class TokenBudgetManager:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def total_tokens(self):
        return self.total_input_tokens + self.total_output_tokens

    def add_usage(self, input_tokens: int, output_tokens: int):
        new_total = self.total_tokens + input_tokens + output_tokens
        if new_total > self.max_tokens:
            raise BudgetExceeded(f"Budget terlampaui. Batas: {self.max_tokens}, Diminta: {new_total}")

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def remaining_tokens(self):
        return self.max_tokens - self.total_tokens


# ============================================================
# Exercise 4
# Stream Response Directly to File (.txt)
# ============================================================
def stream_to_file(prompt: str, output_path: str):
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )

    print(f"Writing response to file '{output_path}' in real-time...\n")
    with open(output_path, "w", encoding="utf-8") as file:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
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


# ============================================================
# Main Execution Block (Menjalankan Exercise 1, 2, & 4)
# ============================================================
if __name__ == "__main__":
    # Test Exercise 1
    print("=== MENJALANKAN EXERCISE 1 ===")
    mock_client = MockRateLimitClient(failures_before_success=3)
    response = retry_on_rate_limit(mock_client, messages=[{"role": "user", "content": "Halo"}])
    print("Hasil Exercise 1:")
    print(response.choices[0].message.content)
    print(f"Token terpakai -> Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}")
    print("\n" + "="*50 + "\n")

    # Test Exercise 2
    print("=== MENJALANKAN EXERCISE 2 ===")
    budget = TokenBudgetManager(max_tokens=1000)
    input_used_1, output_used_1 = 200, 150
    budget.add_usage(input_used_1, output_used_1)
    print(f"Pemakaian 1 -> Input: {input_used_1} token, Output: {output_used_1} token")
    print(f"Total kumulatif terpakai: {budget.total_tokens} token")
    print(f"Sisa token dalam budget: {budget.remaining_tokens()} token")
    print("\n" + "="*50 + "\n")

    # Test Exercise 4
    print("=== MENJALANKAN EXERCISE 4 ===")
    stream_to_file("Explain retrieval-augmented generation.", "exercise_4_output.txt")