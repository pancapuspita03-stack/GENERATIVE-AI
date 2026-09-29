from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str

@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str

class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        ...

class AnthropicClient(BaseLLMClient):
    # Nama class tetap 'AnthropicClient' agar sesuai dengan modul, 
    # tapi mesinnya diganti pakai OpenRouter (OpenAI SDK) agar gratis.
    def __init__(self, model: str = "openrouter/free"):
        self.model = model
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        )

    def chat(
        self, 
        messages, 
        system="", 
        max_tokens=1024, 
        temperature=0.7
    ) -> ChatResponse:
        
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})

        api_messages += [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )

        return ChatResponse(
            text=resp.choices[0].message.content,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            model=self.model,
        )

class OpenAIClient(BaseLLMClient):
    # Ini OpenAIClient asli, namun juga saya alihkan ke OpenRouter karena limit OpenAI Anda.
    def __init__(self, model: str = "openrouter/free"):
        self.model = model
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        )

    def chat(
        self, 
        messages, 
        system="", 
        max_tokens=1024, 
        temperature=0.7
    ) -> ChatResponse:
        
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})

        api_messages += [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )

        return ChatResponse(
            text=resp.choices[0].message.content,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            model=self.model,
        )

# Usage - Komentarnya sudah dibuka agar langsung dites:
print("Menguji AnthropicClient (via OpenRouter)...")
client: BaseLLMClient = AnthropicClient()
# client: BaseLLMClient = OpenAIClient()

msgs = [ChatMessage(role="user", content="What is a vector database?")]
result = client.chat(msgs, system="Be concise.")
print(result.text)
print(f"Cost estimate: {result.input_tokens} in, {result.output_tokens} out")
