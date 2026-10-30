# 9.2 Chat Models (Halaman 101-103)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Add it to your environment or .env file."
    )

# InclusionAI model via OpenRouter
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/nemotron-3.5-super-vl:free",
    temperature=0.7,
    max_tokens=1024,
    api_key=api_key,
)

messages = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LangChain Expression Language?"),
]

if __name__ == "__main__":
    # Invoke (blocking)
    print("=== Invoke OpenRouter model ===")
    response = llm.invoke(messages)
    print(response.content)

    # Stream
    print("\n=== Stream OpenRouter model ===")
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
    print()

    # Batch - run multiple inputs in parallel
    print("\n=== Batch OpenRouter model ===")
    responses = llm.batch([
        [HumanMessage(content="What is RAG?")],
        [HumanMessage(content="What is a vector database?")],
    ])
    for r in responses:
        print(r.content[:80])
