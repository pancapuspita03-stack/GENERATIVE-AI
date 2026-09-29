import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Menggunakan OpenRouter (format OpenAI) karena saldo Anthropic habis
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# 1. Define the tool schema (OpenAI format)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window size and cost per 1K tokens for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The model identifier, e.g. 'gpt-4o' or 'claude-sonnet-4-5'."
                    }
                },
                "required": ["model_name"]
            }
        }
    }
]

# 2. The actual function the tool will call
def get_model_info(model_name: str) -> dict:
    db = {
        "claude-sonnet-4-5": {
            "context_k": 200,
            "cost_input": 3.00,
            "cost_output": 15.00
        },
        "gpt-4o": {
            "context_k": 128,
            "cost_input": 2.50,
            "cost_output": 10.00
        },
        "gemini-1.5-pro": {
            "context_k": 1000,
            "cost_input": 1.25,
            "cost_output": 5.00
        },
    }

    return db.get(
        model_name,
        {"error": f"Unknown model: {model_name}"}
    )

messages=[
    {
        "role": "user",
        "content": "How large is the context window of claude-sonnet-4-5?"
    }
]

# 3. First API call
response = client.chat.completions.create(
    model="openrouter/free",
    tools=tools,
    messages=messages
)

# 4. Check if model wants to use a tool
if response.choices[0].finish_reason == "tool_calls":
    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_input = json.loads(tool_call.function.arguments)

    # 5. Execute the function
    result = get_model_info(**tool_input)

    print(f"Tool called: {tool_name}({tool_input})")
    print(f"Tool result: {result}")

    # 6. Send tool result back to the model
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

    final = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
        tools=tools
    )

    print("\nFinal answer:")
    print(final.choices[0].message.content)
else:
    print("\nFinal answer:")
    print(response.choices[0].message.content)