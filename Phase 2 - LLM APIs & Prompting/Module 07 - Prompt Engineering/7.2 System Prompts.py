import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# Poor system prompt - vague, no constraints
WEAK_SYSTEM = "You are an AI assistant."

# Strong system prompt - explicit role, rules, format
STRONG_SYSTEM = """You are a senior Python engineer reviewing
code for a production AI pipeline.
Your job:
- Identify bugs, security issues, and performance problems
- Suggest concrete improvements with code examples
- Explain WHY each issue matters
Rules:
- Be direct. Do not pad with compliments.
- If code is correct, say so briefly and move on.
- Always include the corrected code when suggesting a fix.
Format:
Return your review as a numbered list. Each item: Issue → Impact → Fix."""

messages = [
    {"role": "system", "content": STRONG_SYSTEM},
    {
        "role": "user",
        "content": """Review this function:
def get_user(user_id):
key = os.getenv('DB_KEY')
result = requests.get(f'http://db/{user_id}?key={key}')
return result.json()"""
    }
]

response = client.chat.completions.create(
    model="openrouter/free",
    max_tokens=1024,
    messages=messages,
)

print(response.choices[0].message.content)