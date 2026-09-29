import os
from dotenv import load_dotenv
load_dotenv()
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
print("API Keys loaded successfully!")