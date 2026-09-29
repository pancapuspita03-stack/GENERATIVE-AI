import sys
from pathlib import Path

# Root folder "Generative AI"
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from my_ai_project.config import LLMConfig


config = LLMConfig(model="gpt-4")

print(f"Model: {config.model}")
print(f"Temperature: {config.temperature}")