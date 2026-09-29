import os
import json
import re
from dataclasses import dataclass, asdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL_NAME = "openrouter/free"

# ============================================================
# EXERCISE 1
# Three versions of a Code Review Assistant
# ============================================================

BASIC_SYSTEM = """You are a code review assistant.
Review the provided code and identify bugs or problems.
Give a short explanation and suggest improvements."""

INTERMEDIATE_SYSTEM = """You are an experienced code reviewer.
Review the provided code for:
- Bugs
- Security issues
- Performance problems
- Readability issues

For each issue, explain the problem and provide a suggested fix.
Be concise and practical."""

EXPERT_SYSTEM = """You are a senior Python engineer reviewing
production code for an AI software system.

Perform a comprehensive code review covering:
1. Correctness and potential bugs
2. Security vulnerabilities
3. Performance and scalability
4. Error handling
5. Maintainability and readability
6. Python best practices

For every issue, use this format:
Issue:
Impact:
Recommendation:
Corrected code:

Do not give unnecessary compliments.
If no issue exists in a category, state that briefly.
Prioritize issues by severity: Critical, High, Medium, Low."""

code_snippets = [
    """
def divide(a, b):
    return a / b
""",
    """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return database.execute(query)
""",
    """
def calculate_average(numbers):
    total = 0
    for number in numbers:
        total += number
    return total / len(numbers)
""",
    """
def load_file(path):
    file = open(path, "r")
    data = file.read()
    return data
""",
    """
def find_user(users, name):
    for user in users:
        if user["name"] == name:
            return user
    return None
"""
]

def review_code(system_prompt: str, code: str) -> str:
    """Send code to the model for review."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Review this code:\n\n{code}"}
            ],
        )
        if not hasattr(response, 'choices') or not response.choices:
            return "ERROR: OpenRouter returned empty response (rate limit or down)."
        return response.choices[0].message.content or "ERROR: Empty content."
    except Exception as e:
        return f"ERROR: {str(e)}"

def evaluate_review_prompts():
    """Evaluate Basic, Intermediate, and Expert prompts."""
    prompts = {
        "Basic": BASIC_SYSTEM,
        "Intermediate": INTERMEDIATE_SYSTEM,
        "Expert": EXPERT_SYSTEM,
    }

    for level, prompt in prompts.items():
        print("\n" + "=" * 70)
        print(f"{level.upper()} CODE REVIEW")
        print("=" * 70)

        for index, code in enumerate(code_snippets, start=1):
            print(f"\n--- Code Snippet {index} ---")
            print(code.strip())
            result = review_code(prompt, code)
            print("\nReview:")
            print(result)


# ============================================================
# EXERCISE 2
# PromptLibrary
# ============================================================

@dataclass
class PromptTemplate:
    """A reusable, versioned prompt template."""
    name: str
    system: str
    user: str
    version: str = "1.0"

class PromptLibrary:
    """Store named PromptTemplate instances."""

    def __init__(self):
        self.templates: dict[str, PromptTemplate] = {}
        self.last_evaluation: dict[str, str] = {}

    def add(self, template: PromptTemplate):
        """Add or replace a prompt template."""
        self.templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        """Get a prompt template by name."""
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found.")
        return self.templates[name]

    def save(self, path: str):
        """Save the prompt library to JSON."""
        data = {
            "templates": {
                name: asdict(template)
                for name, template in self.templates.items()
            },
            "last_evaluation": self.last_evaluation,
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load the prompt library from JSON."""
        library = cls()
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            
        for name, template_data in data["templates"].items():
            library.templates[name] = PromptTemplate(**template_data)
            
        library.last_evaluation = data.get("last_evaluation", {})
        return library

    def record_evaluation(self, template_name: str, version: str):
        """Record the version used for the last evaluation."""
        self.last_evaluation[template_name] = version

def run_prompt_library_demo():
    print("\n" + "=" * 70)
    print("EXERCISE 2 - PROMPT LIBRARY")
    print("=" * 70)

    library = PromptLibrary()
    
    library.add(PromptTemplate(
        name="code_review",
        version="1.0",
        system=BASIC_SYSTEM,
        user="Review the following code:\n\n{code}"
    ))
    library.add(PromptTemplate(
        name="code_review_intermediate",
        version="1.1",
        system=INTERMEDIATE_SYSTEM,
        user="Review the following code:\n\n{code}"
    ))
    library.add(PromptTemplate(
        name="code_review_expert",
        version="2.0",
        system=EXPERT_SYSTEM,
        user="Review the following code:\n\n{code}"
    ))

    library.record_evaluation("code_review", "1.0")
    library.record_evaluation("code_review_intermediate", "1.1")
    library.record_evaluation("code_review_expert", "2.0")

    library.save("prompt_library.json")
    print("\nPrompt library saved to: prompt_library.json")
    
    print("\nLast evaluation versions:")
    for name, version in library.last_evaluation.items():
        print(f"- {name}: version {version}")

    loaded_library = PromptLibrary.load("prompt_library.json")
    print("\nLoaded templates:")
    for name, template in loaded_library.templates.items():
        print(f"- {name} v{template.version}")


# ============================================================
# EXERCISE 3
# Automatic JSON Repair
# ============================================================

def safe_json_parse(text: str) -> dict:
    """
    Parse JSON safely.
    Step 1: Try json.loads()
    Step 2: Remove markdown fences and try again.
    Step 3: Ask the model to repair the JSON.
    """
    # Step 1 - Direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 2 - Strip markdown fences
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 3 - Ask model to repair JSON
    repair_prompt = f"""Repair the following invalid JSON.
Return ONLY valid JSON.
Do not use markdown.
Do not provide explanations.

Invalid JSON:
{cleaned}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=[
                {"role": "user", "content": repair_prompt}
            ],
        )
        if not hasattr(response, 'choices') or not response.choices:
            return {"error": "API returned empty choices."}
            
        repaired = response.choices[0].message.content or ""
        repaired = repaired.strip()
        
        repaired = re.sub(r"^```json\s*", "", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"^```\s*", "", repaired)
        repaired = re.sub(r"\s*```$", "", repaired)
        repaired = repaired.strip()

        return json.loads(repaired)
    except Exception as e:
        return {"error": f"Failed to repair JSON: {str(e)}"}


def run_json_repair_demo():
    print("\n" + "=" * 70)
    print("EXERCISE 3 - AUTOMATIC JSON REPAIR")
    print("=" * 70)

    invalid_json = (
        "```json\n"
        "{\n"
        '    "name": "Model",\n'
        '    "version": "1.0",\n'
        '    "provider": "OpenAI",\n'
        "}\n" # Trailing comma causing error
        "```"
    )

    print("Attempting to parse invalid JSON:")
    print(invalid_json)
    
    result = safe_json_parse(invalid_json)
    
    print("\nResult after safe_json_parse:")
    print(json.dumps(result, indent=2))


# ============================================================
# EXERCISE 4
# Chain-of-Thought (CoT) Prompt for Ranking Models
# ============================================================

def evaluate_models_cot(scores_text: str) -> str:
    """Send scores to the model to rank using CoT."""
    
    system_prompt = """You are an AI evaluation expert.
You will be given a list of LLM evaluation scores across 3 tasks.
Perform the following steps:
1. Think step-by-step to calculate the average score for each model.
2. Rank the models from highest average to lowest.
3. Write a final 2-sentence recommendation on which model to choose.

Provide your output in the following format:
THINKING:
(Your step-by-step calculations)

RANKING:
1. [Model Name] (Average: [Score])
...

RECOMMENDATION:
(2 sentences)"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here are the scores:\n\n{scores_text}"}
            ],
        )
        if not hasattr(response, 'choices') or not response.choices:
            return "ERROR: OpenRouter returned empty response."
        return response.choices[0].message.content or "ERROR: Empty content."
    except Exception as e:
        return f"ERROR: {str(e)}"

def run_cot_demo():
    print("\n" + "=" * 70)
    print("EXERCISE 4 - CHAIN OF THOUGHT EVALUATION")
    print("=" * 70)
    
    test_inputs = [
        """
        Model A: Task1: 80, Task2: 90, Task3: 85
        Model B: Task1: 85, Task2: 85, Task3: 85
        Model C: Task1: 90, Task2: 95, Task3: 92
        """,
        """
        GPT-3.5: Task1: 70, Task2: 75, Task3: 80
        GPT-4: Task1: 95, Task2: 92, Task3: 98
        Claude-3: Task1: 94, Task2: 90, Task3: 96
        """,
        """
        Model X: Task1: 50, Task2: 60, Task3: 55
        Model Y: Task1: 55, Task2: 55, Task3: 65
        Model Z: Task1: 60, Task2: 50, Task3: 60
        """
    ]
    
    for i, scores in enumerate(test_inputs, start=1):
        print(f"\n--- Test Input {i} ---")
        print(scores.strip())
        result = evaluate_models_cot(scores.strip())
        print("\nModel Output:")
        print(result)


if __name__ == "__main__":
    # Comment/uncomment the functions you want to test
    evaluate_review_prompts()
    # run_prompt_library_demo()
    # run_json_repair_demo()
    # run_cot_demo()
    run_cot_demo()