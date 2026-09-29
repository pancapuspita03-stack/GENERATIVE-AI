import os
from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# Load API Key
# ============================================================

load_dotenv()


client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)



# ============================================================
# Few Shot System Prompt
# ============================================================

FEW_SHOT_SYSTEM = """
You are a data extractor.

Given a raw AI benchmark result string,
extract:
- model name
- task
- score

Return ONLY the JSON object.
No explanation.


Examples:

Input:
"GPT-4o scored 87.3% on the MMLU science subset"

Output:
{
  "model": "gpt-4o",
  "task": "MMLU science",
  "score": 87.3
}


Input:
"Claude Sonnet 4.5 achieved 92.1 on HumanEval"

Output:
{
  "model": "claude-sonnet-4-5",
  "task": "HumanEval",
  "score": 92.1
}


Input:
"Gemini 1.5 Pro: 78.9% accuracy on GSM8K math"

Output:
{
  "model": "gemini-1.5-pro",
  "task": "GSM8K math",
  "score": 78.9
}
"""



# ============================================================
# Test Data
# ============================================================

test_inputs = [

    "GPT-4o-mini reached 82.0% on MMLU",

    "Llama 3.1 70B: 86.4 on TruthfulQA",

    "Claude Opus 4.5 scored 96.7% on SWE-bench Verified",

]



# ============================================================
# Run Few Shot Prompting
# ============================================================

for text in test_inputs:


    try:

        response = client.chat.completions.create(

            # OpenRouter memilih model free yang tersedia
            model="openrouter/free",


            max_tokens=128,


            temperature=0,


            messages=[

                {
                    "role": "system",
                    "content": FEW_SHOT_SYSTEM
                },

                {
                    "role": "user",
                    "content": text
                }

            ]
        )


        print("=" * 50)

        print("Input:")
        print(text)


        print("\nOutput:")

        print(
            response
            .choices[0]
            .message
            .content
        )


    except Exception as error:

        print("=" * 50)

        print("ERROR:")
        print(error)