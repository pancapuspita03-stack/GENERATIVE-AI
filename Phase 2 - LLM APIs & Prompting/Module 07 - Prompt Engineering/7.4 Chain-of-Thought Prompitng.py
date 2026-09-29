import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)



# Without CoT
DIRECT_PROMPT = """
If a model costs $3.00 per million input tokens and 
$15.00 per million output tokens,

and a request uses 2,400 input tokens and 
800 output tokens,

what is the total cost in USD?
"""



# With CoT
COT_PROMPT = """
If a model costs $3.00 per million input tokens and 
$15.00 per million output tokens,

and a request uses 2,400 input tokens and 
800 output tokens,

what is the total cost in USD?

Think through this step by step before giving the final answer.
"""



# Zero-shot CoT
ZERO_SHOT_COT = """
Solve this problem.

Think step by step, showing each calculation.

Finally, state:
ANSWER: $X.XXXXXX


Problem:

A pipeline makes 50 API calls per hour.

Each call uses an average of:
1,200 input tokens
400 output tokens

The model costs:
$3.00/M input tokens
$15.00/M output tokens

What is the daily cost?
"""



for label, prompt in [

    ("Direct", DIRECT_PROMPT),

    ("CoT", COT_PROMPT),

    ("Zero-shot CoT", ZERO_SHOT_COT)

]:


    resp = client.chat.completions.create(

        model="openrouter/free",

        max_tokens=512,

        temperature=0,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )


    print("=" * 50)

    print(f"=== {label} ===")

    print(
        resp
        .choices[0]
        .message
        .content
    )

    print()