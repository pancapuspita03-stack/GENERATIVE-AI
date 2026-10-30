# 9.4 LCEL - LangChain Expression Language (Halaman 104-107)
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/nemotron-3.5-super-vl:free",
    api_key=os.environ.get("OPENROUTER_API_KEY", "your-openrouter-key")
)
parser = StrOutputParser()

# Basic chain: prompt | model | parser
prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise the following text in exactly {sentences} sentences."),
    ("human", "{text}"),
])
chain = prompt | llm | parser

# Branching with RunnableParallel
summarise_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise this in one sentence."),
    ("human", "{text}"),
])
keywords_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract 3 keywords from this text. Return them comma-separated."),
    ("human", "{text}"),
])

parallel_chain = RunnableParallel(
    summary=summarise_prompt | llm | parser,
    keywords=keywords_prompt | llm | parser,
)

# Conditional routing
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Python expert. Provide a complete, working code example."),
    ("human", "{input}"),
])
prose_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a clear technical writer. Explain conceptually."),
    ("human", "{input}"),
])

code_chain = code_prompt | llm | parser
prose_chain = prose_prompt | llm | parser

def route(input_dict: dict) -> str:
    """Route to code or prose chain based on the question."""
    q = input_dict["input"].lower()
    if any(w in q for w in ["write", "code", "implement", "example", "function"]):
        return "code"
    return "prose"

def run_routed(input_dict: dict) -> str:
    choice = route(input_dict)
    if choice == "code":
        return code_chain.invoke(input_dict)
    return prose_chain.invoke(input_dict)

router_chain = RunnableLambda(run_routed)

if __name__ == "__main__":
    sample_text = """LangChain is a framework for developing applications powered by large language models.
It provides abstractions for prompts, models, memory, and retrieval so developers can
compose complex AI workflows from simple building blocks without writing boilerplate."""
    print("=== Basic Chain ===")
    print(chain.invoke({"sentences": 2, "text": sample_text}))

    print("\n=== Parallel Chain ===")
    p_res = parallel_chain.invoke({"text": "Retrieval-Augmented Generation (RAG) improves LLM answers by fetching relevant documents at inference time."})
    print("Summary :", p_res["summary"])
    print("Keywords:", p_res["keywords"])

    print("\n=== Router Chain ===")
    print(router_chain.invoke({"input": "What is attention in transformers?"}))
