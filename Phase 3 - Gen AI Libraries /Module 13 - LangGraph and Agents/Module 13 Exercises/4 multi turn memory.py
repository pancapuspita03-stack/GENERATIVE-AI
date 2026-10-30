"""Exercise 4: preserve multi-turn context with MemorySaver and a fixed thread."""

from __future__ import annotations

import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


THREAD_ID = "multi-turn-demo-thread"


class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def build_agent():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY belum disetel di .env atau environment.")
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="nvidia/nemotron-3.5-lightning:free",
        api_key=api_key,
    )

    def chat_node(state: ConversationState) -> dict[str, list[BaseMessage]]:
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(ConversationState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=MemorySaver())


def demonstrate_memory():
    agent = build_agent()
    config = {"configurable": {"thread_id": THREAD_ID}}
    first = agent.invoke(
        {"messages": [HumanMessage(content="My favorite programming language is Python.")]},
        config=config,
    )
    second = agent.invoke(
        {"messages": [HumanMessage(content="What is my favorite programming language?")]},
        config=config,
    )
    return first["messages"][-1].content, second["messages"][-1].content


if __name__ == "__main__":
    first, second = demonstrate_memory()
    print("First:", first)
    print("Second:", second)
