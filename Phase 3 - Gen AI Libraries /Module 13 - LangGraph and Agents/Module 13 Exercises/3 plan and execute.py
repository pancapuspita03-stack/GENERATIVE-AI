"""Exercise 3: Plan-and-Execute graph with a tracked current step index."""

from __future__ import annotations

import os
import re
import json
import math
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


class PlanState(TypedDict):
    request: str
    steps: list[str]
    current_step: int
    results: list[str]
    final_answer: str


@tool
def calculator(expression: str) -> str:
    """Evaluate a safe mathematical expression."""
    allowed = {name: value for name, value in math.__dict__.items() if not name.startswith("_")}
    allowed.update({"abs": abs, "round": round})
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except (ArithmeticError, SyntaxError, TypeError):
        return "Error: invalid expression."


@tool
def token_cost_lookup(model: str) -> str:
    """Look up input and output cost per one million tokens."""
    costs = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }
    if model not in costs:
        return f"Unknown model: {model}."
    return json.dumps({"model": model, **costs[model]})


TOOLS = {calculator.name: calculator, token_cost_lookup.name: token_cost_lookup}


def build_llm() -> ChatOpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY belum disetel di .env atau environment.")
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="nvidia/nemotron-3.5-lightning:free",
        api_key=api_key,
    )


def plan_and_execute():
    llm = build_llm()
    executor_llm = llm.bind_tools(list(TOOLS.values()))

    def planner(state: PlanState) -> dict[str, object]:
        response = llm.invoke(
            "Create a numbered plan with no more than 5 steps for:\n" + state["request"]
        ).content
        steps = [
            re.sub(r"^\s*\d+[\).\s-]*", "", line).strip()
            for line in str(response).splitlines()
            if re.match(r"^\s*\d+[\).\s-]+", line)
        ]
        if not steps:
            raise ValueError("Planner tidak menghasilkan numbered steps.")
        return {"steps": steps, "current_step": 0, "results": []}

    def executor(state: PlanState) -> dict[str, object]:
        step = state["steps"][state["current_step"]]
        response = executor_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Execute exactly this plan step. Use an available tool when "
                        "the step requires a calculation or model cost lookup. "
                        "Otherwise report the result directly."
                    )
                ),
                HumanMessage(content=step),
            ]
        )
        tool_results = []
        for tool_call in response.tool_calls:
            tool = TOOLS.get(tool_call["name"])
            if tool is None:
                raise ValueError(f"Unknown tool requested: {tool_call['name']}")
            tool_results.append(
                ToolMessage(
                    content=tool.invoke(tool_call["args"]),
                    tool_call_id=tool_call["id"],
                )
            )
        result = (
            "\n".join(message.content for message in tool_results)
            if tool_results
            else str(response.content)
        )
        return {
            "results": state["results"] + [result],
            "current_step": state["current_step"] + 1,
        }

    def route(state: PlanState) -> str:
        return "execute" if state["current_step"] < len(state["steps"]) else "finish"

    def finish(state: PlanState) -> dict[str, str]:
        return {"final_answer": "\n".join(state["results"])}

    graph = StateGraph(PlanState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("finish", finish)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route, {"execute": "executor", "finish": "finish"})
    graph.add_conditional_edges("executor", route, {"execute": "executor", "finish": "finish"})
    graph.add_edge("finish", END)
    return graph.compile()


if __name__ == "__main__":
    result = plan_and_execute().invoke(
        {"request": "Research the benefits of vector databases.", "steps": [], "current_step": 0, "results": [], "final_answer": ""}
    )
    print(result["final_answer"])
