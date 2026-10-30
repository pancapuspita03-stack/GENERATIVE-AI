"""Exercise 2: ReAct graph that stops after eight tool calls."""

from __future__ import annotations

import ast
import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


MAX_TOOL_CALLS = 8


@tool
def calculator(expression: str) -> str:
    """Evaluate only simple arithmetic expressions."""
    allowed = set("0123456789+-*/(). ")
    if (
        not expression
        or len(expression) > 128
        or any(character not in allowed for character in expression)
    ):
        return "Error: unsupported expression."

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp):
            operations = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
            }
            operation = operations.get(type(node.op))
            if operation is not None:
                return operation(evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp):
            operations = {ast.UAdd: operator.pos, ast.USub: operator.neg}
            operation = operations.get(type(node.op))
            if operation is not None:
                return operation(evaluate(node.operand))
        raise ValueError("Only basic arithmetic is allowed.")

    try:
        return str(evaluate(ast.parse(expression, mode="eval")))
    except (ArithmeticError, SyntaxError, ValueError):
        return "Error: invalid expression."


class ReActState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    tool_calls: int
    final_answer: str


def build_graph():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    offline = os.environ.get("REACT_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or not api_key
    llm = None
    if not offline:
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model="nvidia/nemotron-3.5-lightning:free",
            api_key=api_key,
        ).bind_tools([calculator])

    def agent_node(state: ReActState) -> dict[str, object]:
        if offline:
            tool_messages = [
                message for message in state["messages"] if isinstance(message, ToolMessage)
            ]
            if tool_messages:
                response = AIMessage(
                    content=f"Offline calculation result: {tool_messages[-1].content}"
                )
            else:
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "2 + 2"},
                            "id": "offline_calculator",
                            "type": "tool_call",
                        }
                    ],
                )
        else:
            assert llm is not None
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            f"Use the calculator when needed. Stop after at most "
                            f"{MAX_TOOL_CALLS} calculator calls."
                        )
                    )
                ]
                + state["messages"]
            )
        return {"messages": [response]}

    def tool_node(state: ReActState) -> dict[str, object]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return {"messages": []}
        outputs = []
        for call in last.tool_calls:
            if call["name"] != "calculator":
                outputs.append(
                    ToolMessage(
                        content=f"Error: unknown tool {call['name']!r}.",
                        tool_call_id=call["id"],
                    )
                )
                continue
            outputs.append(
                ToolMessage(
                    content=calculator.invoke(call["args"]),
                    tool_call_id=call["id"],
                )
            )
        return {
            "messages": outputs,
            "tool_calls": state["tool_calls"] + len(outputs),
        }

    def route(state: ReActState) -> str:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return "done"
        if state["tool_calls"] + len(last.tool_calls) > MAX_TOOL_CALLS:
            return "limit"
        return "tools"

    def limit_node(_: ReActState) -> dict[str, str]:
        return {"final_answer": "Reached step limit."}

    graph = StateGraph(ReActState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("limit", limit_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route, {"tools": "tools", "done": END, "limit": "limit"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("limit", END)
    return graph.compile()


if __name__ == "__main__":
    load_dotenv()
    if os.environ.get("REACT_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or not os.environ.get("OPENROUTER_API_KEY"):
        print("Mode offline: menggunakan kalkulator lokal tanpa API key.")
    state = build_graph().invoke(
        {"messages": [HumanMessage(content="What is 2 + 2?")], "tool_calls": 0, "final_answer": ""}
    )
    print(state.get("final_answer") or state["messages"][-1].content)
