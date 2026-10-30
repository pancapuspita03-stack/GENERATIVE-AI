# 13.3 ReAct Agent - Reason + Act
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from typing import TypedDict, Annotated
import operator, json, os, math
from dotenv import load_dotenv

load_dotenv()

# Tools
def calculator(expression: str) -> str:
    """Evaluate a safe mathematical expression."""
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def token_cost_lookup(model: str) -> str:
    """Return cost per 1M tokens for a given model."""
    costs = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    }
    if model not in costs:
        return f"Unknown model: {model}. Known: {list(costs.keys())}"
    c = costs[model]
    return json.dumps({"model": model, "input_per_1M": c["input"], "output_per_1M": c["output"]})

TOOLS = {
    "calculator": calculator,
    "token_cost_lookup": token_cost_lookup,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Use Python math syntax.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression to evaluate"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "token_cost_lookup",
            "description": "Look up the cost per 1M tokens for a given LLM model.",
            "parameters": {
                "type": "object",
                "properties": {"model": {"type": "string", "description": "Model identifier"}},
                "required": ["model"],
            },
        },
    },
]

class ReActState(TypedDict):
    messages: Annotated[list, operator.add]

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise RuntimeError("OPENROUTER_API_KEY belum disetel.")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=openrouter_api_key,
).bind_tools(TOOL_SCHEMAS)

SYSTEM = SystemMessage(content="""You are a helpful AI assistant with access to two tools:
- calculator: for math calculations
- token_cost_lookup: for LLM pricing information
Always use tools when calculations or pricing lookups are needed. Think step by step.""")

def agent_node(state: ReActState) -> dict:
    response = llm.invoke([SYSTEM] + state["messages"])
    return {"messages": [response]}

def tool_node(state: ReActState) -> dict:
    last_msg = state["messages"][-1]
    results = []
    for tool_call in last_msg.tool_calls:
        fn = TOOLS[tool_call["name"]]
        output = fn(**tool_call["args"])
        results.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return {"messages": results}

def should_use_tools(state: ReActState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

graph = StateGraph(ReActState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_use_tools, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
agent = graph.compile()

if __name__ == "__main__":
    question = """I'm running a pipeline that processes 10,000 requests per day.
Each request uses 800 input tokens and 300 output tokens with gpt-4o.
What is my daily cost in USD?"""
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            print("Final answer:\n", msg.content)
