# 13.2 Core Concepts (Halaman 144-145)
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

# State is a TypedDict - shared across all nodes
class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]
    step_count: int
    final_answer: str

# Nodes are plain Python functions
def think_node(state: AgentState) -> dict:
    thought = f"Step {state['step_count'] + 1}: analysing the problem..."
    return {
        "messages": [thought],
        "step_count": state["step_count"] + 1,
    }

def answer_node(state: AgentState) -> dict:
    return {"final_answer": "Based on my analysis, the answer is X."}

# Conditional edge - decides where to go next
def should_continue(state: AgentState) -> str:
    if state["step_count"] < 2:
        return "think"
    return "answer"

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("think", think_node)
graph.add_node("answer", answer_node)
graph.add_edge(START, "think")
graph.add_conditional_edges("think", should_continue, {"think": "think", "answer": "answer"})
graph.add_edge("answer", END)
app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"messages": [], "step_count": 0, "final_answer": ""})
    print("Messages:", result["messages"])
    print("Answer  :", result["final_answer"])
