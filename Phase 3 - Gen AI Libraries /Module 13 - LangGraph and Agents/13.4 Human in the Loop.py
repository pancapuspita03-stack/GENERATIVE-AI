# 13.4 Human-in-the-Loop with Checkpoints (Halaman 150-152)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator

class ReviewState(TypedDict):
    plan: str
    approved: bool
    result: str
    messages: Annotated[list[str], operator.add]

def plan_node(state: ReviewState) -> dict:
    plan = "Step 1: Query vector DB. Step 2: Summarise results. Step 3: Format response."
    return {"plan": plan, "messages": [f"Plan created: {plan}"]}

def execute_node(state: ReviewState) -> dict:
    if not state["approved"]:
        return {"result": "Execution skipped - plan not approved.", "messages": ["Skipped."]}
    return {"result": "Pipeline executed successfully.", "messages": ["Executed."]}

checkpointer = MemorySaver()
graph = StateGraph(ReviewState)
graph.add_node("plan", plan_node)
graph.add_node("execute", execute_node)
graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", END)

# interrupt_before pauses the graph before the named node
app = graph.compile(checkpointer=checkpointer, interrupt_before=["execute"])

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "review-session-1"}}
    # Run until interrupt
    state = app.invoke(
        {"plan": "", "approved": False, "result": "", "messages": []},
        config=config
    )
    print("Plan:", state["plan"])
    print("Waiting for human approval...")

    # Human approves and resumes
    state["approved"] = True
    final = app.invoke(state, config=config)
    print("Result:", final["result"])
