from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel


class State(BaseModel):
    x: int

def increment(state: State):
    print(f"[increment] 当前 x = {state.x}")
    return State(x=state.x + 1)

def is_done(state: State) -> bool:
    print(f"[is_done] 当前 x = {state.x}")
    return state.x >= 3

builder = StateGraph(State)

builder.add_node("increment", increment)

builder.add_edge(START, "increment")
builder.add_conditional_edges("increment", is_done, {
    True: END,
    False: "increment"
})

graph = builder.compile()

print(graph.invoke(State(x=0)))