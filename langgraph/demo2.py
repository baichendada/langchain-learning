from IPython.core.display import Image
from IPython.core.display_functions import display
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel


class State(BaseModel):
    x: int

load_dotenv(override=True)

def addition(state: State) -> State:
    print(f"add_state: {state}")
    return State(x=state.x + 1)

def subtraction(state: State) -> State:
    print(f"sub_state: {state}")
    return State(x=state.x - 2)

builder = StateGraph(state_schema=State)

print(builder)
print(builder.schemas)

builder.add_node("add", addition)
builder.add_node("sub", subtraction)

builder.add_edge(START, "add")
builder.add_edge("add", "sub")
builder.add_edge("sub", END)

print(builder.edges)
print(builder.nodes)

graph = builder.compile()

# display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

response = graph.invoke(State(x=10))
print(response)