from typing import Optional

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel


class State(BaseModel):
    x: int
    result: Optional[str] = None

def check_x(state: State)-> State:
    print(f"[check_x] Received state: {state}")
    return state

def is_even(state: State) -> bool:
    return state.x % 2 == 0

def handle_even(state: State) -> State:
    print(f"[handle_even] Received state: {state}")
    return State(x=state.x, result="even")

def handle_odd(state: State) -> State:
    print(f"[handle_odd] Received state: {state}")
    return State(x=state.x, result="odd")

def print_result(state: State) -> State:
    print(f"[print_result] Received state: {state}")
    return state

builder = StateGraph(State)

builder.add_node("check_x", check_x)
builder.add_node("handle_even", handle_even)
builder.add_node("handle_odd", handle_odd)
builder.add_node("print_result", print_result)

builder.add_edge(START, "check_x")
builder.add_conditional_edges("check_x", is_even, {
    True: "handle_even",
    False: "handle_odd"
})
builder.add_edge("handle_even", "print_result")
builder.add_edge("handle_odd", "print_result")
builder.add_edge("print_result", END)

graph = builder.compile()

print("\n✅ 测试 x=4（偶数）")
graph.invoke(State(x=4))

print("\n✅ 测试 x=3（奇数）")
graph.invoke(State(x=3))