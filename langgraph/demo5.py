from typing import Optional

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel


class State(BaseModel):
    x: int

def check_x(state: State) -> State:
    print(f"[check_x] 当前 x = {state.x}")
    return state

def is_even(state: State) -> bool:
    return state.x % 2 == 0

def increment(state: State) -> State:
    print(f"[increment] x 是偶数，执行 +1 → {state.x + 1}")
    return State(x=state.x + 1)

def done_print(state: State)-> State:
    print(f"[done_print] 最终结果：{state}")
    return state

builder = StateGraph(State)

builder.add_node("increment", increment)
builder.add_node("check_x", check_x)
builder.add_node("done_print", done_print)

builder.add_edge(START, "check_x")
builder.add_conditional_edges("check_x", is_even, {
    False: "done_print",
    True: "increment"
})
builder.add_edge("increment", "check_x")
builder.add_edge("done_print", END)

graph = builder.compile()

print("\n✅ 初始 x=6（偶数，进入循环）")
final_state1 = graph.invoke(State(x=6))
print("[最终结果1] ->", final_state1)

print("\n✅ 初始 x=3（奇数，直接 done）")
final_state2 = graph.invoke(State(x=3))
print("[最终结果2] ->", final_state2)