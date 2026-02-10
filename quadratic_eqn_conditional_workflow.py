from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Literal

class QuadraticState(TypedDict):
    a: float
    b: float
    c: float
    eqn: str
    discriminant: float
    roots: str

graph = StateGraph(QuadraticState)

def show_equation(state: QuadraticState) -> QuadraticState:
    equation = f"{state['a']}x^2 + {state['b']}x + {state['c']} = 0"
    return {"eqn": equation}

def calculate_discriminant(state: QuadraticState) -> QuadraticState:
    discriminant= state['b']**2 - (4*state['a']*state['c'])
    return {"discriminant": discriminant}

def calculate_real_roots(state: QuadraticState) -> QuadraticState:
        root1 = (-state['b'] + state['discriminant']**0.5) / (2*state['a'])
        root2 = (-state['b'] - state['discriminant']**0.5) / (2*state['a'])
        roots = f"Two distinct roots: {root1}, {root2}"
        return {"roots": roots}

def calculate_non_real_roots(state: QuadraticState) -> QuadraticState:
    roots = "No real roots"
    return {"roots": roots}

def calculate_repeated_root(state: QuadraticState) -> QuadraticState:
    root = -state['b'] / (2*state['a'])
    roots = f"One repeated root: {root}"
    return {"roots": roots}

def router(state: QuadraticState)-> Literal["calculate_real_roots", "calculate_non_real_roots", "calculate_repeated_root"]:
    if state['discriminant'] > 0:
        return "calculate_real_roots"
    elif state['discriminant'] == 0:
        return "calculate_repeated_root"
    else:
        return "calculate_non_real_roots"

graph.add_node("show_equation", show_equation)
graph.add_node("calculate_discriminant", calculate_discriminant)
graph.add_node("calculate_real_roots", calculate_real_roots)
graph.add_node("calculate_non_real_roots", calculate_non_real_roots)
graph.add_node("calculate_repeated_root", calculate_repeated_root)

graph.add_edge(START, "show_equation")
graph.add_edge("show_equation", "calculate_discriminant")
graph.add_conditional_edges("calculate_discriminant", router)
graph.add_edge("calculate_real_roots", END)
graph.add_edge("calculate_non_real_roots", END)
graph.add_edge("calculate_repeated_root", END)

workflow = graph.compile()

initial_state ={
    "a": 4,
    "b": -5,
    "c": -4,
}

final_state =workflow.invoke(initial_state)

if __name__ == "__main__":
    print(final_state)