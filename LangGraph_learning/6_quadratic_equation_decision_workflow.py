from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal


# -----------------------------
# State
# -----------------------------
class QuadState(BaseModel):
    a: int
    b: int
    c: int

    equation: str = ""
    discriminant: float = 0
    result: str = ""


# -----------------------------
# Node 1: Show equation
# -----------------------------
def show_equation(state: QuadState):
    equation = f"{state.a}x² + {state.b}x + {state.c} = 0"

    return {"equation": equation}


# -----------------------------
# Node 2: Calculate discriminant
# -----------------------------
def calculate_discriminant(state: QuadState):
    discriminant = state.b**2 - (4 * state.a * state.c)

    return {"discriminant": discriminant}


# -----------------------------
# Node 3: Real roots
# -----------------------------
def real_roots(state: QuadState):
    root1 = (-state.b + state.discriminant**0.5) / (2 * state.a)

    root2 = (-state.b - state.discriminant**0.5) / (2 * state.a)

    result = f"The roots are {root1} and {root2}"

    return {"result": result}


# -----------------------------
# Node 4: Repeated root
# -----------------------------
def repeated_roots(state: QuadState):
    root = (-state.b) / (2 * state.a)

    result = f"The repeated root is {root}"

    return {"result": result}


# -----------------------------
# Node 5: No real roots
# -----------------------------
def no_real_roots(state: QuadState):
    result = "No real roots"

    return {"result": result}


# -----------------------------
# Conditional routing
# -----------------------------
def check_condition(
    state: QuadState,
) -> Literal["real_roots", "repeated_roots", "no_real_roots"]:

    if state.discriminant > 0:
        return "real_roots"

    elif state.discriminant == 0:
        return "repeated_roots"

    else:
        return "no_real_roots"


# -----------------------------
# Create graph
# -----------------------------
graph = StateGraph(QuadState)


# Add nodes
graph.add_node("show_equation", show_equation)
graph.add_node("calculate_discriminant", calculate_discriminant)
graph.add_node("real_roots", real_roots)
graph.add_node("repeated_roots", repeated_roots)
graph.add_node("no_real_roots", no_real_roots)
# -----------------------------
# Add edges
# -----------------------------

# START → show equation
graph.add_edge(START, "show_equation")

# show equation → calculate discriminant
graph.add_edge("show_equation", "calculate_discriminant")
# graph.add_edge("calculate_discriminant", END)

# calculate discriminant → conditional routing
graph.add_conditional_edges("calculate_discriminant", check_condition)

# # Each result node → END
graph.add_edge("real_roots", END)
graph.add_edge("repeated_roots", END)
graph.add_edge("no_real_roots", END)


# -----------------------------
# Compile
# -----------------------------
workflow = graph.compile()

initial_state = QuadState(a=2, b=-5, c=2)

final_state = workflow.invoke(initial_state)
print(final_state)
