from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# defin state
class BMIState(TypedDict):
    weight: float
    height: float
    bmi: float
    category: str


def calculate_bmi(state: BMIState) -> BMIState:
    weight = state["weight"]
    height = state["height"]

    bmi = weight / (height**2)
    state["bmi"] = round(bmi, 2)
    return state


def label_bmi(state: BMIState) -> BMIState:
    bmi = state["bmi"]
    if bmi < 15:
        state["category"] = "underweight"
    elif 15 < bmi < 25:
        state["category"] = "normal"
    else:
        state["category"] = "overweight"
    return state


# define graph
graph = StateGraph(BMIState)

# add notes to graph
graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node("label_bmi", label_bmi)

# add edges to grph
graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", "label_bmi")
graph.add_edge("label_bmi", END)

# compile the graph
workflow = graph.compile()

# execute the graph
intial_state = {"weight": 80, "height": 2}
final_state = workflow.invoke(intial_state)

print(final_state)


from IPython.display import Image

png_data = workflow.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_data)

print("Saved graph.png")
