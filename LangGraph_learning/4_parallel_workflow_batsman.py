from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from model_config import model
from IPython.display import Image, Markdown, display


class BatsmanState(BaseModel):
    runs: int
    balls: int
    fours: int
    sixes: int

    strike_rate: float = 0
    balls_per_boundry: float = 0
    boundary_percent: float = 0

    summary: str = ""


def calculate_sr(state: BatsmanState):
    sr = (state.runs / state.balls) * 100
    # state.strike_rate = sr
    return {"strike_rate": sr}


def calculate_bpb(state: BatsmanState):
    bpb = state.balls / (state.fours + state.sixes)
    # state.balls_per_boundry = bpb
    return {"balls_per_boundry": bpb}


def calculate_bp(state: BatsmanState):
    bp = (((state.fours * 4) + (state.sixes * 6)) / state.runs) * 100
    # state.boundary_percent = bp
    return {"boundary_percent": bp}


def summary(state: BatsmanState):
    summary = f"""
        Strike Rate - {state.strike_rate} \n
        Balls per boundry - {state.balls_per_boundry} \n
        Boundry percent - {state.boundary_percent}
    """
    # state.summary = summary
    return {"summary": summary}


graph = StateGraph(BatsmanState)
graph.add_node("calculate_sr", calculate_sr)
graph.add_node("calculate_bpb", calculate_bpb)
graph.add_node("calculate_bp", calculate_bp)
graph.add_node("summary", summary)

graph.add_edge(START, "calculate_sr")
graph.add_edge(START, "calculate_bpb")
graph.add_edge(START, "calculate_bp")

graph.add_edge("calculate_sr", "summary")
graph.add_edge("calculate_bpb", "summary")
graph.add_edge("calculate_bp", "summary")

graph.add_edge("summary", END)

# compile
workflow = graph.compile()
# execute
initial_state = BatsmanState(runs=100, balls=50, fours=6, sixes=4)

final_state = workflow.invoke(initial_state)

png_data = workflow.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_data)

print("Saved graph.png")
