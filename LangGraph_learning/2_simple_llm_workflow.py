from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from pydantic import BaseModel
from model_config import model
from IPython.display import Image


# create state
class LLMState(BaseModel):
    question: str
    answer: str = ""


def llm_qa(state: LLMState) -> LLMState:
    question = state.question
    prompt = f"Answer the following question, {question}"

    raw = model.invoke(prompt).content
    answer_text: str = raw if isinstance(raw, str) else str(raw)

    state.answer = answer_text
    return state


# create graph
graph = StateGraph(LLMState)

# add nodes
graph.add_node("llm_qa", llm_qa)

# add edges
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)

# compile
workflow = graph.compile()

# execute
initial_state = LLMState(question="how far is the moon from the earth?")
final_state = workflow.invoke(initial_state)

print(final_state["answer"])


# png_data = workflow.get_graph().draw_mermaid_png()

# with open("graph.png", "wb") as f:
#     f.write(png_data)

# print("Saved graph.png")
