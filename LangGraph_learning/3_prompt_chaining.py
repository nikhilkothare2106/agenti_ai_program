from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from pydantic import BaseModel
from model_config import model
from IPython.display import Image, Markdown, display


# create state
class BlogState(BaseModel):
    title: str
    outline: str = ""
    content: str = ""


def create_outline(state: BlogState) -> BlogState:
    # fetch title
    title = state.title

    # call llm to generate outline
    prompt = f"Generate a detailed outline for a blog on the topic - {title}"
    response = model.invoke(prompt).content

    # update state
    outline: str = response if isinstance(response, str) else str(response)
    state.outline = outline
    return state


def create_blog(state: BlogState) -> BlogState:
    title = state.title
    outline = state.outline

    prompt = (
        f"Write a detailed blog on the title - {title} using the outline - {outline}"
    )

    response = model.invoke(prompt).content

    # update state
    content: str = response if isinstance(response, str) else str(response)
    state.content = content

    return state


# create graph
graph = StateGraph(BlogState)

# add nodes
graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)

# add edges
graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", END)

# compile
workflow = graph.compile()

# execute
initial_state = BlogState(title="Rise of AI in india")

final_state = workflow.invoke(initial_state)
print(final_state["content"])

# display(Markdown(f"# {final_state['title']}\n\n{final_state['outline']}\n\n{final_state['content']}"))
