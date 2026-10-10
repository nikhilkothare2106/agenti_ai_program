import sys
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from model_config import model as llm
from tools import tools

MAX_STEPS = 10
RECURSION_LIMIT = 3 * MAX_STEPS + 10

# --------------------------------------------------------------------------
# Plan schema and graph state
# --------------------------------------------------------------------------
ToolName = Literal[
    "get_current_weather",
    "wikipedia_lookup",
    "duckduckgo_results_json",
    "search_employee_handbook",
]


class SubTask(BaseModel):
    goal: str = Field(description="What this sub-task must find out or compute")
    tool: ToolName = Field(description="Registered tool best suited to this sub-task")


class Plan(BaseModel):
    sub_tasks: list[SubTask] = Field(
        description=(
            "Useful sub-tasks, ordered so later ones can use earlier results. "
            "One tool call per sub-task. Use fewer only when the query does not "
            "need more; never add filler."
        ),
        min_length=1,
        max_length=MAX_STEPS,
    )


class AgentState(TypedDict):
    query: str
    plan: list[SubTask]
    results: list[str]  # results[i] is the output for plan[i]
    answer: str
    messages: Annotated[list[BaseMessage], add_messages]


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def planner(state: AgentState) -> dict:
    tool_docs = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content=(
                    f"Break the user's request into sequential sub-tasks (at most {MAX_STEPS}). "
                    "Each sub-task is handled by exactly ONE tool call, so give every "
                    "distinct topic its own sub-task (e.g. one handbook search per policy). "
                    "Order them so later steps can build on earlier results. "
                    "Never add filler steps. Pick tools by their descriptions: use the "
                    "handbook for company policy, the web search for current figures, "
                    "and Wikipedia only for general background.\n"
                    "Available tools:\n" + tool_docs
                )
            ),
            HumanMessage(content=state["query"]),
        ]
    )

    if plan is None:
        raise RuntimeError("Planner returned no plan; try rephrasing the query.")

    print("\n--- PLAN ---")
    print(f"Planner created {len(plan.sub_tasks)} tasks.")
    for i, s in enumerate(plan.sub_tasks, 1):
        print(f"Task {i}: {s.tool} | goal={s.goal}")
    print("\n--- EXECUTION LOG ---")

    return {"plan": plan.sub_tasks, "results": []}


def executor(state: AgentState) -> dict:
    """Runs sub-task. It only ASKS for the tool call; the
    ToolNode executes it, and store_tool_result records the output."""

    i = len(state["results"])  # index of the sub-task to run now

    step = state["plan"][i]
    context = (
        "\n".join(
            f"Step {j+1} ({state['plan'][j].goal}): {result}"
            for j, result in enumerate(state["results"])
        )
        or "None yet."
    )

    selected_tools = [t for t in tools if t.name == step.tool]
    tool_choice = {"type": "function", "function": {"name": step.tool}}
    caller = llm.bind_tools(selected_tools, tool_choice=tool_choice)
    msg = caller.invoke(
        [
            SystemMessage(
                content="Call the tool with arguments that accomplish the goal, using prior results where needed."
            ),
            HumanMessage(
                content=f"Original query: {state['query']}\nGoal: {step.goal}\nPrior results:\n{context}"
            ),
        ]
    )

    total = len(state["plan"])
    if msg.tool_calls:
        for call in msg.tool_calls:
            print(f"[{i + 1}/{total}] CALL {call['name']} | args={call['args']}")
    else:
        print(f"[{i + 1}/{total}] {step.tool}: model produced no tool call")

    return {"messages": [msg]}


def store_tool_result(state: AgentState) -> dict:
    """Copies the ToolMessage(s) from this step into `results`."""
    i = len(state["results"])
    step = state["plan"][i]
    tool_messages = []
    for message in reversed(state["messages"]):
        if message.type != "tool":
            break
        tool_messages.append(message)

    output = (
        "\n".join(str(m.content) for m in reversed(tool_messages))
        or "No tool call produced."
    )

    failed = not tool_messages or any(
        getattr(m, "status", None) == "error" for m in tool_messages
    )
    preview = output[:100].replace("\n", " ")
    # print(
    #     f"[{i + 1}/{len(state['plan'])}] {'FAILED' if failed else 'DONE'} "
    #     f"{step.tool} -> {preview}{'...' if len(output) > 100 else ''}"
    # )

    return {"results": state["results"] + [output]}


def synthesizer(state: AgentState) -> dict:
    findings = "\n".join(
        f"{i+1}. {step.goal} -> {result}"
        for i, (step, result) in enumerate(zip(state["plan"], state["results"]))
    )
    answer = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Combine the sub-task results into one clear, direct answer to the original query. "
                    "Write it as a single paragraph only, with no bullet points, no headings, no numbered lists, "
                    "and no separate sections. Mention which findings came from the employee handbook and which came from web searches. "
                    "Note any step that failed."
                )
            ),
            HumanMessage(
                content=f"Query: {state['query']}\n\nSub-task results:\n{findings}"
            ),
        ]
    ).content
    # print("\n--- FINAL ANSWER ---")
    # print(answer[:1200] + ("..." if len(answer) > 1200 else ""))
    return {"answer": answer}


def route_after_executor(
    state: AgentState,
) -> Literal["tools", "store_tool_result"]:
    return "tools" if tools_condition(state) == "tools" else "store_tool_result"


def route_after_store(state: AgentState) -> Literal["executor", "synthesizer"]:
    return "executor" if len(state["results"]) < len(state["plan"]) else "synthesizer"


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("store_tool_result", store_tool_result)
    graph.add_node("synthesizer", synthesizer)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", route_after_executor)
    graph.add_edge("tools", "store_tool_result")
    graph.add_conditional_edges("store_tool_result", route_after_store)
    # graph.add_edge("store_tool_result", "executor")
    graph.add_edge("synthesizer", END)
    return graph.compile()


if __name__ == "__main__":
    user_query = input("Enter a query for the multi-step agent: ")
    print(f"\nUser query: {user_query}")
    graph = build_graph()
    result = graph.invoke({"query": user_query})
    print("\n=== ANSWER ===")
    print(result["answer"])

    # png_data = graph.get_graph().draw_mermaid_png()

    # with open("graph.png", "wb") as f:
    #     f.write(png_data)

    # print("Saved graph.png")


# I’m considering relocating to Mumbai. Search the employee handbook for remote-work, relocation, and PTO policies; check Mumbai’s current weather; and search the web for current average rent and cost of living. Then summarize how these factors affect the move and list questions I should ask HR. Cite which findings came from the handbook and which came from web searches.


# I am considering relocating to Mumbai. Research my company’s remote-work, relocation, and PTO policies, along with Mumbai’s current weather, rental prices, and cost of living. Summarize how these factors could affect my relocation decision and list important questions to ask HR.