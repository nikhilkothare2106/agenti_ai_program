from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from model_config import model as llm


class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str


def generate_joke(state: JokeState):

    prompt = f'generate a joke on the topic {state["topic"]}'
    response = llm.invoke(prompt).content

    return {"joke": response}


def generate_explanation(state: JokeState):

    prompt = f'write an explanation for the joke in short - {state["joke"]}'
    response = llm.invoke(prompt).content

    return {"explanation": response}


graph = StateGraph(JokeState)

graph.add_node("generate_joke", generate_joke)
graph.add_node("generate_explanation", generate_explanation)

graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "generate_explanation")
graph.add_edge("generate_explanation", END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)
config1 = {"configurable": {"thread_id": "1"}}

print("\n=== Starting joke generation workflow ===")
print("Invoking workflow with topic='pizza'")
result_1 = workflow.invoke({"topic": "pizza"}, config=config1)
print("First workflow result:")
print(result_1)

print("\nCurrent state after first invoke:")
print(workflow.get_state(config1))

print("\nState history after first invoke:")
history = list(workflow.get_state_history(config1))
for entry in history:
    print(entry, "\n")

latest_configurable = history[-2].config.get("configurable") or {}
checkpoint_id = latest_configurable.get("checkpoint_id")
if checkpoint_id is None:
    raise ValueError("No checkpoint_id found in the latest workflow history entry.")

print(f"\nUsing checkpoint_id from latest history state: {checkpoint_id}")
print(
    workflow.get_state(
        {
            "configurable": {
                "thread_id": "1",
                "checkpoint_id": checkpoint_id,
            }
        }
    )
)

print("\nRe-invoking workflow with checkpoint resume:")
resume_result = workflow.invoke(
    None,
    {
        "configurable": {
            "thread_id": "1",
            "checkpoint_id": checkpoint_id,
        }
    },
)
print(resume_result)

print("\nHistory after resume invoke:")
for i in workflow.get_state_history(config1):
    print(i, "\n")

print("\nUpdating state with new topic='samosa'")
last_history_state = list(workflow.get_state_history(config1))[-1]
last_configurable = last_history_state.config.get("configurable") or {}
last_checkpoint_id = last_configurable.get("checkpoint_id")
if last_checkpoint_id is None:
    raise ValueError("No checkpoint_id found for the state update.")

update_config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": last_checkpoint_id,
        "checkpoint_ns": last_configurable.get("checkpoint_ns", ""),
    }
}
workflow.update_state(update_config, {"topic": "samosa"})

history = list(workflow.get_state_history(config1))
print("Updated state history in reverse order:")
for k, i in enumerate(range(len(history) - 1, -1, -1)):
    print(k, " ", history[i], "\n")

print("\nFinal workflow invoke after state update:")
final_history_state = list(workflow.get_state_history(config1))[-1]
final_checkpoint_id = (final_history_state.config.get("configurable") or {}).get(
    "checkpoint_id"
)
if final_checkpoint_id is None:
    raise ValueError("No checkpoint_id found for the final resume invocation.")

final_result = workflow.invoke(
    None,
    {
        "configurable": {
            "thread_id": "1",
            "checkpoint_id": final_checkpoint_id,
        }
    },
)
print(final_result)

print("\nFull final state history:")
for state in workflow.get_state_history(config1):
    print(state)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
import time


# 1. Define the state
class CrashState(TypedDict):
    input: str
    step1: str
    step2: str


# 2. Define steps
def step_1(state: CrashState) -> CrashState:
    print("✅ Step 1 executed")
    return {"step1": "done", "input": state["input"]}


def step_2(state: CrashState) -> CrashState:
    print(
        "⏳ Step 2 hanging... now manually interrupt from the notebook toolbar (STOP button)"
    )
    time.sleep(20)  # Simulate long-running hang
    return {"step2": "done"}


def step_3(state: CrashState) -> CrashState:
    print("✅ Step 3 executed")
    return {"step3": "done"}


# 3. Build the graph
builder = StateGraph(CrashState)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

builder.set_entry_point("step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
try:
    print("\n▶️ Running crash graph: Please manually interrupt during Step 2...")
    graph.invoke({"input": "start"}, config={"configurable": {"thread_id": "thread-1"}})
except KeyboardInterrupt:
    print("❌ Kernel manually interrupted (crash simulated).")

print("\nCurrent state after crash simulation:")
print(graph.get_state({"configurable": {"thread_id": "thread-1"}}))

print("\nState history after crash simulation:")
for entry in graph.get_state_history({"configurable": {"thread_id": "thread-1"}}):
    print(entry)

# 6. Re-run to show fault-tolerant resume
print("\n🔁 Re-running the graph to demonstrate fault tolerance...")
final_state = graph.invoke(None, config={"configurable": {"thread_id": "thread-1"}})
print("\n✅ Final State:", final_state)

print("\nState history after resume:")
for entry in graph.get_state_history({"configurable": {"thread_id": "thread-1"}}):
    print(entry)
