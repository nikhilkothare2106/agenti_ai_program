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
workflow.invoke({"topic": "pizza"}, config=config1)
workflow.get_state(config1)
list(workflow.get_state_history(config1))
workflow.get_state(
    {
        "configurable": {
            "thread_id": "1",
            "checkpoint_id": "1f19612e-024a-620c-8001-7ba0df6f018f",
        }
    }
)
workflow.invoke(
    None,
    {
        "configurable": {
            "thread_id": "1",
            "checkpoint_id": "1f19612e-024a-620c-8001-7ba0df6f018f",
        }
    },
)
# list(workflow.get_state_history(config1))

for i in workflow.get_state_history(config1):
    print(i, "\n")
workflow.update_state(
    {
        "configurable": {
            "thread_id": "1",
            "checkpoint_id": "1f19612d-fd2d-655f-bfff-b4904beafe15",
            "checkpoint_ns": "",
        }
    },
    {"topic": "samosa"},
)
history = list(workflow.get_state_history(config1))

for k, i in enumerate(range(len(history) - 1, -1, -1)):
    print(k, " ", history[i], "\n")
workflow.invoke(
    None,
    {
        "configurable": {
            "thread_id": "1",
            "checkpoint_id": "1f196132-dfe6-65a5-8000-9176d4598d3d",
        }
    },
)

list(workflow.get_state_history(config1))
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
    print("▶️ Running graph: Please manually interrupt during Step 2...")
    graph.invoke({"input": "start"}, config={"configurable": {"thread_id": "thread-1"}})
except KeyboardInterrupt:
    print("❌ Kernel manually interrupted (crash simulated).")
graph.get_state({"configurable": {"thread_id": "thread-1"}})
list(graph.get_state_history({"configurable": {"thread_id": "thread-1"}}))
# 6. Re-run to show fault-tolerant resume
print("\n🔁 Re-running the graph to demonstrate fault tolerance...")
final_state = graph.invoke(None, config={"configurable": {"thread_id": "thread-1"}})
print("\n✅ Final State:", final_state)
list(graph.get_state_history({"configurable": {"thread_id": "thread-1"}}))
