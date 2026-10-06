from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from model_config import model
from typing import Annotated
from IPython.display import Image, Markdown, display
import operator


class EvaluationsSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")

    score: int = Field(description="Score out of 10", ge=0, le=10)


structured_model = model.with_structured_output(EvaluationsSchema)
essay = """
# AI Impact

Artificial Intelligence is changing the world very fast. It help peoples in many area like education, hospital, business and transport. AI can do works more faster then humans and sometimes it make better decisions also.

In schools, students is using AI for learning new things and solving there homework. This is good because it save time, but some student become lazy and dont learn by themself. Teachers also use AI for checking assignments which make there work easy.

In companies AI is replacing some jobs and many peoples are worried about it. But AI also create new jobs which need different skills. If peoples learn new technology they can work together with AI instead of fear it.

There are also some bad impact of AI. Sometimes AI gives wrong informations or biased answers. It also can be used for making fake videos and fake news, that create problems in society. Because of this, AI should be use carefully and with proper rules.

In conclusion, AI is very useful technology but it also have some disadvantages. We should use AI in a good way so it can helping humans and make life more easier in the future.

"""

# res = strutured_model.invoke(prompt)


# display(Markdown(res.feedback))
class EssayState(BaseModel):
    essay: str

    language_feedback: str = ""
    analysis_feedback: str = ""
    clarity_feedback: str = ""
    overall_feedback: str = ""

    individual_scores: Annotated[list[int], operator.add] = Field(default_factory=list)

    avg_score: float = 0.0


def evaluate_language(state: EssayState):

    prompt = f"""
    Evaluate the language quality of the following essay.

    Check:
    - Grammar
    - Vocabulary
    - Sentence construction
    - Spelling
    - Word usage

    Provide detailed feedback and assign a score out of 10.

    Essay:
    {state.essay}
    """

    output = structured_model.invoke(prompt)

    return {"language_feedback": output.feedback, "individual_scores": [output.score]}


def evaluate_analysis(state: EssayState):

    prompt = f"""
    Evaluate the depth of analysis of the following essay.

    Check:
    - Depth of arguments
    - Supporting points
    - Examples
    - Critical thinking
    - Balance of arguments

    Provide detailed feedback and assign a score out of 10.

    Essay:
    {state.essay}
    """

    output = structured_model.invoke(prompt)

    return {"analysis_feedback": output.feedback, "individual_scores": [output.score]}


def evaluate_thought(state: EssayState):

    prompt = f"""
    Evaluate the clarity of thought of the following essay.

    Check:
    - Logical flow
    - Organization
    - Clarity of ideas
    - Connection between paragraphs
    - Overall coherence

    Provide detailed feedback and assign a score out of 10.

    Essay:
    {state.essay}
    """

    output = structured_model.invoke(prompt)

    return {"clarity_feedback": output.feedback, "individual_scores": [output.score]}


def final_evaluation(state: EssayState):

    prompt = f"""
    Based on the following feedback, create a summarized overall
    evaluation of the essay.

    Language feedback:
    {state.language_feedback}

    Depth of analysis feedback:
    {state.analysis_feedback}

    Clarity of thought feedback:
    {state.clarity_feedback}
    """

    output = model.invoke(prompt)

    avg_score = (
        sum(state.individual_scores) / len(state.individual_scores)
        if state.individual_scores
        else 0.0
    )

    return {"overall_feedback": output.content, "avg_score": avg_score}


# Create graph
graph = StateGraph(EssayState)

# Add nodes
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)
# Parallel execution
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")

# Join parallel branches
graph.add_edge("evaluate_language", "final_evaluation")

graph.add_edge("evaluate_analysis", "final_evaluation")

graph.add_edge("evaluate_thought", "final_evaluation")

# End
graph.add_edge("final_evaluation", END)


workflow = graph.compile()
initial_state = EssayState(essay=essay)
final_state = workflow.invoke(initial_state)

print("Language Feedback:", final_state["language_feedback"])
print("Analysis Feedback:", final_state["analysis_feedback"])
print("Clarity Feedback:", final_state["clarity_feedback"])
print("Overall Feedback:", final_state["overall_feedback"])
print("Average Score:", final_state["avg_score"])
