from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch,
    RunnableLambda,
)
from model_config import model

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}", input_variables=["text"]
)

parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | model | parser),
    RunnablePassthrough(),
    # (condition1, runnable1),
    # (condition2, runnable2),
    # default_runnable
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({"topic": "Russia vs Ukraine"}))
