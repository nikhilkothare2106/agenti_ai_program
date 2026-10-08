from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
)
from model_config import model

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}", input_variables=["text"]
)

joke_gen_chain = prompt1 | model | parser

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": prompt2 | model | parser,
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({"topic": "cricket"}))
