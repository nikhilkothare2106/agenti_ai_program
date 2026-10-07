from config import model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}", input_variables=["topic"]
)

parser = StrOutputParser()
chain = prompt | model | parser

print(chain.invoke({"topic": "black hole"}))

chain.get_graph().print_ascii()
