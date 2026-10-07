from langchain_core.output_parsers import PydanticOutputParser

from config import model
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate


class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(gt=18, description="The age of the person")
    city: str = Field(description="The city where the person lives")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of the fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# chain = template | model | parser

# result = chain.invoke({"place": "indian"})
# print(result)


prompt = template.invoke({"place": "indian"})

# print(prompt)

result = model.invoke(prompt)
print(result.content)

print(parser.parse(result.content))
