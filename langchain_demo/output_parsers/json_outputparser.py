from config import model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
# template = PromptTemplate(
#     template="Give me the name, age and city of a fictional person \n {format_instruction}",
#     input_variables=[],
#     partial_variables={"format_instruction": parser.get_format_instructions()},
# )

# prompt = template.format()
# print(prompt)

# result = model.invoke(prompt)
# print(result.content)
# final_result = parser.parse(result.content)
# print(final_result)
# print(type(final_result))

# chain = template | model | parser
# final_result = chain.invoke({})
# print(final_result)


template = PromptTemplate(
    template="Give me 5 facts about {topic}  \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)


chain = template | model | parser
final_result = chain.invoke({"topic": "Python programming"})
print(final_result)
