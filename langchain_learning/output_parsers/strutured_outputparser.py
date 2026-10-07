# from config import model
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema


# schema = [
#     ResponseSchema(name="fact1", description="The first fact about the topic"),
#     ResponseSchema(name="fact2", description="The second fact about the topic"),
#     ResponseSchema(name="fact3", description="The third fact about the topic"),
# ]

# parser = StructuredOutputParser.from_response_schemas(schema)

# template = PromptTemplate(
#     template="Give me 3 facts about {topic}  \n {format_instruction}",
#     input_variables=["topic"],
#     partial_variables={"format_instruction": parser.get_format_instructions()},
# )


# prompt = template.invoke({"topic": "Python programming"})

# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)
