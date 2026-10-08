import random

class DemoLLM:
    def __init__(self):
        print("DemoLLM initialized")

    def predict(self, prompt: str) -> dict:
        # Simulate a response from the model
        response_list = [
            "Delhi is the capital of India.",
            "The Taj Mahal is a famous monument in India.",
            "AI stands for Artificial Intelligence.",
        ]
        return {"response": random.choice(response_list)}


# llm = DemoLLM()
# print(llm.predict("What is the capital of India?"))


class DemoPromptTemplate:
    def __init__(self, template: str, input_variables: list):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict: dict) -> str:
        return self.template.format(**input_dict)


template = DemoPromptTemplate(
    template="Write a {length} poem about {topic}.", input_variables=["length", "topic"]
)
prompt = template.format({"length": "short", "topic": "nature"})
# llm = DemoLLM()
# llm_response = llm.predict(prompt)

# print(f"LLM Response: {llm_response['response']}")


class DemoLLMChain:
    def __init__(self, prompt: DemoPromptTemplate, llm: DemoLLM):
        self.prompt = prompt
        self.llm = llm

    def run(self, input_dict: dict) -> dict:
        prompt = self.prompt.format(input_dict)
        result = self.llm.predict(prompt)
        return result["response"]


llm = DemoLLM()
chain = DemoLLMChain(prompt=template, llm=llm)

print(chain.run({"length": "short", "topic": "nature"}))
