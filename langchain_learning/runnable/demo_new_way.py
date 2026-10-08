from abc import ABC, abstractmethod
import random
from warnings import deprecated

class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass


class DemoLLM(Runnable):
    def __init__(self):
        pass

    def invoke(self, prompt: str) -> dict:
        # Simulate a response from the model
        response_list = [
            "Delhi is the capital of India.",
            "The Taj Mahal is a famous monument in India.",
            "AI stands for Artificial Intelligence.",
        ]
        return {"response": random.choice(response_list)}

    @deprecated(
        "The predict method is deprecated. Please use the invoke method instead."
    )
    def predict(self, prompt: str) -> dict:
        # Simulate a response from the model
        response_list = [
            "Delhi is the capital of India.",
            "The Taj Mahal is a famous monument in India.",
            "AI stands for Artificial Intelligence.",
        ]
        return {"response": random.choice(response_list)}


class DemoPromptTemplate(Runnable):
    def __init__(self, template: str, input_variables: list):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict: dict) -> str:
        return self.template.format(**input_dict)

    def invoke(self, input_dict: dict) -> str:
        prompt = self.template.format(**input_dict)
        return prompt


class DemoStrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data: dict) -> dict:
        # Simulate parsing the output
        return input_data['response']


class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data: dict) -> dict:
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data


template = DemoPromptTemplate(
    template="Write a {length} poem about {topic}.", input_variables=["length", "topic"]
)

# llm = DemoLLM()
# parser = DemoStrOutputParser()

# chain = RunnableConnector([template, llm, parser])
# result = chain.invoke({"length": "short", "topic": "nature"})
# print(f"LLM Response: {result['response']}")


template1 = DemoPromptTemplate(
    template = "Write a joke about {topic}.", input_variables=["topic"]
)
template2 = DemoPromptTemplate(
    template = 'Explain the joke: "{response}".', input_variables=["response"]
)

llm = DemoLLM()
parser = DemoStrOutputParser()
chain1 = RunnableConnector([template1, llm])
joke_result = chain1.invoke({"topic": "programming"})
print(f"Joke: {joke_result['response']}")

chain2 = RunnableConnector([template2, llm, parser])
explanation_result = chain2.invoke({"response": joke_result['response']})
print(f"Explanation: {explanation_result}")

final_chain = RunnableConnector([chain1, chain2])
final_result = final_chain.invoke({"topic": "cricket"})
print(f"Final Joke: {final_result}")