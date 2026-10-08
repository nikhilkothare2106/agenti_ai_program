from langchain_core.callbacks import BaseCallbackHandler


class LoggingHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM started with prompt:", prompts[0][:80])

    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool called: {serialized.get('name')} with {input_str}")

    def on_llm_end(self, response, **kwargs):
        print("\nLLM finished")
