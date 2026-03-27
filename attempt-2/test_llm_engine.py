from llm_engine import LLMEngine

engine = LLMEngine(max_new_tokens=3)
engine.add_request("Hello, my name is")
engine.add_request("The meaning of life is")
results = engine.generate()
for r in results:
    print(r)
    print("---")
