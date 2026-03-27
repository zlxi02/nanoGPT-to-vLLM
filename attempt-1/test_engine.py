from llm_engine import LLM_Engine
                                                                                                                                                                                                                                                                                                                       
engine = LLM_Engine()
engine.add_request("Hello, my name is")                                                                                                                                                                                                                                                                              
engine.add_request("The meaning of life is")                                                                                                                                                                                                                                                                       
results = engine.generate()
for r in results:
    print(r)
    print("---")
