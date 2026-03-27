from llm_engine import LLMEngine                               

engine = LLMEngine(device='cpu')
engine.add_request('Hello, my name is', max_new_tokens=20)
engine.add_request('The meaning of life is', max_new_tokens=20)
                
results = engine.generate()
for i, text in enumerate(results):
    print(f'--- Result {i} ---')
    print(text)      