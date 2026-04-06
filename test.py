# test_1gpu.py
from vllm import LLM, SamplingParams

llm = LLM(
    model="/home/kishoryd/LLM_Bench/data/Param2-17B",
    tensor_parallel_size=1,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.90,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=512,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the BharatGen mission? Answer in 3 sentences."},
]

# Apply chat template
tokenizer = llm.get_tokenizer()
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

outputs = llm.generate([prompt], sampling_params)
for out in outputs:
    print("=" * 60)
    print(out.outputs[0].text)
