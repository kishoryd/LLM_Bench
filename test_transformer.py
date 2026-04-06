import sys
sys.path.append("/home/kishoryd/LLM_Bench/data/Param2-17B")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from parsers import parse_model_output

model_name = "bharatgenai/Param2-17B-A2.4B-Thinking"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto"
)

conversation = [
    {"role": "system", "content": "You are helpful assistant."},
    {"role": "user", "content": "What is the BharatGen Mission?"}
]

inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

with torch.no_grad():
    output = model.generate(
        inputs,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

generated_tokens = output[0][inputs.shape[-1]:]

# 🔥 IMPORTANT: skip_special_tokens=False
generated_text = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=False
)

parsed = parse_model_output(generated_text)

print("\n========== RAW ==========\n", generated_text)
print("\n========== REASONING ==========\n", parsed["reasoning"])
print("\n========== TOOL CALLS ==========\n", parsed["tool_calls"])
print("\n========== FINAL ANSWER ==========\n", parsed["final_answer"])

