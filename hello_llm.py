from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading Qwen 0.5B…")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)

prompt = "The capital of France is"

inputs = tokenizer(prompt, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=10)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
