from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Downloading TinyLlama…")

name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.float32
)

print("Model ready!")

prompt = "Say only: Hello LLM"
inputs = tok(prompt, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=30)
print(tok.decode(out[0], skip_special_tokens=True))
