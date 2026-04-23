import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained("gpt2")
inputs = tokenizer(["test", "a very long test sentence to force padding"], return_tensors="pt", padding=True)

out = model(**inputs, output_hidden_states=True)
print("Hidden states have nan:", torch.isnan(out.hidden_states[-1]).any().item())
print("Pad tokens have nan:", torch.isnan(out.hidden_states[-1][0, 0]).any().item())
