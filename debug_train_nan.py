import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
llm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

context = ["Hello world", "This is a context"]
target = ["how are you", "short"]
full_texts = [c + " " + t for c, t in zip(context, target)]

full_enc = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
full_ids = full_enc.input_ids.to(device)
full_mask = full_enc.attention_mask.to(device)

ctx_enc = tokenizer(context, return_tensors="pt", padding=True, truncation=True)
ctx_lens = ctx_enc.attention_mask.sum(dim=1)

with torch.no_grad():
    full_out = llm(full_ids, attention_mask=full_mask, output_hidden_states=True)

for i in range(2):
    pad_len_i = int(full_ids.shape[1] - full_mask[i].sum().item())
    split_idx = pad_len_i + int(ctx_lens[i].item())
    split_idx = max(split_idx, pad_len_i + 1)
    
    H_query = full_out.hidden_states[-1][i:i+1, split_idx-1 : split_idx, :]
    H_target_raw = full_out.hidden_states[-1][i:i+1, split_idx:, :]
    mask_i = full_mask[i:i+1, split_idx:].float().unsqueeze(-1)
    H_target = torch.sum(H_target_raw * mask_i, dim=1, keepdim=True) / mask_i.sum(dim=1, keepdim=True).clamp(min=1e-8)
    
    print(f"Batch {i}: H_query nan: {torch.isnan(H_query).any().item()}, H_target nan: {torch.isnan(H_target).any().item()}")
    print(f"Batch {i}: H_query size: {H_query.shape}, H_target size: {H_target.shape}")

