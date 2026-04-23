import sys
import logging
from unittest.mock import MagicMock
sys.modules['h5py'] = MagicMock()
sys.modules['osfclient'] = MagicMock()
sys.modules['osfclient.api'] = MagicMock()

import torch
import numpy as np

import convminds as cm
from convminds.data.benchmarks.huth_alignment import HuthAlignmentDataset
from torch.utils.data import DataLoader
from convminds.models.residual_steer import ResidualSteerLM
from convminds.pipelines.residual_steer import ResidualSteerPipeline

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

device = torch.device("cpu")

# Mock data locally instead of trying to load h5py
class MockHuthDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.samples = []
        for i in range(32):
            self.samples.append({
                "bold": torch.randn(4, 1000),
                "context": "this is a context " + str(i),
                "target": "target " + str(i),
                "subject": "S1",
                "story": "test",
                "tr": i,
                "time_window": (0.0, 4.0)
            })
    def __len__(self): return 32
    def __getitem__(self, idx): return self.samples[idx]

train_set = MockHuthDataset()
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)

model = ResidualSteerLM(llm_id="gpt2", injection_layers=[6])

batch = next(iter(train_loader))

B = torch.nan_to_num(batch["bold"].to(device), nan=0.0)

full_texts = [c + " " + t for c, t in zip(batch["context"], batch["target"])]
full_enc = model.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
full_ids = full_enc.input_ids.to(device)
full_mask = full_enc.attention_mask.to(device)

ctx_enc = model.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
ctx_lens = ctx_enc.attention_mask.sum(dim=1)

with torch.no_grad():
    full_out = model.llm(full_ids, attention_mask=full_mask, output_hidden_states=True)

for layer in model.injection_layers:
    batch_mse = 0
    valid_items = 0
    for i in range(B.shape[0]):
        pad_len_i = int(full_ids.shape[1] - full_mask[i].sum().item())
        split_idx = pad_len_i + int(ctx_lens[i].item())
        
        if split_idx >= full_ids.shape[1]:
            continue
            
        split_idx = max(split_idx, pad_len_i + 1)
        
        H_query = full_out.hidden_states[layer][i:i+1, split_idx-1 : split_idx, :]
        H_target_raw = full_out.hidden_states[layer][i:i+1, split_idx:, :]
        mask_i = full_mask[i:i+1, split_idx:].float().unsqueeze(-1)
        
        div = mask_i.sum(dim=1, keepdim=True).clamp(min=1e-8)
        H_target = torch.sum(H_target_raw * mask_i, dim=1, keepdim=True) / div
        
        delta_target = H_target - H_query
        v_steer = model.adapters[str(layer)](B[i:i+1], H_query)
        
        loss_val = torch.nn.functional.mse_loss(v_steer, delta_target)
        if torch.isnan(loss_val):
            print(f"NAN found at batch index {i}!")
            
        if type(batch_mse) is int:
            batch_mse = loss_val
        else:
            batch_mse = batch_mse + loss_val
        valid_items += 1
        
    print(f"Layer {layer} valid_items: {valid_items}, batch_mse type/val: {type(batch_mse)}")
    total_mse = batch_mse / valid_items
    print(f"total_mse nan: {torch.isnan(total_mse).any().item() if isinstance(total_mse, torch.Tensor) else total_mse}")
