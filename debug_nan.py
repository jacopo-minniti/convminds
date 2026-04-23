import sys
from unittest.mock import MagicMock
sys.modules['h5py'] = MagicMock()
sys.modules['osfclient'] = MagicMock()
sys.modules['osfclient.api'] = MagicMock()

import torch
import torch.nn.functional as F
from convminds.models.residual_steer import ResidualSteerLM
from convminds.pipelines.residual_steer import ResidualSteerPipeline
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = [
            {"context": "Hello world", "target": "how are you", "bold": torch.randn(4, 1000)},
            {"context": "This is a longer context", "target": "short", "bold": torch.randn(4, 1000)},
        ]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    return {
        "context": [b["context"] for b in batch],
        "target": [b["target"] for b in batch],
        "bold": torch.stack([b["bold"] for b in batch]),
    }

model = ResidualSteerLM(llm_id="gpt2", brain_dim=1000, injection_layers=[6], n_frames=4)
pipeline = ResidualSteerPipeline(model=model, device=torch.device("cpu"))
loader = DataLoader(DummyDataset(), batch_size=2, collate_fn=collate_fn)

pipeline.train(loader, phase_epochs=[1, 1])
