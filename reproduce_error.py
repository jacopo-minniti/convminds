
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from convminds.models.residual_steer import ResidualSteerLM
from convminds.pipelines.residual_steer import ResidualSteerPipeline
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='[%(levelname)s] %(name)s: %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)

class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return {
            "bold": torch.randn(4, 1000),
            "context": "This is a context",
            "target": "and a target",
            "subject": "S1",
            "story": "story1",
            "tr": idx
        }

def test_repro():
    device = torch.device("cpu")
    # Small model for faster testing
    model = ResidualSteerLM(llm_id="gpt2", injection_layers=[6], dropout=0.1)
    pipeline = ResidualSteerPipeline(model=model, lr=1e-4, device=device)
    
    dataset = DummyDataset(size=4)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    print("Starting Phase 1 training...")
    try:
        pipeline.train(loader, phase_epochs=[1], eval_loader=loader, eval_interval=1)
        print("Phase 1 training completed successfully!")
    except Exception as e:
        print(f"Phase 1 training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_repro()
