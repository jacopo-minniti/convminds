import sys
import torch
from torch.utils.data import DataLoader
import logging
import argparse

from accelerate import Accelerator
from accelerate.logging import get_logger

import convminds as cm
from convminds.data.benchmarks.huth_alignment import HuthAlignmentDataset
from convminds.models.residual_steer import ResidualSteerLM
from convminds.pipelines.residual_steer import ResidualSteerPipeline

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = get_logger(__name__)

def main():
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(description="Convergent Minds: Phase-based Brain Steering")
    parser.add_argument("--epochs", type=str, default="5,10", help="Comma-separated epochs for Ph1 and Ph2")
    parser.add_argument("--eval-interval", type=int, default=1, help="Epoch interval for validation logging")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Phase 1")
    parser.add_argument("--lr-phase2", type=float, default=None, help="Learning rate for Phase 2 (default: lr/3)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the adapter")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="L2 regularization (weight decay) for AdamW")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--tr-window", type=str, default="1,5", help="BOLD window offset relative to TR X (default: 1,5 for X+1 to X+4)")
    parser.add_argument("--subject", type=str, default="S1", help="Subject ID (e.g., S1, S2)")
    parser.add_argument("--llm", type=str, default="gpt2", help="Base LLM ID from HuggingFace")
    parser.add_argument("--layers", type=str, default="6", help="Comma-separated injection layers (e.g., 2,6,10)")
    args = parser.parse_args()

    cm.set_seed(0)
    phase_epochs = [int(e) for e in args.epochs.split(",")]
    injection_layers = [int(l) for l in args.layers.split(",")]
    tr_window = tuple(int(x) for x in args.tr_window.split(","))

    # 1. Dataset & DataLoaders
    logger.info(f"Initializing Huth Alignment Dataset for Subject {args.subject} (TR Window: {tr_window})...")
    train_set = HuthAlignmentDataset(subject_ids=[args.subject], split="train", tr_window=tr_window)
    test_set = HuthAlignmentDataset(subject_ids=[args.subject], split="test", tr_window=tr_window)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    train_loader, test_loader = accelerator.prepare(train_loader, test_loader)

    # 2. Model & Pipeline
    logger.info(f"Initializing ResidualSteerLM ({args.llm}) with layers {injection_layers} (dropout: {args.dropout})...")
    model = ResidualSteerLM(llm_id=args.llm, injection_layers=injection_layers, dropout=args.dropout)

    pipeline = ResidualSteerPipeline(
        model=model,
        lr=args.lr,
        lr_phase2=args.lr_phase2,
        weight_decay=args.weight_decay,
        accelerator=accelerator,
    )

    # 3. Training
    if any(e > 0 for e in phase_epochs):
        pipeline.train(
            train_loader,
            phase_epochs=phase_epochs,
            eval_loader=test_loader,
            eval_interval=args.eval_interval,
        )

    # 4. Evaluation
    logger.info("Starting Multi-Baseline Evaluation...")
    results = pipeline.evaluate(test_loader, samples_to_show=2)

    # 5. Save — only on main process to avoid concurrent writes
    if accelerator.is_main_process:
        model_dir = cm.cache.ensure_cache_dir("models")
        epochs_str = args.epochs.replace(",", "-")
        layers_str = args.layers.replace(",", "-")
        save_name = f"steer_{args.llm}_{args.subject}_L{layers_str}_ep{epochs_str}.pt"
        save_path = model_dir / save_name
        torch.save(pipeline.unwrapped.adapters.state_dict(), save_path)
        logger.info(f"Steering Adapter weights saved to {save_path}")

if __name__ == "__main__":
    main()
