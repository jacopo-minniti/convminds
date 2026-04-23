from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import logging

from accelerate import Accelerator
from accelerate.logging import get_logger

from convminds.pipelines.base import BasePipeline
from convminds.models.residual_steer import ResidualSteerLM
from convminds.metrics.text import calculate_text_report

logger = get_logger(__name__, log_level="INFO")


def _split_idx(i: int, full_ids: torch.Tensor, full_mask: torch.Tensor, ctx_lens: torch.Tensor) -> int:
    """
    Position of the first target token for sample i in the padded full sequence.
    Clamped so H_query never indexes into padding (handles empty-context edge case).
    """
    pad_len = int(full_ids.shape[1] - full_mask[i].sum().item())
    idx = pad_len + int(ctx_lens[i].item())
    return max(idx, pad_len + 1)


class ResidualSteerPipeline(BasePipeline):
    """
    Pipeline for Brain-to-LLM steering.
    Phase 1 — MSE Warmup: train the adapter to predict the residual delta
      from the last context hidden state to the mean target hidden state.
    Phase 2 — CE Injection: fine-tune the adapter end-to-end with
      cross-entropy loss on steered target-token predictions.
    """

    def __init__(
        self,
        model: ResidualSteerLM,
        lr: float = 1e-4,
        lr_phase2: float | None = None,
        weight_decay: float = 0.01,
        accelerator: Accelerator | None = None,
    ):
        self.accelerator = accelerator or Accelerator()
        self.device = self.accelerator.device
        self.lr = lr
        self.lr_phase2 = lr_phase2 if lr_phase2 is not None else lr / 3
        self.weight_decay = weight_decay

        self.model = model.to(self.device)
        self.optimizer = AdamW(self.model.adapters.parameters(), lr=lr, weight_decay=weight_decay)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        # Keep an unwrapped reference for direct submodule access (DDP does not
        # proxy __getattr__ to the underlying module in all PyTorch versions).
        self.unwrapped = self.accelerator.unwrap_model(self.model)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenize(self, batch: dict, device: torch.device):
        """Unified tokenization: full sequence + context-only for split detection."""
        full_texts = [c + " " + t for c, t in zip(batch["context"], batch["target"])]
        full_enc = self.unwrapped.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
        full_ids = full_enc.input_ids.to(device)
        full_mask = full_enc.attention_mask.to(device)

        ctx_enc = self.unwrapped.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
        ctx_lens = ctx_enc.attention_mask.sum(dim=1).to(device)

        return full_ids, full_mask, ctx_lens

    def _num_steer_tokens(self, full_ids: torch.Tensor, full_mask: torch.Tensor, ctx_lens: torch.Tensor) -> int:
        """
        Number of trailing tokens to steer in Phase 2.
        Covers the target portion of the batch sample with the earliest split point,
        so only target tokens (never padding or context) receive steering.
        """
        pad_lens = full_ids.shape[1] - full_mask.sum(dim=1)
        split_idxs = (pad_lens + ctx_lens).long()
        return max(1, full_ids.shape[1] - int(split_idxs.min().item()))

    def _gather_mean(self, local_avg: float) -> float:
        """Average a scalar metric across all processes."""
        t = torch.tensor(local_avg, device=self.device)
        return self.accelerator.gather(t).mean().item()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        phase_epochs: list[int],
        eval_loader: DataLoader | None = None,
        eval_interval: int = 1,
    ) -> dict[str, float]:
        history = {}
        is_main = self.accelerator.is_main_process

        # --- PHASE 1: MSE Warmup ---
        if phase_epochs[0] > 0:
            logger.info(f"Starting Phase 1: MSE Warmup ({phase_epochs[0]} epochs)")
            for epoch in range(1, phase_epochs[0] + 1):
                self.unwrapped.adapters.train()
                epoch_losses = []
                pbar = tqdm(train_loader, desc=f"Ph1 Ep {epoch}", disable=not is_main)

                for batch in pbar:
                    B = torch.nan_to_num(batch["bold"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
                    B = torch.clamp(B, min=-20.0, max=20.0)
                    full_ids, full_mask, ctx_lens = self._tokenize(batch, self.device)

                    with torch.no_grad():
                        full_out = self.unwrapped.llm(full_ids, attention_mask=full_mask, output_hidden_states=True)

                    layer_losses = []
                    for layer in self.unwrapped.injection_layers:
                        item_losses = []
                        for i in range(B.shape[0]):
                            sidx = _split_idx(i, full_ids, full_mask, ctx_lens)
                            if sidx >= full_ids.shape[1]:
                                continue

                            H_query = full_out.hidden_states[layer][i:i+1, sidx-1:sidx, :]

                            H_target_raw = full_out.hidden_states[layer][i:i+1, sidx:, :]
                            mask_i = full_mask[i:i+1, sidx:].float().unsqueeze(-1)
                            H_target = (
                                torch.sum(H_target_raw * mask_i, dim=1, keepdim=True)
                                / mask_i.sum(dim=1, keepdim=True).clamp(min=1e-8)
                            )

                            delta_target = H_target - H_query
                            v_steer = self.unwrapped.adapters[str(layer)](B[i:i+1], H_query)
                            item_losses.append(F.mse_loss(v_steer, delta_target))

                        if item_losses:
                            layer_losses.append(sum(item_losses) / len(item_losses))

                    if not layer_losses:
                        continue

                    loss = sum(layer_losses) / len(layer_losses)
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    self.optimizer.step()

                    epoch_losses.append(loss.item())
                    pbar.set_postfix({"mse": f"{loss.item():.4f}"})

                local_avg = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
                avg_loss = self._gather_mean(local_avg)
                log_str = f"Phase 1 | Ep {epoch} | Avg MSE: {avg_loss:.6f}"

                if eval_loader and epoch % eval_interval == 0:
                    val_avg = self._eval_phase1(eval_loader)
                    log_str += f" | Val MSE: {val_avg:.6f}"
                    history[f"ph1_epoch_{epoch}_val"] = val_avg

                logger.info(log_str)
                history[f"ph1_epoch_{epoch}"] = avg_loss

        # --- PHASE 2: CE Injection ---
        if len(phase_epochs) > 1 and phase_epochs[1] > 0:
            logger.info(f"Starting Phase 2: CE Injection ({phase_epochs[1]} epochs)")
            # Reset optimizer: fresh moments and lower LR for CE fine-tuning.
            self.optimizer = AdamW(self.unwrapped.adapters.parameters(), lr=self.lr_phase2, weight_decay=self.weight_decay)
            self.optimizer = self.accelerator.prepare(self.optimizer)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(1, phase_epochs[1] + 1):
                self.unwrapped.adapters.train()
                epoch_losses = []
                pbar = tqdm(train_loader, desc=f"Ph2 Ep {epoch}", disable=not is_main)

                for batch in pbar:
                    B = torch.nan_to_num(batch["bold"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
                    B = torch.clamp(B, min=-20.0, max=20.0)
                    full_ids, full_mask, ctx_lens = self._tokenize(batch, self.device)

                    num_steer = self._num_steer_tokens(full_ids, full_mask, ctx_lens)
                    # Call through self.model (DDP wrapper) so gradient sync is handled correctly.
                    logits, _ = self.model(B, full_ids, num_steer_tokens=num_steer, attention_mask=full_mask)

                    item_losses = []
                    for i in range(B.shape[0]):
                        sidx = _split_idx(i, full_ids, full_mask, ctx_lens)
                        if sidx >= full_ids.shape[1]:
                            continue
                        # logit at sidx-1 predicts token at sidx (first target token)
                        target_logits = logits[i, sidx-1:-1, :]
                        target_labels = full_ids[i, sidx:]
                        item_losses.append(criterion(target_logits, target_labels))

                    if not item_losses:
                        continue

                    loss = sum(item_losses) / len(item_losses)
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    self.optimizer.step()

                    epoch_losses.append(loss.item())
                    pbar.set_postfix({"ce_loss": f"{loss.item():.4f}"})

                local_avg = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
                avg_loss = self._gather_mean(local_avg)
                log_str = f"Phase 2 | Ep {epoch} | Avg CE: {avg_loss:.6f}"

                if eval_loader and epoch % eval_interval == 0:
                    val_avg = self._eval_phase2(eval_loader, criterion)
                    log_str += f" | Val CE: {val_avg:.6f}"
                    history[f"ph2_epoch_{epoch}_val"] = val_avg

                logger.info(log_str)
                history[f"ph2_epoch_{epoch}"] = avg_loss

        return history

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _eval_phase1(self, eval_loader: DataLoader) -> float:
        self.unwrapped.adapters.eval()
        val_losses = []
        with torch.no_grad():
            for batch in eval_loader:
                B = torch.nan_to_num(batch["bold"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
                B = torch.clamp(B, min=-20.0, max=20.0)
                full_ids, full_mask, ctx_lens = self._tokenize(batch, self.device)
                full_out = self.unwrapped.llm(full_ids, attention_mask=full_mask, output_hidden_states=True)

                layer_losses = []
                for layer in self.unwrapped.injection_layers:
                    item_losses = []
                    for i in range(B.shape[0]):
                        sidx = _split_idx(i, full_ids, full_mask, ctx_lens)
                        if sidx >= full_ids.shape[1]:
                            continue

                        H_query = full_out.hidden_states[layer][i:i+1, sidx-1:sidx, :]

                        H_target_raw = full_out.hidden_states[layer][i:i+1, sidx:, :]
                        mask_i = full_mask[i:i+1, sidx:].float().unsqueeze(-1)
                        H_target = (
                            torch.sum(H_target_raw * mask_i, dim=1, keepdim=True)
                            / mask_i.sum(dim=1, keepdim=True).clamp(min=1e-8)
                        )

                        delta_target = H_target - H_query
                        v_steer = self.unwrapped.adapters[str(layer)](B[i:i+1], H_query)
                        item_losses.append(F.mse_loss(v_steer, delta_target).item())

                    if item_losses:
                        layer_losses.append(float(np.mean(item_losses)))

                if layer_losses:
                    val_losses.append(float(np.mean(layer_losses)))

        local_avg = float(np.mean(val_losses)) if val_losses else 0.0
        return self._gather_mean(local_avg)

    def _eval_phase2(self, eval_loader: DataLoader, criterion: nn.Module) -> float:
        self.unwrapped.adapters.eval()
        val_losses = []
        with torch.no_grad():
            for batch in eval_loader:
                B = torch.nan_to_num(batch["bold"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
                B = torch.clamp(B, min=-20.0, max=20.0)
                full_ids, full_mask, ctx_lens = self._tokenize(batch, self.device)

                num_steer = self._num_steer_tokens(full_ids, full_mask, ctx_lens)
                logits, _ = self.unwrapped.forward_steered(full_ids, B, num_steer_tokens=num_steer, attention_mask=full_mask)

                item_losses = []
                for i in range(B.shape[0]):
                    sidx = _split_idx(i, full_ids, full_mask, ctx_lens)
                    if sidx >= full_ids.shape[1]:
                        continue
                    target_logits = logits[i, sidx-1:-1, :]
                    target_labels = full_ids[i, sidx:]
                    item_losses.append(criterion(target_logits, target_labels).item())

                if item_losses:
                    val_losses.append(float(np.mean(item_losses)))

        local_avg = float(np.mean(val_losses)) if val_losses else 0.0
        return self._gather_mean(local_avg)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_loader: DataLoader,
        samples_to_show: int = 2,
        max_new_tokens: int = 15,
    ) -> dict[str, float]:
        self.unwrapped.adapters.eval()

        correct_steered = correct_base = correct_random = total = 0
        all_preds_steered, all_preds_base, all_refs = [], [], []
        shown_samples = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation", disable=not self.accelerator.is_main_process):
                B = torch.nan_to_num(batch["bold"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
                B = torch.clamp(B, min=-20.0, max=20.0)

                full_ids, full_mask, ctx_lens = self._tokenize(batch, self.device)

                ctx_enc = self.unwrapped.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
                ctx_ids = ctx_enc.input_ids.to(self.device)
                ctx_mask = ctx_enc.attention_mask.to(self.device)

                outputs_base = self.unwrapped.llm(ctx_ids, attention_mask=ctx_mask)
                logits_steered, _ = self.unwrapped.forward_steered(ctx_ids, B, attention_mask=ctx_mask)

                for i in range(B.shape[0]):
                    sidx = _split_idx(i, full_ids, full_mask, ctx_lens)
                    if sidx >= full_ids.shape[1]:
                        continue

                    target_token = full_ids[i, sidx]
                    pred_base = torch.argmax(outputs_base.logits[i, -1, :])
                    pred_steered = torch.argmax(logits_steered[i, -1, :])

                    correct_base += (pred_base == target_token).item()
                    correct_steered += (pred_steered == target_token).item()
                    correct_random += (torch.randint(0, self.unwrapped.tokenizer.vocab_size, (1,), device=self.device) == target_token).item()
                    total += 1

                gen_base = self.unwrapped.llm.generate(ctx_ids, max_new_tokens=max_new_tokens, attention_mask=ctx_mask, pad_token_id=self.unwrapped.tokenizer.pad_token_id)
                text_base = self.unwrapped.tokenizer.batch_decode(gen_base[:, ctx_ids.shape[1]:], skip_special_tokens=True)
                all_preds_base.extend(text_base)

                gen_steered = self.unwrapped.generate_steered(ctx_ids, B, max_new_tokens=max_new_tokens, attention_mask=ctx_mask)
                text_steered = self.unwrapped.tokenizer.batch_decode(gen_steered[:, ctx_ids.shape[1]:], skip_special_tokens=True)
                all_preds_steered.extend(text_steered)

                all_refs.extend(batch["target"])

                while shown_samples < samples_to_show and shown_samples < len(batch["context"]):
                    logger.info(f"\n--- Sample {shown_samples + 1} ---")
                    logger.info(f"Context: \"{batch['context'][shown_samples]}\"")
                    logger.info(f"Target:  \"{batch['target'][shown_samples]}\"")
                    logger.info(f"Base GPT-2:  \"{text_base[shown_samples].strip()}\"")
                    logger.info(f"Steered:     \"{text_steered[shown_samples].strip()}\"")
                    shown_samples += 1

        # Gather accuracy counters across all processes for correct global metrics.
        counters = torch.tensor(
            [correct_steered, correct_base, correct_random, total],
            dtype=torch.long, device=self.device,
        )
        counters = self.accelerator.reduce(counters, reduction="sum")
        correct_steered, correct_base, correct_random, total = [int(x) for x in counters.tolist()]

        steered_metrics = calculate_text_report(all_preds_steered, all_refs)
        base_metrics = calculate_text_report(all_preds_base, all_refs)

        results = {
            "random_acc": 100 * correct_random / total,
            "base_acc": 100 * correct_base / total,
            "steered_acc": 100 * correct_steered / total,
            "steered_bleu1": steered_metrics["bleu1"],
            "steered_rougeL": steered_metrics["rougeL"],
            "steered_wer": steered_metrics["wer"],
        }

        logger.info("\n" + "=" * 45)
        logger.info(f"{'METRIC':<20} | {'BASE GPT-2':<10} | {'STEERED':<10}")
        logger.info("-" * 45)
        logger.info(f"{'Top-1 Accuracy':<20} | {results['base_acc']:>9.2f}% | {results['steered_acc']:>9.2f}%")
        logger.info(f"{'BLEU-1':<20} | {base_metrics['bleu1']:>10.4f} | {results['steered_bleu1']:>10.4f}")
        logger.info(f"{'ROUGE-L':<20} | {base_metrics['rougeL']:>10.4f} | {results['steered_rougeL']:>10.4f}")
        logger.info(f"{'WER (Lower Better)':<20} | {base_metrics['wer']:>10.4f} | {results['steered_wer']:>10.4f}")
        logger.info("=" * 45)

        return results

    def predict(self, input_ids: torch.Tensor, brain_batch: torch.Tensor, **kwargs):
        self.unwrapped.adapters.eval()
        with torch.no_grad():
            logits, _ = self.unwrapped.forward_steered(input_ids, brain_batch, **kwargs)
        return logits
