from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import logging

from convminds.pipelines.base import BasePipeline
from convminds.models.residual_steer import ResidualSteerLM
from convminds.metrics.text import calculate_text_report

logger = logging.getLogger(__name__)

class ResidualSteerPipeline(BasePipeline):
    """
    Standardized pipeline for Brain-to-LLM steering.
    Encapsulates the two-phase alignment approach:
    1. Phase 1: Global Semantic Alignment (MSE Warmup)
    2. Phase 2: Localized Generative Alignment (CE Injection)
    """

    def __init__(
        self,
        model: ResidualSteerLM,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = AdamW(self.model.adapters.parameters(), lr=lr, weight_decay=weight_decay)

    def train(
        self, 
        train_loader: DataLoader, 
        phase_epochs: list[int], 
        eval_loader: DataLoader | None = None,
        eval_interval: int = 1,
        log_interval: int = 50
    ) -> dict[str, float]:
        """
        Executes the two-phase training loop.
        """
        history = {}
        
        # --- PHASE 1: MSE Warmup ---
        if phase_epochs[0] > 0:
            logger.info(f"Starting Phase 1: MSE Warmup ({phase_epochs[0]} epochs)")
            for epoch in range(1, phase_epochs[0] + 1):
                self.model.adapters.train()
                epoch_losses = []
                pbar = tqdm(train_loader, desc=f"Ph1 Ep {epoch}")
                
                for batch in pbar:
                    B = torch.nan_to_num(batch["bold"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
                    B = torch.clamp(B, min=-20.0, max=20.0)
                    
                    # 1. Unified Tokenization to handle BPE space sensitivity
                    # We tokenize the full sequence to get correct word-boundary tokens
                    full_texts = [c + " " + t for c, t in zip(batch["context"], batch["target"])]
                    full_enc = self.model.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
                    full_ids = full_enc.input_ids.to(self.device)
                    full_mask = full_enc.attention_mask.to(self.device)
                    
                    # Also need just context to find the split point
                    ctx_enc = self.model.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
                    ctx_lens = ctx_enc.attention_mask.sum(dim=1)
                    
                    # 2. Capture ALL layer hidden states at once efficiently
                    with torch.no_grad():
                        full_out = self.model.llm(full_ids, attention_mask=full_mask, output_hidden_states=True)
                    
                    total_mse = 0
                    valid_count = 0
                    for layer in self.model.injection_layers:
                        layer_mse = 0
                        layer_count = 0
                        for i in range(B.shape[0]):
                            pad_len_i = int(full_ids.shape[1] - full_mask[i].sum().item())
                            split_idx = pad_len_i + int(ctx_lens[i].item())

                            if split_idx >= full_ids.shape[1]:
                                continue

                            split_idx = max(split_idx, pad_len_i + 1)
                            H_query = full_out.hidden_states[layer][i:i+1, split_idx-1 : split_idx, :]

                            H_target_raw = full_out.hidden_states[layer][i:i+1, split_idx:, :]
                            mask_i = full_mask[i:i+1, split_idx:].float().unsqueeze(-1)
                            H_target = torch.sum(H_target_raw * mask_i, dim=1, keepdim=True) / mask_i.sum(dim=1, keepdim=True).clamp(min=1e-8)

                            delta_target = H_target - H_query
                            v_steer = self.model.adapters[str(layer)](B[i:i+1], H_query)
                            if type(layer_mse) is int:
                                layer_mse = F.mse_loss(v_steer, delta_target)
                            else:
                                layer_mse = layer_mse + F.mse_loss(v_steer, delta_target)
                            layer_count += 1

                        if layer_count > 0:
                            total_mse += layer_mse / layer_count
                            valid_count += 1

                    if valid_count == 0:
                        continue
                    
                    loss = total_mse / valid_count
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    pbar.set_postfix({"mse": f"{loss.item():.4f}"})
                
                avg_loss = np.mean(epoch_losses)
                log_str = f"Phase 1 | Ep {epoch} | Avg MSE: {avg_loss:.6f}"
                
                # --- Phase 1 Validation ---
                if eval_loader and epoch % eval_interval == 0:
                    self.model.adapters.eval()
                    val_losses = []
                    with torch.no_grad():
                        for batch in eval_loader:
                            B = torch.nan_to_num(batch["bold"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
                            B = torch.clamp(B, min=-20.0, max=20.0)
                            full_texts = [c + " " + t for c, t in zip(batch["context"], batch["target"])]
                            full_enc = self.model.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
                            full_ids = full_enc.input_ids.to(self.device)
                            full_mask = full_enc.attention_mask.to(self.device)
                            
                            ctx_enc = self.model.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
                            ctx_lens = ctx_enc.attention_mask.sum(dim=1)
                            
                            full_out = self.model.llm(full_ids, attention_mask=full_mask, output_hidden_states=True)
                            
                            batch_mse = 0
                            for layer in self.model.injection_layers:
                                layer_mse = 0
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
                                    H_target = torch.sum(H_target_raw * mask_i, dim=1, keepdim=True) / mask_i.sum(dim=1, keepdim=True).clamp(min=1e-8)
                                    
                                    delta_target = H_target - H_query
                                    v_steer = self.model.adapters[str(layer)](B[i:i+1], H_query)
                                    
                                    if type(layer_mse) is int:
                                        layer_mse = F.mse_loss(v_steer, delta_target)
                                    else:
                                        layer_mse = layer_mse + F.mse_loss(v_steer, delta_target)
                                    valid_items += 1
                                    
                                if valid_items > 0:
                                    batch_mse += (layer_mse / valid_items)
                            
                            val_losses.append((batch_mse / len(self.model.injection_layers)).item())
                    
                    val_avg = np.mean(val_losses)
                    log_str += f" | Val MSE: {val_avg:.6f}"
                    history[f"ph1_epoch_{epoch}_val"] = val_avg
                
                logger.info(log_str)
                history[f"ph1_epoch_{epoch}"] = avg_loss

        # --- PHASE 2: CE Injection ---
        if len(phase_epochs) > 1 and phase_epochs[1] > 0:
            logger.info(f"Starting Phase 2: CE Injection ({phase_epochs[1]} epochs)")
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(1, phase_epochs[1] + 1):
                self.model.adapters.train()
                epoch_losses = []
                pbar = tqdm(train_loader, desc=f"Ph2 Ep {epoch}")
                
                for batch in pbar:
                    B = torch.nan_to_num(batch["bold"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
                    B = torch.clamp(B, min=-20.0, max=20.0)
                    
                    # Unified Tokenization
                    full_texts = [c + " " + t for c, t in zip(batch["context"], batch["target"])]
                    full_enc = self.model.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
                    full_ids = full_enc.input_ids.to(self.device)
                    full_mask = full_enc.attention_mask.to(self.device)
                    
                    ctx_enc = self.model.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
                    ctx_lens = ctx_enc.attention_mask.sum(dim=1)
                    
                    # We start steering from the last context token to predict the first target token
                    # max_tgt_len is full_ids.shape[1] - ctx_lens.min()
                    # But since we have padding, it's safer to use the hook on the whole thing 
                    # and ensure num_steer_tokens is large enough.
                    max_steer = full_ids.shape[1] # Simple upper bound
                    
                    logits, _ = self.model.forward_steered(full_ids, B, num_steer_tokens=max_steer, attention_mask=full_mask)
                    
                    # Calculate loss ONLY on the target positions
                    batch_loss = 0
                    valid_items = 0
                    for i in range(B.shape[0]):
                        pad_len_i = int(full_ids.shape[1] - full_mask[i].sum().item())
                        split_idx = pad_len_i + int(ctx_lens[i].item())
                        
                        if split_idx >= full_ids.shape[1]:
                            continue
                            
                        split_idx = max(split_idx, pad_len_i + 1)
                        
                        start_idx = split_idx - 1
                        end_idx = full_ids.shape[1] - 1
                        
                        target_logits = logits[i, start_idx:end_idx, :]
                        target_labels = full_ids[i, start_idx+1:]
                        
                        if type(batch_loss) is int:
                            batch_loss = criterion(target_logits, target_labels)
                        else:
                            batch_loss = batch_loss + criterion(target_logits, target_labels)
                        valid_items += 1
                        
                    if valid_items == 0:
                        batch_loss = logits.sum() * 0.0
                        valid_items = 1
                    
                    loss = batch_loss / valid_items
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    pbar.set_postfix({"ce_loss": f"{loss.item():.4f}"})
                
                avg_loss = np.mean(epoch_losses)
                log_str = f"Phase 2 | Ep {epoch} | Avg CE: {avg_loss:.6f}"
                
                # --- Phase 2 Validation ---
                if eval_loader and epoch % eval_interval == 0:
                    self.model.adapters.eval()
                    criterion = nn.CrossEntropyLoss()
                    val_losses = []
                    with torch.no_grad():
                        for batch in eval_loader:
                            B = torch.nan_to_num(batch["bold"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
                            B = torch.clamp(B, min=-20.0, max=20.0)
                            full_texts = [c + " " + t for c, t in zip(batch["context"], batch["target"])]
                            full_enc = self.model.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
                            full_ids = full_enc.input_ids.to(self.device)
                            full_mask = full_enc.attention_mask.to(self.device)
                            
                            ctx_enc = self.model.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
                            ctx_lens = ctx_enc.attention_mask.sum(dim=1)
                            
                            max_steer = full_ids.shape[1]
                            logits, _ = self.model.forward_steered(full_ids, B, num_steer_tokens=max_steer, attention_mask=full_mask)
                            
                            batch_loss = 0
                            valid_items = 0
                            for i in range(B.shape[0]):
                                pad_len_i = int(full_ids.shape[1] - full_mask[i].sum().item())
                                split_idx = pad_len_i + int(ctx_lens[i].item())
                                
                                if split_idx >= full_ids.shape[1]:
                                    continue
                                    
                                split_idx = max(split_idx, pad_len_i + 1)
                                
                                start_idx = split_idx - 1
                                end_idx = full_ids.shape[1] - 1
                                target_logits = logits[i, start_idx:end_idx, :]
                                target_labels = full_ids[i, start_idx+1:]
                                
                                if type(batch_loss) is int:
                                    batch_loss = criterion(target_logits, target_labels)
                                else:
                                    batch_loss = batch_loss + criterion(target_logits, target_labels)
                                valid_items += 1
                                
                            if valid_items > 0:
                                val_losses.append((batch_loss / valid_items).item())
                    
                    val_avg = np.mean(val_losses)
                    log_str += f" | Val CE: {val_avg:.6f}"
                    history[f"ph2_epoch_{epoch}_val"] = val_avg
                
                logger.info(log_str)
                history[f"ph2_epoch_{epoch}"] = avg_loss

        return history

    def evaluate(
        self, 
        test_loader: DataLoader, 
        samples_to_show: int = 2, 
        max_new_tokens: int = 15
    ) -> dict[str, float]:
        """
        Multi-baseline benchmark comparison using both token-level accuracy 
        and sequence-level metrics (BLEU, ROUGE, WER).
        """
        self.model.adapters.eval()
        
        # Accuracy Counters
        correct_steered = total = 0
        correct_base = 0
        correct_random = 0
        
        # Sequence Lists
        all_preds_steered = []
        all_preds_base = []
        all_refs = []
        
        shown_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation"):
                B = torch.nan_to_num(batch["bold"].to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
                B = torch.clamp(B, min=-20.0, max=20.0)
                
                # Unified Tokenization for Ground Truth tokens
                full_texts = [c + " " + t for c, t in zip(batch["context"], batch["target"])]
                full_enc = self.model.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
                full_ids = full_enc.input_ids.to(self.device)
                full_mask = full_enc.attention_mask.to(self.device)
                
                ctx_enc = self.model.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
                ctx_ids = ctx_enc.input_ids.to(self.device)
                ctx_mask = ctx_enc.attention_mask.to(self.device)
                ctx_lens = ctx_mask.sum(dim=1)
                
                # 1. Baseline: GPT-2 (Base model predicts from context only)
                outputs_base = self.model.llm(ctx_ids, attention_mask=ctx_mask)
                
                # 2. Intervention: Steered
                logits_steered, _ = self.model.forward_steered(ctx_ids, B, attention_mask=ctx_mask)
                
                # Accuracy check: compare predictions for the first token after the context
                for i in range(B.shape[0]):
                    pad_len_i = int(full_ids.shape[1] - full_mask[i].sum().item())
                    split_idx = pad_len_i + int(ctx_lens[i].item())
                    
                    if split_idx >= full_ids.shape[1]:
                        continue # No valid target token to evaluate
                        
                    split_idx = max(split_idx, pad_len_i + 1)
                    
                    # The ground truth first target token from the unified tokenization
                    target_token = full_ids[i, split_idx]
                    
                    pred_base = torch.argmax(outputs_base.logits[i, -1, :])
                    pred_steered = torch.argmax(logits_steered[i, -1, :])
                    
                    correct_base += (pred_base == target_token).item()
                    correct_steered += (pred_steered == target_token).item()
                    
                    random_token = torch.randint(0, self.model.tokenizer.vocab_size, (1,), device=self.device)
                    correct_random += (random_token == target_token).item()
                    total += 1
                
                # Sequence metrics
                gen_base = self.model.llm.generate(ctx_ids, max_new_tokens=max_new_tokens, attention_mask=ctx_mask, pad_token_id=self.model.tokenizer.pad_token_id)
                new_tokens_base = gen_base[:, ctx_ids.shape[1]:]
                text_base = self.model.tokenizer.batch_decode(new_tokens_base, skip_special_tokens=True)
                all_preds_base.extend(text_base)
                
                gen_steered = self.model.generate_steered(ctx_ids, B, max_new_tokens=max_new_tokens, attention_mask=ctx_mask)
                new_tokens_steered = gen_steered[:, ctx_ids.shape[1]:]
                text_steered = self.model.tokenizer.batch_decode(new_tokens_steered, skip_special_tokens=True)
                all_preds_steered.extend(text_steered)
                
                all_refs.extend(batch["target"])
                
                # Qualitative samples
                while shown_samples < samples_to_show and shown_samples < len(batch["context"]):
                    logger.info(f"\n--- Sample {shown_samples + 1} ---")
                    logger.info(f"Context: \"{batch['context'][shown_samples]}\"")
                    logger.info(f"Target:  \"{batch['target'][shown_samples]}\"")
                    logger.info(f"Base GPT-2 Decode: \"{text_base[shown_samples].strip()}\"")
                    logger.info(f"Steered Decode:    \"{text_steered[shown_samples].strip()}\"")
                    shown_samples += 1

        # Calculate Full Text Report
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
        
        logger.info("\n" + "="*45)
        logger.info(f"{'METRIC':<20} | {'BASE GPT-2':<10} | {'STEERED':<10}")
        logger.info("-" * 45)
        logger.info(f"{'Top-1 Accuracy':<20} | {results['base_acc']:>9.2f}% | {results['steered_acc']:>9.2f}%")
        logger.info(f"{'BLEU-1':<20} | {base_metrics['bleu1']:>10.4f} | {results['steered_bleu1']:>10.4f}")
        logger.info(f"{'ROUGE-L':<20} | {base_metrics['rougeL']:>10.4f} | {results['steered_rougeL']:>10.4f}")
        logger.info(f"{'WER (Lower Better)':<20} | {base_metrics['wer']:>10.4f} | {results['steered_wer']:>10.4f}")
        logger.info("=" * 45)
        
        return results

    def predict(self, input_ids: torch.Tensor, brain_batch: torch.Tensor, **kwargs):
        """Standard steering inference."""
        self.model.adapters.eval()
        with torch.no_grad():
            logits, _ = self.model.forward_steered(input_ids, brain_batch, **kwargs)
        return logits
