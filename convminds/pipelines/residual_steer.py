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
                    B = batch["bold"].to(self.device)
                    
                    # Extract context and target hidden states for ALL layers
                    ctx_enc = self.model.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
                    tgt_enc = self.model.tokenizer(batch["target"], return_tensors="pt", padding=True, truncation=True)
                    
                    ctx_ids, ctx_mask = ctx_enc.input_ids.to(self.device), ctx_enc.attention_mask.to(self.device)
                    tgt_ids, tgt_mask = tgt_enc.input_ids.to(self.device), tgt_enc.attention_mask.to(self.device)
                    
                    # 1. Capture ALL layer hidden states at once efficiently
                    with torch.no_grad():
                        ctx_out = self.model.llm(ctx_ids, attention_mask=ctx_mask, output_hidden_states=True)
                        tgt_out = self.model.llm(tgt_ids, attention_mask=tgt_mask, output_hidden_states=True)
                    
                    total_mse = 0
                    for layer in self.model.injection_layers:
                        H_query = ctx_out.hidden_states[layer][:, -1:, :]
                        
                        # Masked mean for target signal at this layer
                        H_target_raw = tgt_out.hidden_states[layer]
                        mask_expanded = tgt_mask.unsqueeze(-1).float()
                        sum_h = torch.sum(H_target_raw * mask_expanded, dim=1, keepdim=True)
                        count = torch.sum(mask_expanded, dim=1, keepdim=True).clamp(min=1e-8)
                        H_target = sum_h / count
                        
                        delta_target = H_target - H_query
                        v_steer = self.model.adapters[str(layer)](B, H_query)
                        
                        total_mse += F.mse_loss(v_steer, delta_target)
                    
                    loss = total_mse / len(self.model.injection_layers)
                    
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
                            B = batch["bold"].to(self.device)
                            ctx_enc = self.model.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
                            tgt_enc = self.model.tokenizer(batch["target"], return_tensors="pt", padding=True, truncation=True)
                            
                            ctx_ids, ctx_mask = ctx_enc.input_ids.to(self.device), ctx_enc.attention_mask.to(self.device)
                            tgt_ids, tgt_mask = tgt_enc.input_ids.to(self.device), tgt_enc.attention_mask.to(self.device)
                            
                            ctx_out = self.model.llm(ctx_ids, attention_mask=ctx_mask, output_hidden_states=True)
                            tgt_out = self.model.llm(tgt_ids, attention_mask=tgt_mask, output_hidden_states=True)
                            
                            batch_mse = 0
                            for layer in self.model.injection_layers:
                                H_query = ctx_out.hidden_states[layer][:, -1:, :]
                                
                                H_target_raw = tgt_out.hidden_states[layer]
                                mask_expanded = tgt_mask.unsqueeze(-1).float()
                                sum_h = torch.sum(H_target_raw * mask_expanded, dim=1, keepdim=True)
                                count = torch.sum(mask_expanded, dim=1, keepdim=True).clamp(min=1e-8)
                                H_target = sum_h / count
                                
                                delta_target = H_target - H_query
                                v_steer = self.model.adapters[str(layer)](B, H_query)
                                batch_mse += F.mse_loss(v_steer, delta_target)
                            
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
                    B = batch["bold"].to(self.device)
                    
                    # Construction of steers for multi-token phrase loss
                    context_raw = batch["context"]
                    target_raw = batch["target"]
                    
                    ctx_enc = self.model.tokenizer(context_raw, return_tensors="pt", padding=True, truncation=True)
                    tgt_enc = self.model.tokenizer(target_raw, return_tensors="pt", padding=True, truncation=True)
                    
                    ctx_ids, ctx_mask = ctx_enc.input_ids.to(self.device), ctx_enc.attention_mask.to(self.device)
                    tgt_ids, tgt_mask = tgt_enc.input_ids.to(self.device), tgt_enc.attention_mask.to(self.device)
                    
                    full_ids = torch.cat([ctx_ids, tgt_ids], dim=1)
                    full_mask = torch.cat([ctx_mask, tgt_mask], dim=1)
                    num_steer = tgt_ids.shape[1] + 1
                    
                    logits, _ = self.model.forward_steered(full_ids, B, num_steer_tokens=num_steer, attention_mask=full_mask)
                    
                    start_idx = ctx_ids.shape[1] - 1
                    end_idx = full_ids.shape[1] - 1
                    logits_at_target_positions = logits[:, start_idx:end_idx, :].reshape(-1, logits.size(-1))
                    target_labels = full_ids[:, start_idx+1:].reshape(-1)
                    loss = criterion(logits_at_target_positions, target_labels)
                    
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
                            B = batch["bold"].to(self.device)
                            context_raw = batch["context"]
                            target_raw = batch["target"]
                            
                            # Multi-token validation
                            ctx_enc = self.model.tokenizer(context_raw, return_tensors="pt", padding=True, truncation=True)
                            tgt_enc = self.model.tokenizer(target_raw, return_tensors="pt", padding=True, truncation=True)
                            ctx_ids, ctx_mask = ctx_enc.input_ids.to(self.device), ctx_enc.attention_mask.to(self.device)
                            tgt_ids, tgt_mask = tgt_enc.input_ids.to(self.device), tgt_enc.attention_mask.to(self.device)
                            
                            full_ids = torch.cat([ctx_ids, tgt_ids], dim=1)
                            full_mask = torch.cat([ctx_mask, tgt_mask], dim=1)
                            num_steer = tgt_ids.shape[1] + 1
                            
                            logits, _ = self.model.forward_steered(full_ids, B, num_steer_tokens=num_steer, attention_mask=full_mask)
                            
                            start_idx = ctx_ids.shape[1] - 1
                            end_idx = full_ids.shape[1] - 1
                            logits_at_target_positions = logits[:, start_idx:end_idx, :].reshape(-1, logits.size(-1))
                            target_labels = full_ids[:, start_idx+1:].reshape(-1)
                            val_losses.append(criterion(logits_at_target_positions, target_labels).item())
                    
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
                B = batch["bold"].to(self.device)
                ctx_enc = self.model.tokenizer(batch["context"], return_tensors="pt", padding=True, truncation=True)
                input_ids, attention_mask = ctx_enc.input_ids.to(self.device), ctx_enc.attention_mask.to(self.device)
                
                # Single-token prediction targets
                target_tokens = [self.model.tokenizer.encode(" " + t)[0] if len(t) > 0 else self.model.tokenizer.eos_token_id for t in batch["target"]]
                target_label = torch.tensor(target_tokens, device=self.device)
                
                # 1. Baseline: GPT-2 (Single Token for Accuracy, Multi-Token for Sequence Metrics)
                outputs_base = self.model.llm(input_ids, attention_mask=attention_mask)
                preds_base_single = torch.argmax(outputs_base.logits[:, -1, :], dim=-1)
                correct_base += (preds_base_single == target_label).sum().item()
                
                gen_base = self.model.llm.generate(input_ids, max_new_tokens=max_new_tokens, attention_mask=attention_mask, pad_token_id=self.model.tokenizer.pad_token_id)
                new_tokens_base = gen_base[:, input_ids.shape[1]:]
                text_base = self.model.tokenizer.batch_decode(new_tokens_base, skip_special_tokens=True)
                all_preds_base.extend(text_base)
                
                # 2. Intervention: Steered (Single Token for Accuracy, Multi-Token for Sequence Metrics)
                logits_steered, _ = self.model.forward_steered(input_ids, B, attention_mask=attention_mask)
                preds_steered_single = torch.argmax(logits_steered[:, -1, :], dim=-1)
                correct_steered += (preds_steered_single == target_label).sum().item()
                
                gen_steered = self.model.generate_steered(input_ids, B, max_new_tokens=max_new_tokens, attention_mask=attention_mask)
                new_tokens_steered = gen_steered[:, input_ids.shape[1]:]
                text_steered = self.model.tokenizer.batch_decode(new_tokens_steered, skip_special_tokens=True)
                all_preds_steered.extend(text_steered)
                
                # 3. Reference and Random
                all_refs.extend(batch["target"])
                random_preds = torch.randint(0, self.model.tokenizer.vocab_size, (target_label.size(0),), device=self.device)
                correct_random += (random_preds == target_label).sum().item()
                
                total += target_label.size(0)
                
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
