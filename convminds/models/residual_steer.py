from __future__ import annotations

import torch
import torch.nn as nn
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from convminds.models.base import BrainLanguageModel

logger = logging.getLogger(__name__)

class BrainSteerAdapter(nn.Module):
    """
    Adapter that maps brain activity (4 TRs) to a steering vector 
    via cross-attention with the LLM's current hidden state.
    """
    def __init__(self, brain_dim=1000, llm_dim=768, num_heads=12, n_frames=4, dropout: float = 0.1):
        super().__init__()
        self.brain_dim = brain_dim
        self.llm_dim = llm_dim
        self.n_frames = n_frames
        
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames, brain_dim) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)
        
        self.W_K = nn.Linear(brain_dim, llm_dim)
        self.W_V = nn.Linear(brain_dim, llm_dim)
        self.W_Q = nn.Linear(llm_dim, llm_dim)
        
        # Built-in attention dropout
        self.attn = nn.MultiheadAttention(
            embed_dim=llm_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim * 4, llm_dim),
            nn.Dropout(dropout)
        )

    def forward(self, B, H_query):
        B = self.pos_dropout(B + self.pos_embed)
        K = self.W_K(B)
        V = self.W_V(B)
        Q = self.W_Q(H_query)
        
        A, _ = self.attn(query=Q, key=K, value=V)
        return self.mlp(A)

class ResidualSteerLM(BrainLanguageModel):
    """
    Residual Steering LM with support for multi-layer and multi-token injection.
    """
    def __init__(
        self, 
        llm_id: str = "gpt2", 
        brain_dim: int = 1000, 
        injection_layers: list[int] = [6], 
        n_frames: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.llm = AutoModelForCausalLM.from_pretrained(llm_id)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
            
        self.llm_dim = self.llm.config.hidden_size
        self.injection_layers = sorted(injection_layers)
        
        self.adapters = nn.ModuleDict({
            str(layer): BrainSteerAdapter(
                brain_dim=brain_dim, 
                llm_dim=self.llm_dim, 
                n_frames=n_frames,
                dropout=dropout
            )
            for layer in self.injection_layers
        })
        
        self.freeze_base_model()

    def get_h_at_layer(self, input_ids: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, output_hidden_states=True, **kwargs)
            return outputs.hidden_states[layer_idx]

    def forward(self, brain_batch: torch.Tensor, input_ids: torch.Tensor, **kwargs):
        return self.forward_steered(input_ids, brain_batch, **kwargs)

    def forward_steered(
        self, 
        input_ids: torch.Tensor, 
        brain_batch: torch.Tensor, 
        num_steer_tokens: int = 1,
        **kwargs
    ):
        """
        Execute forward pass with multi-layer residual injection.
        
        Args:
            input_ids: Input tokens.
            brain_batch: Brain activity (B, 4, 1000).
            num_steer_tokens: Number of trailing tokens to apply steering to. 
                              Default 1 (last token of context).
        """
        v_steer_cache = {}

        def get_steering_hook(layer_str):
            def steering_hook(module, inputs, output):
                is_tuple = isinstance(output, tuple)
                hidden_states = output[0] if is_tuple else output
                
                # Dynamic Multi-Token Steering: 
                # Query the brain adapter using the hidden states of ALL tokens 
                # in the steering window independently.
                context_pos = -num_steer_tokens
                H_query = hidden_states[:, context_pos:, :]
                
                v_steer = self.adapters[layer_str](brain_batch, H_query)
                v_steer_cache[layer_str] = v_steer
                
                # Apply dynamic steer to each position
                front_ids = hidden_states[:, :context_pos, :]
                steered_chunk = hidden_states[:, context_pos:, :] + v_steer
                
                steered_hidden_states = torch.cat([front_ids, steered_chunk], dim=1)
                
                if is_tuple:
                    return (steered_hidden_states,) + output[1:]
                return steered_hidden_states
            return steering_hook

        handles = []
        for layer in self.injection_layers:
            target_layer = self.llm.transformer.h[layer - 1]
            handles.append(target_layer.register_forward_hook(get_steering_hook(str(layer))))
        
        try:
            outputs = self.llm(input_ids=input_ids, **kwargs)
        finally:
            for h in handles: h.remove()
            
        return outputs.logits, v_steer_cache

    def generate_steered(
        self, 
        input_ids: torch.Tensor, 
        brain_batch: torch.Tensor, 
        max_new_tokens: int = 15,
        **kwargs
    ) -> torch.Tensor:

        def get_persistent_hook(layer_str):
            def persistent_steering_hook(module, inputs, output):
                is_tuple = isinstance(output, tuple)
                hidden_states = output[0] if is_tuple else output
                
                # Dynamic Autoregressive Steering:
                # Calculate v_steer on-the-fly using the newest token's hidden state
                H_query_current = hidden_states[:, -1:, :]
                v_steer_dynamic = self.adapters[layer_str](brain_batch, H_query_current)
                
                # During generation, we always steer the last token
                # (which is the new token being predicted)
                front_context = hidden_states[:, :-1, :]
                last_token_steered = hidden_states[:, -1:, :] + v_steer_dynamic
                
                steered_hidden_states = torch.cat([front_context, last_token_steered], dim=1)
                
                if is_tuple:
                    return (steered_hidden_states,) + output[1:]
                return steered_hidden_states
            return persistent_steering_hook

        handles = []
        for layer in self.injection_layers:
            target_layer = self.llm.transformer.h[layer - 1]
            handles.append(target_layer.register_forward_hook(get_persistent_hook(str(layer))))

        try:
            generated = self.llm.generate(
                input_ids=input_ids, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        finally:
            for h in handles: h.remove()

        return generated

    def freeze_base_model(self):
        for param in self.llm.parameters():
            param.requires_grad = False
        logger.info(f"Base LLM frozen. Multi-layer steering active at: {self.injection_layers}")