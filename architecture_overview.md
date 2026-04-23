# Convergent Minds: In-Depth Architecture & Pipeline Analysis

This document provides a highly detailed, technical breakdown of the architecture, data alignment, training strategies, and available ablations within the Convergent Minds (`convminds`) repository, specifically focusing on the execution path of `main.py`.

---

## 1. Architectural Philosophy: Why Residual Steering?

The primary goal of this system is to map human brain representations (captured via fMRI BOLD signals) into the latent space of a Large Language Model (LLM) to reconstruct the semantic content the subject is perceiving.

The codebase supports multiple paradigms for this task:
1. **Prompt Conditioning (`PromptConditionedLM`)**: A traditional approach where brain activity is encoded into "soft prompt" tokens and prepended to the LLM's input embeddings (`inputs_embeds`). 
2. **Residual Steering (`ResidualSteerLM`)**: The default and primary method executed by `main.py`. Instead of prepending tokens at the input layer, it dynamically computes a "steering vector" ($v_{steer}$) that is added as a residual connection directly to the hidden states of intermediate transformer layers (e.g., layer 6). 

**Rationale for Residual Steering:** By injecting the brain signal midway through the LLM, the model can rely on its lower layers to process the syntactic structure of the textual context, while the intermediate injection biases the high-level semantic trajectory toward the brain's perceived meaning without catastrophic forgetting.

---

## 2. Data Pipeline: The `HuthAlignmentDataset`

fMRI recordings have severe temporal constraints due to the sluggish Hemodynamic Response Function (HRF). The blood oxygenation changes peak several seconds *after* the neural event. The `HuthAlignmentDataset` implements a precise, standardized windowing logic to solve this sequence-to-sequence alignment problem.

### 2.1. Temporal Windowing (The "HRF Shift")
For a given target word spoken during Repetition Time (TR) $X$:
- **Target Word ($TR_X$):** The word the subject is hearing and that the LLM must predict.
- **Story Context ($TR_{X-3}$ to $TR_{X-1}$):** The text transcript from the previous 3 TRs, passed to the LLM as linguistic context.
- **Brain Input ($TR_{X+1}$ to $TR_{X+4}$):** A 4-TR window (typically 8 seconds total) of brain activity captured immediately *after* the target word is spoken. This delay deliberately captures the peak BOLD response corresponding to the stimulus at $TR_X$.

### 2.2. Dimensionality Reduction (`VoxelPCA`)
Raw fMRI arrays (tens of thousands of voxels per TR) are too large and noisy for direct neural mapping. 
- For each subject, a Principal Component Analysis (`VoxelPCA`) model is fit strictly on the *training* splits of their story runs.
- The voxel data is projected down to a default of **1000 dimensions** (`pca_dim=1000`).
- The resulting continuous signals are **Z-score normalized** independently for each story run (mean 0, std 1) to remove inter-run baseline drift.

### 2.3. Preprocessing and Text Alignment
Transcripts are rigorously aligned using exact `word_intervals`. To reduce linguistic noise that does not elicit strong semantic BOLD responses, filler tokens (e.g., "uh", "um", "sp") are aggressively filtered out.

---

## 3. Brain Encoders & Ablations

The repository defines several interchangeable modules to map the (4 frames $\times$ 1000 PCA dims) brain signal into the LLM latent space (e.g., 768 dims for GPT-2).

### 3.1. Standard Cross-Attention Adapter (`BrainSteerAdapter`) - *Default*
Used by `ResidualSteerLM`, this module performs cross-attention where the LLM contextualizes the brain signal:
1. **Positional Encoding:** A learnable tensor (`pos_embed`) is added to the 4 BOLD frames to encode temporal sequence.
2. **Cross-Attention:** 
   - **Query ($Q$):** Derived via linear projection from the LLM's *current hidden state* ($H_{query}$).
   - **Keys ($K$) / Values ($V$):** Derived from the 4-frame BOLD tensor sequence.
3. **MLP:** The attention output passes through a 2-layer MLP (expanding by 4x, GELU activation, dropout).
4. **Output:** The final vector represents the semantic residual shift ($v_{steer}$).

### 3.2. MLP & Residual Block Ablations (`BrainEncoder`)
For simpler baselines (found in `brain_adapters.py`), the 4 frames are flattened into a 4000-dimensional vector and passed through standard dense MLPs utilizing `ResidualBlock` architectures (LayerNorm $\rightarrow$ Linear $\rightarrow$ GELU $\rightarrow$ Dropout $\rightarrow$ Linear).

### 3.3. VAE Contrastive Ablation (`VaeBrainAdapter`)
A highly complex ablation that treats the brain encoding as a Variational Autoencoder (VAE) problem.
- **Encoder:** Flattens the 4 frames, processes via an MLP, and projects to a mean ($\mu$) and log-variance ($\log\sigma^2$) in the LLM's dimensional space (768).
- **Reparameterization Trick:** $z = \mu + \epsilon \cdot \sigma$
- **Decoder:** Attempts to reconstruct the original 4000-dim PCA brain signal from the latent $z$ (MSE Loss).
- **Latent Alignment:** During training, this adapter can utilize an **InfoNCE Contrastive Loss** (temperature-scaled cosine similarity) to force the latent brain distribution ($\mu$) to closely match the LLM's text hidden states ($H_{text}$), while a **KL Divergence** loss regularizes the latent space.

---

## 4. Core Architecture: `ResidualSteerLM`

### 4.1. The Frozen Base
The underlying causal language model (e.g., `gpt2`) has `requires_grad=False` applied to all its parameters. The grammatical engine is entirely frozen.

### 4.2. Forward Hook Mechanism
The defining feature of the architecture is its non-invasive modification of the LLM computational graph via PyTorch `register_forward_hook`.
- Hooks are attached to the requested transformer layers (e.g., `self.llm.transformer.h[5]`).
- When the forward pass reaches this layer, the hook intercepts the hidden states ($H_{original}$).
- **Dynamic Multi-Token Steering:** The model separates the "context" tokens from the "target" tokens. It queries the `BrainSteerAdapter` independently for the hidden states of the tokens in the steering window.
- **In-place Shift:** $H_{steered\_chunk} = H_{original\_chunk} + v_{steer}$. The modified tensor is concatenated back with the un-steered context tokens and returned, allowing the forward pass to continue through the upper layers.

---

## 5. The Two-Phase Training Strategy

Aligning continuous brain signals with discrete language tokens is notoriously unstable. `main.py` utilizes `ResidualSteerPipeline` to split the training into two distinct optimization phases via HuggingFace `Accelerate` (DDP support).

### Phase 1: Latent Optimization (MSE Warmup)
**Objective:** Roughly align the adapter's geometric output with the required semantic shift in the hidden space, without the noise of the final softmax layer.
- **Mechanism:** For a given text sample, we isolate the hidden state of the *last token of the context* ($H_{query}$). We then average the hidden states of the actual *target words* that occur in the next TR ($H_{target}$).
- **Target Delta:** The ground truth shift is defined geometrically: $\Delta_{target} = H_{target} - H_{query}$.
- **Loss:** Mean Squared Error (MSE) between the adapter's prediction $v_{steer}$ and the target $\Delta_{target}$.

### Phase 2: End-to-End Fine-Tuning (Cross-Entropy)
**Objective:** Fine-tune the pre-warmed adapter to maximize language modeling performance.
- **Mechanism:** The model runs a standard autoregressive forward pass. The adapter injects its learned offsets dynamically into the targeted layers using the PyTorch hooks.
- **Masking:** Steering is rigorously masked. Padding tokens and historical context tokens *never* receive the residual shift; only the target tokens being predicted are steered.
- **Loss:** Standard Cross-Entropy loss predicting the next target token ID.
- **Optimization:** AdamW optimizer state is reset, and the learning rate is scaled down (default $lr / 3$) to prevent the adapter weights from diverging from their Phase 1 geometric alignment.

---

## 6. Evaluation & Dynamic Generation

Evaluating the steered model requires handling the autoregressive nature of text generation.

### 6.1. Persistent Generation Hooks
To evaluate open-ended text generation, `ResidualSteerLM.generate_steered` utilizes a specialized persistent hook. Because the generation (`llm.generate()`) produces one token at a time, the hook dynamically calculates $v_{steer}$ *on-the-fly* using the hidden state of the newest generated token as the query, injecting the shift just before the model predicts the subsequent word.

### 6.2. Metrics
The pipeline computes rigorous NLP metrics (in `convminds.metrics.text.calculate_text_report`) to measure how well the steered generation matches the actual transcript:
- **Top-1 Accuracy:** The percentage of exact next-token matches, compared against base GPT-2 and a random token baseline. Because LLMs have vocabularies of ~50,000 tokens, even minor improvements in exact Top-1 accuracy over the baseline indicate significant semantic alignment from the brain signal.
- **BLEU-1:** Measures unigram overlap between the generated sequence and the target transcript. It provides a precision-based score: out of all the words the steered model generated, how many actually appeared in the target transcript? It penalizes models that hallucinate irrelevant words.
- **ROUGE-L:** Measures the Longest Common Subsequence (LCS) to evaluate sentence-level structural similarity. Unlike BLEU which looks at isolated words, ROUGE-L requires the words to be in the correct order, evaluating fluency and structural coherence alongside semantic accuracy.
- **WER (Word Error Rate):** A standard speech-recognition metric evaluating the number of substitutions, deletions, and insertions required to match the transcript exactly. Lower is better. WER severely punishes length mismatches and hallucinated text loops, ensuring the generation remains concise and temporally aligned with the audio.

---

## 7. Architectural Ablations & Hyperparameters

The `main.py` entry point and the `ResidualSteerLM` architecture expose several mechanisms to ablate the model and test different alignment hypotheses:

### 7.1. Injection Depth and Multi-Layer Steering (`--layers`)
- **Multiple vs. Single Layer:** The adapter can inject the brain signal into a single layer (e.g., `--layers 6`) or simultaneously into multiple layers (e.g., `--layers 4,8,12`). Multi-layer steering allows the adapter to guide the LLM at multiple levels of abstraction, though it increases the risk of overpowering the LLM's frozen, natural language distribution.
- **Index of the Layer:** The depth of the injection is highly impactful. 
  - **Early Layers (e.g., 2-4):** Injecting here affects low-level syntactic formation and token-level embeddings.
  - **Middle Layers (e.g., 6-8):** Generally the "sweet spot" for fMRI alignment. The LLM has abstracted enough of the text context to build a semantic representation, and there are enough upper layers remaining to decode the brain's semantic shift into a fluent sequence of words.
  - **Late Layers (e.g., 10-12):** Directly biases the final logits, acting almost like an output-space reranker rather than a deep semantic guide.

### 7.2. Temporal Windowing (Number of Tokens to Predict)
- **`num_steer_tokens` (Target Scope):** By default, the adapter applies its computed residual shift strictly to the tokens mapping to the current target TR (the words being spoken). Masking the historical context tokens prevents the brain adapter from destroying the LLM's understanding of the past. Ablating this by steering the *entire* sequence (context + target) typically causes catastrophic degradation of language fluency.
- **Generation Horizon (`max_new_tokens`):** During evaluation, the model auto-regressively predicts `max_new_tokens`. Since the fMRI signal operates on a 2-second resolution, forcing the model to generate too far into the future (e.g., 50 tokens) without fresh brain data will cause the steering vector to become stale, resulting in hallucinated loops.

### 7.3. Regularization (`--dropout`, `--weight-decay`)
- **Overfitting Prevention:** fMRI datasets are notoriously small (only a few hours of audio per subject) and voxel data is extremely noisy. Regularization is vital to prevent the high-capacity `BrainSteerAdapter` from memorizing noise.
- **Weight Decay:** The architecture utilizes `AdamW` weight decay (default `0.01`) during both Phase 1 and Phase 2 to keep adapter weights small, preventing massive residual shifts that would derail the LLM.
- **Dropout:** Heavy dropout (default `0.1`) is utilized inside the Cross-Attention mechanism (`pos_dropout`, attention dropout) and the MLP blocks to enforce robust manifold learning and prevent over-reliance on specific PCA components.

---

## 8. Causal Intervention & Sanity Checks

To ensure that the model is decoding stimulus-specific neural information rather than fitting noise or temporal artifacts, the pipeline supports a **Temporal Lag Causal Intervention** via the `--tr-window` hyperparameter.

### 8.1. The Principle
By default, the model uses brain data from $TR_{X+1 \rightarrow X+4}$ to predict words at $TR_X$. This 1-to-4 TR offset is the "Positive Control" that accounts for the hemodynamic delay.

### 8.2. Causal Directionality Test
Running the model with brain data that occurred *before* the stimulus serves as a negative control:
- **Command:** `python main.py --tr-window -4,0`
- **Logic:** This forces the model to use brain activity from $TR_{X-4 \rightarrow X-1}$ to predict words at $TR_X$.
- **Expected Outcome:** Because the words haven't been spoken yet, there should be no causal neural information in this brain window. If the performance remains high, the model is likely over-fitting on low-frequency artifacts, subject-specific baselines, or story-level "fingerprints" rather than decoding the immediate semantic content. A successful decoding system should see performance drop to the **Base LLM Zero-Shot** level when the window is shifted to a non-causal position.