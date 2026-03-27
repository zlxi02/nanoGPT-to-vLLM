# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Karpathy's nanoGPT — a minimal GPT-2 training/inference implementation. The goal of this fork is to incrementally build a vLLM-style inference engine on top of nanoGPT's model, following nano-vllm's architecture.

## Key Commands

```bash
# Install dependencies
pip install torch numpy transformers datasets tiktoken wandb tqdm

# Prepare data (character-level Shakespeare — fastest for testing)
python data/shakespeare_char/prepare.py

# Train on Shakespeare (GPU)
python train.py config/train_shakespeare_char.py

# Train on Shakespeare (CPU/Mac)
python train.py config/train_shakespeare_char.py --device=cpu --compile=False

# Sample from a trained model
python sample.py --out_dir=out-shakespeare-char

# Sample from pretrained GPT-2 (no training needed)
python sample.py --init_from=gpt2

# Benchmark model performance
python bench.py

# Multi-GPU training with DDP
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py
```

## Architecture

**Core files (all flat, no packages):**
- `model.py` — Complete GPT model: `GPTConfig` dataclass, `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`, `GPT`. Weight-ties `wte` and `lm_head`. Uses learned position embeddings (not RoPE). Supports loading HuggingFace GPT-2 weights via `GPT.from_pretrained()` (transposes Conv1D→Linear weights).
- `train.py` — Training loop with DDP support. All hyperparameters are top-level globals.
- `sample.py` — Inference script. Loads from checkpoint or pretrained GPT-2.
- `bench.py` — Stripped-down training loop for benchmarking/profiling.
- `configurator.py` — Config override system. Executed via `exec()` into the caller's globals. Accepts a config file path as positional arg and `--key=value` CLI args.

**Configuration pattern:** Config files in `config/` are plain Python that override globals in `train.py`. The `configurator.py` is `exec()`'d into the calling script's namespace — not imported as a module.

**Data pipeline:** Each dataset has a `data/<name>/prepare.py` that produces `train.bin` and `val.bin` (raw uint16 token IDs) and optionally `meta.pkl` (vocab info for char-level models). The training loop uses `np.memmap` to read these directly.

**Model sizes (GPT-2 family):**
- gpt2: 12 layers, 12 heads, 768 embed (124M)
- gpt2-medium: 24 layers, 16 heads, 1024 embed (350M)
- gpt2-large: 36 layers, 20 heads, 1280 embed (774M)
- gpt2-xl: 48 layers, 25 heads, 1600 embed (1558M)

## Key Design Notes

- No KV cache in the base implementation — `generate()` recomputes all attention every step
- Flash Attention used automatically when PyTorch >= 2.0 (via `scaled_dot_product_attention`), falls back to manual attention with causal mask buffer
- `torch.compile` enabled by default; disable with `--compile=False` on unsupported platforms
- Attention uses fused QKV projection (`c_attn` outputs 3*n_embd), then splits
- Residual projection layers (`c_proj`) get special scaled initialization (1/sqrt(2*n_layer))
- `vocab_size` padded to 50304 (nearest multiple of 64) for efficiency when training from scratch; pretrained GPT-2 uses 50257
