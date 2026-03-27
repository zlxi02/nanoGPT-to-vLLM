# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **nanoGPT** (Karpathy's), a minimal GPT-2 training/finetuning codebase. The entire model and training loop fit in two files (~300 lines each). The repo name "nanoGPT-to-vllm-practice/attempt-3" suggests this is being used as a practice project for understanding GPT internals.

## Commands

### Install dependencies
```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### Prepare data
```
python data/shakespeare_char/prepare.py   # character-level Shakespeare (quick, for testing)
python data/shakespeare/prepare.py        # BPE-tokenized Shakespeare (for finetuning)
python data/openwebtext/prepare.py        # OpenWebText (large, for GPT-2 reproduction)
```

### Train
```
# Single GPU with a config file
python train.py config/train_shakespeare_char.py

# Single GPU with inline overrides
python train.py --batch_size=32 --compile=False

# Multi-GPU (DDP)
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py

# CPU/Mac
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

### Sample from a trained model
```
python sample.py --out_dir=out-shakespeare-char
python sample.py --init_from=gpt2-xl --start="Hello" --num_samples=5
```

### Benchmark
```
python bench.py                          # with real data
python bench.py --real_data=False        # with synthetic data
python bench.py --profile=True           # with PyTorch profiler (outputs to ./bench_log)
```

## Architecture

**Core files (the entire codebase that matters):**

- `model.py` — Full GPT model definition. Classes: `GPTConfig` (dataclass), `GPT` (main model with `from_pretrained`, `configure_optimizers`, `generate`), `Block`, `CausalSelfAttention`, `MLP`, `LayerNorm`. Uses weight tying between `wte` and `lm_head`. Supports loading HuggingFace GPT-2 weights (requires transposing Conv1D→Linear).

- `train.py` — Training loop with DDP support. All hyperparameters are module-level globals (not argparse). Supports `init_from`: `'scratch'`, `'resume'`, or `'gpt2*'`. Uses cosine LR decay with warmup, gradient accumulation, mixed precision via `torch.amp`, and optional wandb logging. Checkpoints saved to `out_dir` as `ckpt.pt`.

- `sample.py` — Inference script. Loads from checkpoint or pretrained GPT-2. Uses tiktoken (GPT-2 BPE) by default, or character-level encoding if `meta.pkl` exists in the dataset dir.

- `configurator.py` — Config override system. Executed via `exec()` in train/sample/bench scripts. Accepts a config file path as first positional arg, then `--key=value` CLI overrides. Overrides are applied directly to the calling script's `globals()`.

- `config/` — Preset config files (Python scripts that set globals): `train_gpt2.py`, `train_shakespeare_char.py`, `finetune_shakespeare.py`, `eval_gpt2*.py`.

**Configuration pattern:** There is no argparse. All config is done by setting Python globals in `train.py`/`sample.py`/`bench.py`, then `configurator.py` overrides them from config files and CLI args. Config files are just Python scripts executed in the caller's namespace.

**Data format:** Datasets are prepared as `train.bin` and `val.bin` (raw uint16 numpy arrays of token IDs) plus optional `meta.pkl` (vocab mapping for character-level models). Data is loaded via `np.memmap`.
