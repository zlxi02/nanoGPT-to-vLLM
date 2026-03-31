# nanoGPT-to-vLLM

Building a vLLM-style inference engine on top of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), from scratch, three times.

Each attempt implements the core vLLM components — **KV cache management**, **paged attention**, **scheduling with prefill/decode separation**, and **continuous batching** — layered on top of nanoGPT's GPT-2 model.

## Structure

```
attempt-1/   # First pass — paged KV cache with block tables and slot mapping
attempt-2/   # Second pass — global KV cache tensors, batched decode, cleaner block manager
attempt-3/   # Third pass — simplified architecture, KV cache in model, minimal scheduler
```

Each attempt is a self-contained copy of nanoGPT plus the added inference engine files.

## What's added on top of nanoGPT

Each attempt adds these vLLM-inspired components:

| File | Purpose |
|------|---------|
| `llm_engine.py` | Top-level engine: tokenization, model forward pass, sampling loop |
| `scheduler.py` | Prefill-first scheduling with waiting/running queues |
| `sequence.py` | Sequence state: token IDs, generation status, cache position |
| `block_manager.py` | Physical block allocation/deallocation for paged KV cache |
| `model.py` | Modified nanoGPT model with KV cache support (paged or positional) |
| `test_engine.py` | Quick smoke test — generate from multiple prompts |

## How the attempts differ

**Attempt 1** — Paged attention with explicit slot mapping and block tables passed to the model. Sequences processed one at a time. Block manager tracks `ref_count` and `num_filled` per block.

**Attempt 2** — Global KV cache tensors allocated upfront (`k_cache`, `v_cache` as `[n_layer, num_slots, n_head, head_size]`). Batched decode with `slot_indices` for reading cached K/V. Cleaner block/scheduler separation.

**Attempt 3** — Most minimal version. KV cache lives inside the model (like the original nanoGPT `generate()`). No block manager — the scheduler is just a simple prefill-priority queue. Focus on getting the core prefill/decode loop right with the least abstraction.

## Quick start

```bash
cd attempt-3  # or attempt-1, attempt-2
python -m venv venv && source venv/bin/activate
pip install torch numpy transformers tiktoken

python test_engine.py
```

This downloads GPT-2 (124M) weights from HuggingFace on first run, then generates continuations for two test prompts.

## Based on

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [vLLM](https://github.com/vllm-project/vllm) architecture (PagedAttention, continuous batching)
- [nano-vllm](https://github.com/GeeeekExplworworker/nano-vllm) for reference
