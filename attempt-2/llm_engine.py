import torch
from torch.nn import functional as F
from itertools import count
import tiktoken
from sequence import Sequence
from scheduler import Scheduler
from block_manager import BlockManager
from model import GPT


class LLMEngine:
    def __init__(self, model_type="gpt2", max_new_tokens=50, num_blocks=64, block_size=16):
        self.model = GPT.from_pretrained(model_type)
        self.model.eval()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.max_new_tokens = max_new_tokens
        self.block_size = block_size

        # Global KV caches: (n_layer, num_slots, n_head, head_size)
        config = self.model.config
        num_slots = num_blocks * block_size
        head_size = config.n_embd // config.n_head
        self.k_cache = torch.zeros(config.n_layer, num_slots, config.n_head, head_size)
        self.v_cache = torch.zeros(config.n_layer, num_slots, config.n_head, head_size)

        self.block_manager = BlockManager(num_blocks, block_size)
        self.scheduler = Scheduler(self.block_manager)
        self.seq_counter = count(0)
        self._seqs = []

    def add_request(self, prompt, max_new_tokens=None):
        tokens = self.tokenizer.encode(prompt)
        max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        seq = Sequence(next(self.seq_counter), tokens, max_tokens)
        self.scheduler.add_request(seq)
        self._seqs.append(seq)
        return seq

    def _compute_slot(self, seq, logical_pos):
        block_idx = logical_pos // self.block_size
        block_offset = logical_pos % self.block_size
        physical_block_id = seq.block_table[block_idx].block_id
        return physical_block_id * self.block_size + block_offset

    def step(self):
        scheduled_seqs, is_prefill = self.scheduler.schedule()
        if not scheduled_seqs:
            return

        if is_prefill:
            seq = scheduled_seqs[0]
            tokens = seq.get_token_ids()
            T = len(tokens)

            idx = torch.tensor([tokens], dtype=torch.long)
            positions = torch.arange(0, T, dtype=torch.long).unsqueeze(0)  # (1, T)
            slot_mapping = torch.tensor(
                [self._compute_slot(seq, pos) for pos in range(T)],
                dtype=torch.long
            )

            logits, _ = self.model(
                idx, positions,
                k_cache=self.k_cache, v_cache=self.v_cache,
                slot_mapping=slot_mapping, is_prefill=True,
            )
            seq.cache_pos = T

            next_token = self._sample(logits)
            self.scheduler.post_process({seq.seq_id: next_token})

        else:
            # Batched decode
            B = len(scheduled_seqs)

            idx = torch.tensor([[seq.tokens[-1]] for seq in scheduled_seqs], dtype=torch.long)  # (B, 1)
            positions = torch.tensor([[seq.cache_pos] for seq in scheduled_seqs], dtype=torch.long)  # (B, 1)

            # Slot mapping: where to write the new token for each sequence
            slot_mapping = torch.tensor(
                [self._compute_slot(seq, seq.cache_pos) for seq in scheduled_seqs],
                dtype=torch.long
            )

            # Build slot_indices for reading cached K/V: (B, max_ctx)
            # context_lens includes the new token being written
            context_lens = torch.tensor(
                [seq.cache_pos + 1 for seq in scheduled_seqs],
                dtype=torch.long
            )
            max_ctx = context_lens.max().item()

            slot_indices = torch.zeros(B, max_ctx, dtype=torch.long)
            for i, seq in enumerate(scheduled_seqs):
                for pos in range(context_lens[i].item()):
                    slot_indices[i, pos] = self._compute_slot(seq, pos)

            logits, _ = self.model(
                idx, positions,
                k_cache=self.k_cache, v_cache=self.v_cache,
                slot_mapping=slot_mapping, is_prefill=False,
                block_tables=slot_indices, context_lens=context_lens,
            )

            # Batched sampling
            next_tokens = self._sample_batch(logits)
            seq_token_map = {
                seq.seq_id: next_tokens[i].item()
                for i, seq in enumerate(scheduled_seqs)
            }
            self.scheduler.post_process(seq_token_map)

    def _sample(self, logits, temperature=1.0, top_k=None):
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def _sample_batch(self, logits, temperature=1.0, top_k=None):
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

    def generate(self):
        seqs = self._seqs
        while self.scheduler.has_unfinished():
            self.step()
        results = [self.tokenizer.decode(seq.get_token_ids()) for seq in seqs]
        self._seqs = []
        return results
