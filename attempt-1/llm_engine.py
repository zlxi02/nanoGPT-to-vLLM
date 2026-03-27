from scheduler import Scheduler
from sequence import Sequence, SequenceStatus
from model import GPT
from block_manager import BlockManager
import tiktoken
import torch
from torch.nn import functional as F

class LLM_Engine():

    def __init__(self, temperature=0.8, top_k=200):
        self.model = GPT.from_pretrained('gpt2')
        self.block_manager = BlockManager(
            total_blocks=self.model.config.num_blocks,
            block_size=self.model.config.kv_block_size
        )
        self.scheduler = Scheduler(self.block_manager)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.temperature = temperature
        self.top_k = top_k
        self.completed = []
        
    def _build_slot_mapping(self, seq, batch_type):
        """Map each token to its physical slot in the KV cache."""
        block_size = self.block_manager.block_size
        slots = []
        if batch_type == "prefill":
            # map all prompt tokens to their slots
            for pos in range(len(seq.token_ids)):
                block_idx = pos // block_size
                block_offset = pos % block_size
                physical_block_id = seq.block_table[block_idx].id
                slots.append(physical_block_id * block_size + block_offset)
        else:
            # decode: just the new token position
            pos = seq.cached_tokens  # where the next token goes
            block_idx = pos // block_size
            block_offset = pos % block_size
            physical_block_id = seq.block_table[block_idx].id
            slots.append(physical_block_id * block_size + block_offset)
        return slots

    def _build_block_table(self, seq):
        """Return list of physical block IDs for this sequence."""
        return [block.id for block in seq.block_table]

    def step(self):
        scheduled_seqs, batch_type = self.scheduler.schedule()
        if not scheduled_seqs:
            return
        sampled_dict = dict()
        block_size = self.block_manager.block_size

        # still process one sequence at a time, but now with paged cache tensors
        for seq in scheduled_seqs:
            # build input tokens
            if batch_type == "prefill":
                idx = torch.tensor(seq.token_ids).unsqueeze(0)
            else:
                idx = torch.tensor([seq.token_ids[-1]]).unsqueeze(0)

            # build slot_mapping: [1, T] tensor
            slots = self._build_slot_mapping(seq, batch_type)
            slot_mapping = torch.tensor(slots).unsqueeze(0)

            # build block_table: [1, num_blocks_for_seq] tensor
            block_ids = self._build_block_table(seq)
            block_table = torch.tensor(block_ids).unsqueeze(0)

            # build context_len: [1] tensor — total tokens in cache after this step
            if batch_type == "prefill":
                context_len = torch.tensor([len(seq.token_ids)])
            else:
                context_len = torch.tensor([seq.cached_tokens + 1])

            # forward pass with paged cache
            logits, _ = self.model.forward(idx, seq.cached_tokens, slot_mapping, block_table, context_len)
            logits = logits[:, -1, :] / self.temperature
            # optionally crop the logits to only the top k options
            if self.top_k is not None:
                v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            sampled = torch.multinomial(probs, num_samples=1)
            sampled_dict[seq.seq_id] = sampled.item()

        completed = self.scheduler.post_process(scheduled_seqs, sampled_dict)
        self.completed.extend(completed)

    def add_request(self, prompt):
        token_ids = self.tokenizer.encode(prompt)
        self.scheduler.add_request(Sequence(token_ids))

    def generate(self):
        while self.scheduler.running or self.scheduler.waiting:
            self.step()

        # decode and return results
        results = []
        for seq in self.completed:
            results.append(self.tokenizer.decode(seq.token_ids))
        return results

        
        
