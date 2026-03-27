from math import ceil
from collections import deque
from sequence import SequenceStatus


class Scheduler:
    def __init__(self, block_manager):
        self.block_manager = block_manager
        self.waiting = deque()
        self.running = deque()

    def add_request(self, seq):
        self.waiting.append(seq)

    def schedule(self):
        # Priority: 1 prefill > all decodes
        if self.waiting:
            seq = self.waiting[0]
            num_blocks_needed = ceil(len(seq.tokens) / self.block_manager.block_size)
            if self.block_manager.can_allocate(num_blocks_needed):
                seq = self.waiting.popleft()
                for _ in range(num_blocks_needed):
                    seq.block_table.append(self.block_manager.allocate())
                # Mark how many tokens fill the allocated blocks
                remaining = len(seq.tokens)
                for block in seq.block_table:
                    fill = min(remaining, block.block_size)
                    block.num_tokens = fill
                    remaining -= fill
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                return [seq], True  # is_prefill=True

        if self.running:
            # Allocate new block for sequences whose last block is full
            for seq in self.running:
                last_block = seq.block_table[-1]
                if last_block.is_full():
                    if self.block_manager.can_allocate(1):
                        seq.block_table.append(self.block_manager.allocate())
            return list(self.running), False  # is_prefill=False

        return [], False

    def post_process(self, seq_token_map):
        still_running = deque()
        for seq in self.running:
            if seq.seq_id in seq_token_map:
                seq.append_token(seq_token_map[seq.seq_id])
                seq.cache_pos += 1
                seq.block_table[-1].append_token()
                if seq.is_finished():
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.free(seq.block_table)
                else:
                    still_running.append(seq)
            else:
                still_running.append(seq)
        self.running = still_running

    def has_unfinished(self):
        return len(self.waiting) > 0 or len(self.running) > 0
