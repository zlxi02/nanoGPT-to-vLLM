from collections import deque
from sequence import Sequence, SequenceStatus
from block_manager import BlockManager

class Scheduler():
    def __init__(self, block_manager: BlockManager):
        self.waiting = deque()
        self.running = deque()
        self.block_manager = block_manager

    def schedule(self):
        # continue to do prefills while blocks are available
        prefill_list = []
        while self.waiting:
            seq = self.waiting[0]  # peek without popping
            if self.block_manager.can_allocate(seq):
                seq = self.waiting.popleft()
                self.block_manager.allocate(seq)
                self.running.append(seq)
                prefill_list.append(seq)
            else:
                break # break when we run out of space

        if prefill_list:
            return (prefill_list, "prefill")
        

        decode_list = []
        # if no prefills, move all decodes a step forward
        for seq in self.running:
            last_block = seq.block_table[-1]
            if last_block.num_filled < last_block.block_size:
                # room in current block
                last_block.num_filled += 1
                decode_list.append(seq)
            elif len(self.block_manager.free_blocks) > 0:
                # need a new block
                self.block_manager.allocate(seq)
                seq.block_table[-1].num_filled += 1
                decode_list.append(seq)
            else:
                continue

        return (decode_list, "decode")
            

    def add_request(self, seq):
        self.waiting.append(seq)

    def post_process(self, seq_list : list[Sequence], sampled_tokens : dict):
        completed_sequences = []
        for sequence in seq_list:

            # append token first
            sequence.token_ids.append(sampled_tokens[sequence.seq_id])
            # update the cached tokens after prefill / decode
            if sequence.cached_tokens == 0:
                sequence.cached_tokens = sequence.num_prompt_tokens
            else:
                sequence.cached_tokens += 1

            # checks if the sequence has finished
            if (len(sequence.token_ids) - sequence.num_prompt_tokens >= sequence.max_tokens) or (sequence.token_ids[-1] == 50256):
                completed_sequences.append(sequence)
                sequence.status = SequenceStatus.FINISHED
        
        # removes done sequences and frees their blocks
        for seq in completed_sequences:
            self.running.remove(seq)
            for block in seq.block_table:
                self.block_manager.deallocate(block)

        return completed_sequences
        