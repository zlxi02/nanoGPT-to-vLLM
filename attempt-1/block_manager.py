from collections import deque
from sequence import Sequence

class Block():
    def __init__(self, block_id, block_size=16):
        self.id = block_id
        self.ref_count = 0
        self.block_size = block_size
        self.num_filled = 0

    def reset(self):
        self.ref_count = 0
        self.num_filled = 0

class BlockManager():
    def __init__(self, total_blocks):
        self.total_blocks = total_blocks
        self.block_list = [Block(i) for i in range(total_blocks)]
        self.free_blocks = self.block_list

    def allocate(self, sequence, num_blocks):
        if num_blocks > len(self.free_blocks):
            raise ValueError("Not enough free blofcks")
        for _ in range(num_blocks):
            block = self.free_blocks.pop()
            block.ref_count += 1
            sequence.block_table.append(block)

    def deallocate(self, block):
        block.ref_count -= 1
        if block.ref_count == 0:
            block.reset()
            self.free_blocks.append(block)


    def can_allocate(self, sequence : Sequence):
        if len(sequence.token_ids) > len(self.free_blocks):
            return True
        else:
            return False
