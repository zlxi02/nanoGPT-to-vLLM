from collections import deque


class Block:
    def __init__(self, block_id, block_size):
        self.block_id = block_id
        self.block_size = block_size
        self.num_tokens = 0

    def is_full(self):
        return self.num_tokens >= self.block_size

    def append_token(self):
        self.num_tokens += 1


class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks = deque(range(num_blocks))

    def can_allocate(self, num_blocks_needed):
        return len(self.free_blocks) >= num_blocks_needed

    def allocate(self):
        block_id = self.free_blocks.popleft()
        return Block(block_id, self.block_size)

    def free(self, blocks):
        for block in blocks:
            self.free_blocks.append(block.block_id)

    def get_num_free_blocks(self):
        return len(self.free_blocks)
