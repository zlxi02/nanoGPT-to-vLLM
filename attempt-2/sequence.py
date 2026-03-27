from enum import Enum


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Sequence:
    def __init__(self, seq_id, prompt_tokens, max_new_tokens):
        self.seq_id = seq_id
        self.tokens = list(prompt_tokens)
        self.num_prompt_tokens = len(prompt_tokens)
        self.max_new_tokens = max_new_tokens
        self.status = SequenceStatus.WAITING
        self.cache_pos = 0
        self.block_table = []  # list of Block objects

    def append_token(self, token):
        self.tokens.append(token)

    def get_num_generated_tokens(self):
        return len(self.tokens) - self.num_prompt_tokens

    def get_token_ids(self):
        return self.tokens

    def is_finished(self):
        return self.get_num_generated_tokens() >= self.max_new_tokens
