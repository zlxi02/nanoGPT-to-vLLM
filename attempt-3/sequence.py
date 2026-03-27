from enum import Enum
from itertools import count

class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"

class SequenceCounter:
    _counter = count(0)

    @classmethod
    def next_id(cls):
        return next(cls._counter)

class Sequence:
    def __init__(self, prompt_tokens, max_new_tokens):
        self.seq_id = SequenceCounter.next_id()
        self.status = SequenceStatus.WAITING
        self.prompt_tokens = prompt_tokens
        self.token_ids = list(prompt_tokens)
        self.cached_tokens = 0
        self.max_new_tokens = max_new_tokens

    def add_token(self, token_id):
        self.token_ids.append(token_id)

    def is_finished(self):
        return len(self.token_ids) - len(self.prompt_tokens) >= self.max_new_tokens
