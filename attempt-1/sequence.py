from enum import Enum, auto
from itertools import count

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence():
    counter = count()

    def __init__(self, token_ids: list[int], max_tokens=256):
        self.status = SequenceStatus.WAITING
        self.token_ids = token_ids
        self.seq_id = next(Sequence.counter)

        # how the scheduler knows if a sequence is prefill or decode
        self.cached_tokens = 0

        # how the scheduler knows a sequence is done. Once max tokens is generated, it's finished
        self.max_tokens = max_tokens

        # length of the original token prompt so we can tell when prefill is done
        self.num_prompt_tokens = len(token_ids)

        self.block_table = []

    def append(self, token):
        self.token_ids.append(token)


    