from collections import deque
from sequence import Sequence, SequenceStatus

class Scheduler:
    def __init__(self):
        self.waiting = deque()
        self.running = deque()

    def add_request(self, seq):
        self.waiting.append(seq)

    def schedule(self):
        # prefills take priority: return one waiting request
        if self.waiting:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            return [seq], "prefill"

        # otherwise return all running sequences for decode
        if self.running:
            return list(self.running), "decode"

        return [], None

    def post_process(self, sampled_tokens):
        """sampled_tokens: dict of seq_id -> token_id"""
        finished = []
        for seq in list(self.running):
            if seq.seq_id in sampled_tokens:
                seq.add_token(sampled_tokens[seq.seq_id])
                if seq.is_finished():
                    seq.status = SequenceStatus.FINISHED
                    self.running.remove(seq)
                    finished.append(seq)
        return finished

    def has_requests(self):
        return len(self.waiting) > 0 or len(self.running) > 0
