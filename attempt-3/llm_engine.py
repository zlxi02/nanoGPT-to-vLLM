import torch
import tiktoken
from torch.nn import functional as F
from model import GPT
from scheduler import Scheduler
from sequence import Sequence

class LLMEngine:
    def __init__(self, device="cpu", temperature=0.8, top_k=200):
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.model = GPT.from_pretrained("gpt2", dict(dropout=0.0))
        self.model.eval()
        self.model.to(device)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.scheduler = Scheduler()
        self.completed = []

    def add_request(self, prompt, max_new_tokens=50):
        tokens = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        seq = Sequence(tokens, max_new_tokens)
        self.scheduler.add_request(seq)

    def generate(self):
        while self.scheduler.has_requests():
            self.step()

        results = []
        for seq in self.completed:
            text = self.tokenizer.decode(seq.token_ids)
            results.append(text)

        self.completed = []
        return results

    def sample(self, logits):
        logits = logits[:, -1, :] / self.temperature
        if self.top_k is not None:
            v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    @torch.no_grad()
    def step(self):
        scheduled_seqs, mode = self.scheduler.schedule()
        if not scheduled_seqs:
            return

        sampled_tokens = {}

        for seq in scheduled_seqs:
            if mode == "prefill":
                input_ids = torch.tensor([seq.token_ids], dtype=torch.long, device=self.device)
                start_pos = 0
            else:
                input_ids = torch.tensor([[seq.token_ids[-1]]], dtype=torch.long, device=self.device)
                start_pos = seq.cached_tokens

            logits, _ = self.model(input_ids, start_pos=start_pos)

            if mode == "prefill":
                seq.cached_tokens = len(seq.token_ids)
            else:
                seq.cached_tokens += 1

            next_token = self.sample(logits)
            sampled_tokens[seq.seq_id] = next_token

        finished = self.scheduler.post_process(sampled_tokens)
        self.completed.extend(finished)
