import numpy as np
import json
import os
from collections import deque
from dataclasses import dataclass, field, asdict


@dataclass
class MemoryEntry:
    input_hash: int
    prediction: float
    error: float
    surprise: float
    timestamp: int


class EpisodicMemory:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: deque[MemoryEntry] = deque(maxlen=capacity)
        self.step = 0

    def push(self, entry: MemoryEntry):
        entry.timestamp = self.step
        self.buffer.append(entry)
        self.step += 1

    def sample(self, n: int, rng: np.random.Generator) -> list:
        if len(self.buffer) < n:
            return list(self.buffer)
        priorities = np.array([e.surprise + 0.01 for e in self.buffer])
        probs = priorities / priorities.sum()
        idxs = rng.choice(len(self.buffer), size=n, p=probs, replace=False)
        return [self.buffer[i] for i in idxs]

    def recent(self, n: int) -> list:
        return list(self.buffer)[-n:]

    def save(self, path: str):
        data = [asdict(e) for e in self.buffer]
        with open(path, 'w') as f:
            json.dump({'step': self.step, 'entries': data}, f)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self.step = data['step']
            self.buffer = deque(
                [MemoryEntry(**e) for e in data['entries']],
                maxlen=self.capacity,
            )
