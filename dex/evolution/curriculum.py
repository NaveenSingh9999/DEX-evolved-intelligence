import numpy as np
import os
from collections import deque, defaultdict

# Expanded character map for richer language
_CHARS = ' abcdefghijklmnopqrstuvwxyz,.!\'?;:-'
_CHAR_MAP = {c: i / (len(_CHARS) - 1) for i, c in enumerate(_CHARS)}
_CHAR_KEYS = list(_CHAR_MAP.keys())

_BOOK_PATHS = ['data/text/alice.txt', 'data/text/hawking.txt']


def _load_book_sentences(paths: list[str]) -> list[str]:
    all_sentences = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8', errors='ignore') as f:
            raw = f.read()
        for line in raw.split('\n'):
            line = line.strip().lower()
            if not line or line.startswith('[') or line.startswith('*'):
                continue
            filtered = ''.join(c for c in line if c in _CHAR_MAP)
            if len(filtered) >= 10:
                all_sentences.append(filtered)
    if not all_sentences:
        all_sentences = ['the neural network learns from data']
    return all_sentences


def _text_to_sequence(text: str, length: int) -> np.ndarray:
    seq = np.array([_CHAR_MAP[c] for c in text[:length]], dtype=np.float32)
    if len(seq) < length:
        seq = np.pad(seq, (0, length - len(seq)), mode='wrap')
    return seq


class Curriculum:
    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()
        self.difficulty = 0.0
        self.error_history: deque[float] = deque(maxlen=100)
        self.input_visit_count: dict[int, int] = defaultdict(int)
        self.decay_rate = 0.05
        self.min_exploration = 0.2
        self.sentences = _load_book_sentences(_BOOK_PATHS)
        self.text_pos = 0

    def generate_batch(self, batch_size: int = 16) -> tuple[list, list]:
        d = self.difficulty
        text_ratio = 0.5
        text_count = max(2, int(batch_size * text_ratio))
        math_count = batch_size - text_count

        inputs_list = []
        targets_list = []

        min_len = 4
        max_len = int(8 + d * 24)
        max_len = min(max_len, 64)

        for _ in range(text_count):
            text = self.sentences[self.text_pos % len(self.sentences)]
            self.text_pos += 1
            seq_len = min(max_len, len(text) - 1)
            seq_len = max(min_len, seq_len)
            inp_len = seq_len
            tgt_len = seq_len
            if len(text) < inp_len + 1:
                text = text + text
            start = self.rng.integers(0, max(1, len(text) - inp_len))
            inp_text = text[start:start + inp_len]
            tgt_text = text[start + 1:start + 1 + tgt_len]
            inp = _text_to_sequence(inp_text, inp_len)
            tgt = _text_to_sequence(tgt_text, tgt_len)
            inputs_list.append(inp)
            targets_list.append(tgt)

        if d < 0.3:
            for _ in range(math_count):
                x = self.rng.standard_normal(4).astype(np.float32)
                inputs_list.append(x)
                targets_list.append(np.sin(x[:1]) * 0.5)
        elif d < 0.6:
            for _ in range(math_count):
                x = self.rng.standard_normal(8).astype(np.float32)
                inputs_list.append(x)
                targets_list.append(np.array([np.sum(x ** 2) * 0.1], dtype=np.float32))
        elif d < 0.8:
            for _ in range(math_count):
                x = self.rng.standard_normal(12).astype(np.float32)
                inputs_list.append(x)
                targets_list.append(np.roll(x, 1))
        else:
            for _ in range(math_count):
                x = self.rng.standard_normal(16).astype(np.float32)
                pattern = np.sin(np.linspace(0, 4 * np.pi, 16)).astype(np.float32)
                inputs_list.append(x + pattern * 0.3)
                targets_list.append(x)

        return inputs_list, targets_list

    def update_difficulty(self, avg_error: float):
        self.error_history.append(avg_error)
        if len(self.error_history) >= 20:
            trend = np.mean(list(self.error_history)[-10:]) - np.mean(list(self.error_history)[:10])
            if trend < -0.001:
                self.difficulty = min(1.0, self.difficulty + 0.02)
            elif trend > 0.001:
                self.difficulty = max(0.0, self.difficulty - 0.01)

    def curiosity_score(self, input_hash: int) -> float:
        visits = self.input_visit_count.get(input_hash, 0)
        raw = 1.0 / (1.0 + visits * self.decay_rate)
        return max(self.min_exploration, raw)

    def record_visit(self, input_hash: int):
        self.input_visit_count[input_hash] += 1
        if len(self.input_visit_count) > 10000:
            old_keys = list(self.input_visit_count.keys())[:1000]
            for k in old_keys:
                del self.input_visit_count[k]
