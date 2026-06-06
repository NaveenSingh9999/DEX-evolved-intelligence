import numpy as np
from collections import deque, defaultdict

# Small embedded text dataset — Wikipedia-style sentences from day one.
_TEXTS = [
    "the neural network learns from data and improves over time",
    "evolution is the process of change across successive generations",
    "a genome contains all the information needed to build an organism",
    "curiosity drives exploration and discovery in intelligent systems",
    "memory allows agents to store and recall past experiences",
    "society depends on communication and cooperation between individuals",
    "technology advances through innovation and creative problem solving",
    "nature selects traits that improve survival and reproduction",
    "patterns emerge from complex systems through simple local rules",
    "knowledge grows when information is compressed and connected",
]

_CHAR_MAP = {c: i / 64.0 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz ,.!\'')}
_CHAR_KEYS = list(_CHAR_MAP.keys())


def _text_to_sequence(text: str, length: int) -> np.ndarray:
    chars = [c for c in text.lower() if c in _CHAR_MAP]
    if not chars:
        chars = ['a']
    seq = np.array([_CHAR_MAP[c] for c in chars[:length]], dtype=np.float32)
    if len(seq) < length:
        seq = np.pad(seq, (0, length - len(seq)), mode='wrap')
    return seq


class Curriculum:
    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()
        self.difficulty = 0.0
        self.error_history: deque[float] = deque(maxlen=100)
        # Novelty decay tracker: input_hash -> times_seen
        self.input_visit_count: dict[int, int] = defaultdict(int)
        self.decay_rate = 0.05
        self.min_exploration = 0.2
        self.text_index = 0

    def generate_batch(self, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        d = self.difficulty
        text_ratio = 0.3  # 30% of every batch is text from day one

        text_count = max(1, int(batch_size * text_ratio))
        math_count = batch_size - text_count

        inputs_list = []
        targets_list = []

        for _ in range(text_count):
            text = _TEXTS[self.text_index % len(_TEXTS)]
            self.text_index += 1
            seq_len = max(4, min(len(text), int(8 + d * 24)))
            seq = _text_to_sequence(text, seq_len)
            inp = seq[:-1]
            tgt = seq[1:]
            if len(inp) < 2:
                inp = np.array([0.0], dtype=np.float32)
                tgt = np.array([0.0], dtype=np.float32)
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
                x = self.rng.standard_normal(16).astype(np.float32)
                inputs_list.append(x)
                targets_list.append(np.roll(x, 1))
        else:
            for _ in range(math_count):
                x = self.rng.standard_normal(32).astype(np.float32)
                pattern = np.sin(np.linspace(0, 4 * np.pi, 32)).astype(np.float32)
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
