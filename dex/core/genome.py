import numpy as np
from dataclasses import dataclass, field
from .activations import random_activation

# NEAT-style: hard budget for safety
MAX_NEURONS = 10_000
MIN_NEURONS = 10
INIT_NEURONS = 20
MAX_EDGES_PER_NODE = 50

_global_innovation_counter = 0


def next_innovation_id() -> int:
    global _global_innovation_counter
    _global_innovation_counter += 1
    return _global_innovation_counter


def reset_innovation_counter():
    global _global_innovation_counter
    _global_innovation_counter = 0


def bootstrap_genome(rng: np.random.Generator) -> 'Genome':
    """Seed with a minimal 3-layer MLP instead of random DAG."""
    n = INIT_NEURONS
    adj = np.zeros((n, n), dtype=np.float32)
    hidden = 8
    out = 1
    # input -> hidden
    for i in range(hidden):
        for j in range(n - hidden - out):
            adj[j, i] = rng.standard_normal() * 0.5
    # hidden -> output
    for i in range(n - out, n):
        for j in range(hidden):
            adj[j, i] = rng.standard_normal() * 0.5

    # hidden -> hidden recurrent connections (dense enough for signal flow)
    for i in range(hidden):
        for j in range(hidden):
            if rng.random() < 0.3:
                adj[j, i] = rng.standard_normal() * 0.3
    acts = [rng.choice(['relu', 'tanh', 'sigmoid', 'identity', 'swish']) for _ in range(n)]
    acts[-1] = 'sigmoid'  # output [0,1], derivative always >0, matches char range
    innovs = [next_innovation_id() for _ in range(n)]
    biases = np.zeros(n, dtype=np.float32)
    biases[-1] = 0.0  # output starts neutral
    return Genome(
        neuron_count=n,
        adjacency=adj,
        activations=acts,
        innovations=innovs,
        biases=biases,
        learning_rate=float(rng.uniform(0.001, 0.02)),
        mutation_rate=float(rng.uniform(0.01, 0.3)),
    )


def random_genome(rng: np.random.Generator) -> 'Genome':
    return bootstrap_genome(rng)


@dataclass
class Genome:
    neuron_count: int = INIT_NEURONS
    adjacency: np.ndarray = field(default_factory=lambda: np.zeros((INIT_NEURONS, INIT_NEURONS), dtype=np.float32))
    activations: list = field(default_factory=lambda: ['relu'] * INIT_NEURONS)
    innovations: list = field(default_factory=lambda: [next_innovation_id() for _ in range(INIT_NEURONS)])
    biases: np.ndarray = field(default_factory=lambda: np.zeros(INIT_NEURONS, dtype=np.float32))
    learning_rate: float = 0.01
    mutation_rate: float = 0.1
    age: int = 0
    fitness: float = 0.0
    dirichlet_weights: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32))

    def __post_init__(self):
        if self.adjacency.shape != (self.neuron_count, self.neuron_count):
            self.adjacency = np.zeros((self.neuron_count, self.neuron_count), dtype=np.float32)
        if len(self.activations) != self.neuron_count:
            self.activations = ['relu'] * self.neuron_count
        if len(self.innovations) != self.neuron_count:
            self.innovations = [next_innovation_id() for _ in range(self.neuron_count)]
        if self.biases.shape != (self.neuron_count,):
            self.biases = np.zeros(self.neuron_count, dtype=np.float32)
        if self.dirichlet_weights.shape != (4,):
            self.dirichlet_weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)

    def innovation_map(self) -> dict[int, int]:
        return {innov: i for i, innov in enumerate(self.innovations)}

    def to_dict(self) -> dict:
        return {
            'neuron_count': self.neuron_count,
            'adjacency': self.adjacency.tolist(),
            'activations': self.activations,
            'innovations': self.innovations,
            'biases': self.biases.tolist(),
            'learning_rate': self.learning_rate,
            'mutation_rate': self.mutation_rate,
            'age': self.age,
            'fitness': self.fitness,
            'dirichlet_weights': self.dirichlet_weights.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Genome':
        g = cls(
            neuron_count=data['neuron_count'],
            adjacency=np.array(data['adjacency'], dtype=np.float32),
            activations=data['activations'],
            innovations=data['innovations'],
            biases=np.array(data.get('biases', [0.0]*data['neuron_count']), dtype=np.float32),
            learning_rate=data['learning_rate'],
            mutation_rate=data['mutation_rate'],
            age=data['age'],
            fitness=data['fitness'],
            dirichlet_weights=np.array(data['dirichlet_weights'], dtype=np.float32),
        )
        return g
