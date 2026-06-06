import numpy as np
from .genome import Genome, next_innovation_id, MAX_NEURONS, MIN_NEURONS, MAX_EDGES_PER_NODE


class DAGNetwork:
    def __init__(self, genome: Genome):
        self.genome = genome
        self.n = genome.neuron_count
        self.activation_trace: list[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        g = self.genome
        n = g.neuron_count
        adj = g.adjacency
        from .activations import apply

        state = np.zeros(n, dtype=np.float32)
        input_dim = x.shape[-1]
        state[:input_dim] = x.ravel()[:n]

        for _ in range(n):
            new_state = state.copy()
            for i in range(n):
                incoming = adj[:, i] * state
                total = np.sum(incoming)
                new_state[i] = apply(g.activations[i], total)
            state = new_state

        output = state[-1:] if n > 0 else np.array([0.0])
        return output

    def forward_with_trace(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g = self.genome
        n = g.neuron_count
        adj = g.adjacency
        from .activations import apply

        state = np.zeros(n, dtype=np.float32)
        input_dim = x.shape[-1]
        state[:input_dim] = x.ravel()[:n]

        for _ in range(n):
            new_state = state.copy()
            for i in range(n):
                incoming = adj[:, i] * state
                total = np.sum(incoming)
                new_state[i] = apply(g.activations[i], total)
            state = new_state

        self.activation_trace.append(state.copy())
        if len(self.activation_trace) > 2000:
            self.activation_trace.pop(0)

        return state[-1:], state

    def hebbian_learn(self, inputs: list[np.ndarray], rng: np.random.Generator, lr: float | None = None):
        """Apply Oja's rule within a genome's lifetime so evolution selects for learnability.
        Δw_ij = η * (post_i * pre_j - post_i² * w_ij)
        """
        g = self.genome
        n = g.neuron_count
        eta = lr if lr is not None else g.learning_rate * 2.0

        traces = []
        for x in inputs:
            _ = self.forward_with_trace(x)
            if len(self.activation_trace) >= 1:
                traces.append(self.activation_trace[-1].copy())

        if len(traces) < 2:
            return

        acts = np.array(traces)
        if acts.ndim != 2 or acts.shape[0] < 2:
            return

        avg_post = np.mean(acts, axis=0)
        avg_pre = np.mean(acts, axis=0)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dw = eta * (avg_post[i] * avg_pre[j] - avg_post[i] ** 2 * g.adjacency[i, j])
                g.adjacency[i, j] += float(dw)
                g.adjacency[i, j] = float(np.clip(g.adjacency[i, j], -3.0, 3.0))

    def mutate(self, rng: np.random.Generator) -> 'DAGNetwork':
        new_g = Genome(
            neuron_count=self.genome.neuron_count,
            adjacency=self.genome.adjacency.copy(),
            activations=self.genome.activations.copy(),
            innovations=self.genome.innovations.copy(),
            learning_rate=self.genome.learning_rate,
            mutation_rate=self.genome.mutation_rate,
            age=self.genome.age + 1,
            fitness=self.genome.fitness,
            dirichlet_weights=self.genome.dirichlet_weights.copy(),
        )
        mr = new_g.mutation_rate

        adj_noise = rng.standard_normal(new_g.adjacency.shape).astype(np.float32) * mr * 0.1
        new_g.adjacency += adj_noise

        if rng.random() < mr * 0.5:
            i, j = rng.integers(0, new_g.neuron_count, size=2)
            new_g.adjacency[i, j] = rng.standard_normal() * 0.5

        if rng.random() < mr * 0.3:
            idx = rng.integers(0, new_g.neuron_count)
            from dex.core.activations import random_activation
            new_g.activations[idx] = random_activation(rng)

        if rng.random() < mr * 0.2 and new_g.neuron_count < MAX_NEURONS:
            self._grow_neuron(new_g, rng)
        if rng.random() < mr * 0.15 and new_g.neuron_count > MIN_NEURONS:
            self._prune_neuron(new_g, rng, force=False)

        new_g.learning_rate *= 1 + rng.standard_normal() * mr * 0.1
        new_g.learning_rate = float(np.clip(new_g.learning_rate, 1e-5, 0.1))
        new_g.mutation_rate *= 1 + rng.standard_normal() * mr * 0.1
        new_g.mutation_rate = float(np.clip(new_g.mutation_rate, 0.001, 0.5))

        return DAGNetwork(new_g)

    def prune_dead_neurons(self, rng: np.random.Generator) -> bool:
        if len(self.activation_trace) < 100:
            return False
        recent = np.array(self.activation_trace[-100:])
        mean_acts = np.mean(np.abs(recent), axis=0)
        dead_idxs = [i for i in range(self.n) if mean_acts[i] < 0.01]
        pruned = False
        for idx in sorted(dead_idxs, reverse=True):
            if self.genome.neuron_count <= MIN_NEURONS:
                break
            self._prune_neuron(self.genome, rng, force=True, idx=idx)
            pruned = True
        return pruned

    @staticmethod
    def _grow_neuron(g: Genome, rng: np.random.Generator):
        from dex.core.activations import random_activation
        old_n = g.neuron_count
        new_n = old_n + 1
        new_adj = np.zeros((new_n, new_n), dtype=np.float32)
        new_adj[:old_n, :old_n] = g.adjacency
        conns = rng.integers(0, old_n, size=min(3, old_n))
        for c in conns:
            new_adj[c, old_n] = rng.standard_normal() * 0.1
            new_adj[old_n, c] = rng.standard_normal() * 0.1
        g.adjacency = new_adj
        g.activations.append(random_activation(rng))
        g.innovations.append(next_innovation_id())
        g.neuron_count = new_n

    @staticmethod
    def _prune_neuron(g: Genome, rng: np.random.Generator, force: bool = False, idx: int | None = None):
        if idx is None:
            idx = rng.integers(0, g.neuron_count)
        g.adjacency = np.delete(np.delete(g.adjacency, idx, axis=0), idx, axis=1)
        g.activations.pop(idx)
        g.innovations.pop(idx)
        g.neuron_count -= 1
