import numpy as np
from .genome import Genome, next_innovation_id, MAX_NEURONS, MIN_NEURONS, MAX_EDGES_PER_NODE


class DAGNetwork:
    def __init__(self, genome: Genome):
        self.genome = genome
        self.n = genome.neuron_count
        self.activation_trace: list[np.ndarray] = []

    def _update_state(self, state: np.ndarray, input_clamp: np.ndarray | None = None) -> np.ndarray:
        g = self.genome
        from .activations import apply
        total = g.adjacency.T @ state + g.biases
        total = np.nan_to_num(total, nan=0.0, posinf=3.0, neginf=-3.0)
        for i in range(g.neuron_count):
            state[i] = apply(g.activations[i], float(total[i]))
        if input_clamp is not None:
            state[:len(input_clamp)] = input_clamp
        return state

    def _run_to_equilibrium(self, x: np.ndarray) -> np.ndarray:
        g = self.genome
        n = g.neuron_count
        state = np.zeros(n, dtype=np.float32)
        inp_clamp = x.ravel()[:n].copy()
        state[:len(inp_clamp)] = inp_clamp
        for _ in range(min(20, n)):
            state = self._update_state(state.copy(), input_clamp=inp_clamp)
        return state

    def forward(self, x: np.ndarray) -> np.ndarray:
        state = self._run_to_equilibrium(x)
        return state[-1:] if len(state) > 0 else np.array([0.0])

    def forward_with_trace(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = self._run_to_equilibrium(x)
        self.activation_trace.append(state.copy())
        if len(self.activation_trace) > 2000:
            self.activation_trace.pop(0)
        return state[-1:], state

    def hebbian_learn(self, inputs: list[np.ndarray], rng: np.random.Generator, lr: float | None = None):
        g = self.genome
        n = g.neuron_count
        eta = lr if lr is not None else g.learning_rate

        for x in inputs[:4]:
            self.forward_with_trace(x)

        if len(self.activation_trace) < 2:
            return
        acts = np.array(self.activation_trace[-4:])
        if acts.ndim != 2 or acts.shape[0] < 2:
            return

        acts = np.nan_to_num(acts, nan=0.0)
        a = np.mean(acts, axis=0)
        if np.all(np.abs(a) < 1e-8):
            return
        outer_aa = np.outer(a, a)
        decay = g.adjacency * np.outer(a ** 2, np.ones(n))
        delta = eta * (outer_aa - decay)
        delta = np.nan_to_num(delta, nan=0.0, posinf=0.1, neginf=-0.1)
        np.fill_diagonal(delta, 0)
        g.adjacency += delta.astype(np.float32)
        np.clip(g.adjacency, -3.0, 3.0, out=g.adjacency)
        g.adjacency = np.nan_to_num(g.adjacency, nan=0.0)

    def mutate(self, rng: np.random.Generator) -> 'DAGNetwork':
        new_g = Genome(
            neuron_count=self.genome.neuron_count,
            adjacency=self.genome.adjacency.copy(),
            activations=self.genome.activations.copy(),
            innovations=self.genome.innovations.copy(),
            biases=self.genome.biases.copy(),
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

        new_g.biases += rng.standard_normal(new_g.neuron_count).astype(np.float32) * mr * 0.05

        if rng.random() < mr * 0.2 and new_g.neuron_count < MAX_NEURONS:
            self._grow_neuron(new_g, rng)
        if rng.random() < mr * 0.15 and new_g.neuron_count > MIN_NEURONS:
            self._prune_neuron(new_g, rng, force=False)

        new_g.adjacency = np.nan_to_num(new_g.adjacency, nan=0.0)
        new_g.biases = np.nan_to_num(new_g.biases, nan=0.0)

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
        g.biases = np.append(g.biases, np.float32(0.0))
        g.neuron_count = new_n

    @staticmethod
    def _prune_neuron(g: Genome, rng: np.random.Generator, force: bool = False, idx: int | None = None):
        if idx is None:
            idx = rng.integers(0, g.neuron_count)
        g.adjacency = np.delete(np.delete(g.adjacency, idx, axis=0), idx, axis=1)
        g.activations.pop(idx)
        g.innovations.pop(idx)
        g.biases = np.delete(g.biases, idx)
        g.neuron_count -= 1
