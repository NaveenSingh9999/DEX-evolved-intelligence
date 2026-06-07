import numpy as np
from .genome import Genome, next_innovation_id, MAX_NEURONS, MIN_NEURONS


class DAGNetwork:
    def __init__(self, genome: Genome):
        self.genome = genome
        self.n = genome.neuron_count
        self.activation_trace: list[np.ndarray] = []
        self._cache = {}  # stores intermediates for backprop

    def _update_state(self, state: np.ndarray, input_clamp: np.ndarray | None = None,
                      store: bool = False) -> np.ndarray:
        g = self.genome
        from .activations import apply
        total = g.adjacency.T @ state + g.biases
        total = np.nan_to_num(total, nan=0.0, posinf=3.0, neginf=-3.0)
        for i in range(g.neuron_count):
            state[i] = apply(g.activations[i], float(total[i]))
        if input_clamp is not None:
            state[:len(input_clamp)] = input_clamp
        if store:
            self._cache.setdefault('totals', []).append(total.copy())
            self._cache.setdefault('states', []).append(state.copy())
        return state

    def _run_forward(self, x: np.ndarray, store: bool = False) -> np.ndarray:
        g = self.genome
        n = g.neuron_count
        state = np.zeros(n, dtype=np.float32)
        inp_clamp = x.ravel()[:n].copy()
        state[:len(inp_clamp)] = inp_clamp
        steps = min(8, n)
        if store:
            self._cache = {'steps': steps, 'totals': [], 'states': [state.copy()], 'input_dim': x.shape[-1]}
        for _ in range(steps):
            state = self._update_state(state.copy(), input_clamp=inp_clamp, store=store)
        return state

    def forward(self, x: np.ndarray) -> np.ndarray:
        state = self._run_forward(x, store=False)
        return state[-1:] if len(state) > 0 else np.array([0.0])

    def forward_with_trace(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = self._run_forward(x, store=False)
        self.activation_trace.append(state.copy())
        if len(self.activation_trace) > 2000:
            self.activation_trace.pop(0)
        return state[-1:], state

    def forward_and_cache(self, x: np.ndarray) -> np.ndarray:
        """Forward pass that stores all intermediates for backward()."""
        state = self._run_forward(x, store=True)
        return state[-1:] if len(state) > 0 else np.array([0.0])

    def backward(self, target: np.ndarray, lr: float | None = None) -> float:
        """Backprop through time with gradient clipping. Returns loss before update."""
        g = self.genome
        from .activations import derivative
        cache = self._cache
        if not cache or 'states' not in cache or len(cache['states']) < 2:
            return 0.0

        eta = lr if lr is not None else g.learning_rate
        states = cache['states']
        totals = cache['totals']
        steps = cache['steps']
        n = g.neuron_count
        output_idx = n - 1
        n_inputs = min(cache.get('input_dim', 4), n)

        final_state = states[-1]
        if np.any(np.isnan(final_state)) or np.any(np.isinf(final_state)):
            return 1.0

        pred = np.array([final_state[output_idx]], dtype=np.float32)
        loss = float(np.mean((pred - target[:1]) ** 2))

        d_out = 2 * (pred - target[:1]) / 1.0
        d_state = np.zeros(n, dtype=np.float32)
        d_state[output_idx] = float(d_out[0])

        adj_grad = np.zeros_like(g.adjacency, dtype=np.float32)
        bias_grad = np.zeros(n, dtype=np.float32)
        grad_norm = 0.0

        for t in reversed(range(steps)):
            total_t = totals[t]
            state_prev = states[t - 1] if t > 0 else np.zeros(n, dtype=np.float32)

            if np.any(np.isnan(total_t)) or np.any(np.isinf(total_t)):
                continue

            d_act = np.array([derivative(g.activations[i], float(total_t[i])) for i in range(n)], dtype=np.float32)
            d_act = np.nan_to_num(d_act, nan=0.0)
            d_total = d_state * d_act

            adj_grad += np.outer(state_prev, d_total)
            bias_grad += d_total
            grad_norm += float(np.sum(d_total ** 2))

            if t > 0:
                d_state = g.adjacency @ d_total
                d_state = np.nan_to_num(d_state, nan=0.0, posinf=0.0, neginf=0.0)
                d_state[:n_inputs] = 0.0

        if grad_norm > 100.0:
            scale = 10.0 / float(np.sqrt(grad_norm))
            adj_grad *= scale
            bias_grad *= scale

        adj_grad = np.nan_to_num(adj_grad, nan=0.0)
        bias_grad = np.nan_to_num(bias_grad, nan=0.0)
        np.fill_diagonal(adj_grad, 0)

        g.adjacency -= (eta * adj_grad).astype(np.float32)
        g.biases -= (eta * bias_grad).astype(np.float32)
        np.clip(g.adjacency, -3.0, 3.0, out=g.adjacency)
        g.adjacency = np.nan_to_num(g.adjacency, nan=0.0)
        g.biases = np.nan_to_num(g.biases, nan=0.0)

        return loss

    # ── GA operations (architecture evolution only) ──

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

        if rng.random() < mr * 0.4:
            i, j = rng.integers(0, new_g.neuron_count, size=2)
            new_g.adjacency[i, j] = rng.standard_normal() * 0.5

        if rng.random() < mr * 0.2:
            idx = rng.integers(0, new_g.neuron_count)
            from dex.core.activations import random_activation
            new_g.activations[idx] = random_activation(rng)

        if rng.random() < mr * 0.15 and new_g.neuron_count < MAX_NEURONS:
            self._grow_neuron(new_g, rng)
        if rng.random() < mr * 0.1 and new_g.neuron_count > MIN_NEURONS:
            self._prune_neuron(new_g, rng, force=False)

        new_g.adjacency = np.nan_to_num(new_g.adjacency, nan=0.0)
        new_g.biases = np.nan_to_num(new_g.biases, nan=0.0)
        new_g.learning_rate = float(np.clip(new_g.learning_rate, 1e-5, 0.02))
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
