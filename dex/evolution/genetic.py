import numpy as np
from dex.core.genome import Genome, next_innovation_id
from dex.core.dag_network import DAGNetwork


class GeneticEvolver:
    def __init__(self, pop_size: int = 20, elite_frac: float = 0.2, rng: np.random.Generator = None):
        self.pop_size = pop_size
        self.elite_count = max(1, int(pop_size * elite_frac))
        self.rng = rng or np.random.default_rng()
        self.population: list[DAGNetwork] = []
        self.generation = 0

    def initialize(self):
        from dex.core.genome import bootstrap_genome
        self.population = []
        for _ in range(self.pop_size):
            g = bootstrap_genome(self.rng)
            self.population.append(DAGNetwork(g))

    def evolve(self, fitness_scores: list[float]) -> list[DAGNetwork]:
        idxs = np.argsort(fitness_scores)[::-1]
        ranked = [self.population[i] for i in idxs]

        elite = ranked[:self.elite_count]
        next_gen = list(elite)

        while len(next_gen) < self.pop_size:
            p1 = self._tournament_select(ranked)
            p2 = self._tournament_select(ranked)
            child = self._crossover(p1, p2)
            child = child.mutate(self.rng)
            next_gen.append(child)

        self.population = next_gen[:self.pop_size]
        self.generation += 1
        return self.population

    def _tournament_select(self, ranked: list, k: int = 3) -> DAGNetwork:
        idxs = self.rng.integers(0, len(ranked), size=k)
        best = min(idxs)
        return ranked[best]

    def _crossover(self, p1: DAGNetwork, p2: DAGNetwork) -> DAGNetwork:
        """NEAT-style crossover using innovation numbers.
        Only matching innovations are swapped; disjoint/excess genes
        come from the fitter parent."""
        g1, g2 = p1.genome, p2.genome
        fitter, weaker = (g1, g2) if g1.fitness >= g2.fitness else (g2, g1)

        map1 = g1.innovation_map()
        map2 = g2.innovation_map()
        all_innovs = set(g1.innovations) | set(g2.innovations)

        child_innovs = []
        child_acts = []

        for innov in sorted(all_innovs):
            in1 = innov in map1
            in2 = innov in map2
            if in1 and in2:
                idx1, idx2 = map1[innov], map2[innov]
                if self.rng.random() < 0.5:
                    child_innovs.append(g1.innovations[idx1])
                    child_acts.append(g1.activations[idx1])
                else:
                    child_innovs.append(g2.innovations[idx2])
                    child_acts.append(g2.activations[idx2])
            elif in1:
                child_innovs.append(g1.innovations[map1[innov]])
                child_acts.append(g1.activations[map1[innov]])
            else:
                child_innovs.append(g2.innovations[map2[innov]])
                child_acts.append(g2.activations[map2[innov]])

        child_n = len(child_innovs)
        child_adj = np.zeros((child_n, child_n), dtype=np.float32)
        for i, innov_i in enumerate(child_innovs):
            for j, innov_j in enumerate(child_innovs):
                w = None
                if innov_i in map1 and innov_j in map1:
                    w = g1.adjacency[map1[innov_i], map1[innov_j]]
                if innov_i in map2 and innov_j in map2:
                    w2 = g2.adjacency[map2[innov_i], map2[innov_j]]
                    w = w2 if w is None else (w + w2) / 2
                if w is not None and abs(w) > 1e-8:
                    child_adj[i, j] = w

        child_g = Genome(
            neuron_count=child_n,
            adjacency=child_adj,
            activations=child_acts,
            innovations=child_innovs,
            learning_rate=(g1.learning_rate + g2.learning_rate) / 2,
            mutation_rate=(g1.mutation_rate + g2.mutation_rate) / 2,
            dirichlet_weights=(g1.dirichlet_weights + g2.dirichlet_weights) / 2,
        )
        return DAGNetwork(child_g)
