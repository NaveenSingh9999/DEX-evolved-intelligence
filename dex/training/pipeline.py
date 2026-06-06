import numpy as np
import time
import os
import json
from dex.core.dag_network import DAGNetwork
from dex.core.memory import EpisodicMemory, MemoryEntry
from dex.evolution.genetic import GeneticEvolver
from dex.evolution.fitness import evaluate_fitness, evaluate_diversity
from dex.evolution.curriculum import Curriculum


class TrainingPipeline:
    def __init__(self, evolver: GeneticEvolver, curriculum: Curriculum, memory: EpisodicMemory):
        self.evolver = evolver
        self.curriculum = curriculum
        self.memory = memory
        self.best_net: DAGNetwork | None = None
        self.best_fitness = -float('inf')
        self.total_steps = 0
        self.error_log: list[float] = []
        self.fitness_log: list[float] = []
        self.diversity_coeff = 0.1
        self.last_replay_gen = -5

    def step(self, iterations: int = 10):
        for _ in range(iterations):
            inputs, targets = self.curriculum.generate_batch(32)

            # ── 1. Hebbian learning phase ──
            for net in self.evolver.population:
                net.hebbian_learn(inputs, self.evolver.rng)

            # ── 2. Generational memory replay ──
            gen = self.evolver.generation
            if gen - self.last_replay_gen >= 5 and len(self.memory.buffer) >= 16:
                self.last_replay_gen = gen
                replay_memories = self.memory.sample(8, self.evolver.rng)
                replay_inputs_list = []
                replay_targets_list = []
                for m in replay_memories:
                    noisy = np.array([m.prediction], dtype=np.float32) + self.evolver.rng.standard_normal(1) * 0.05
                    replay_inputs_list.append(noisy)
                    replay_targets_list.append(np.array([m.surprise], dtype=np.float32))
                if len(replay_inputs_list) >= 4:
                    inputs = list(inputs) + replay_inputs_list
                    targets = list(targets) + replay_targets_list

            # ── 3. Evaluate fitness ──
            raw_fitnesses = []
            for net in self.evolver.population:
                fit = evaluate_fitness(net, inputs, targets)
                raw_fitnesses.append(fit)

            div_scores = [evaluate_diversity(n, self.evolver.population) for n in self.evolver.population]
            fitnesses = [r + self.diversity_coeff * d for r, d in zip(raw_fitnesses, div_scores)]

            for i, net in enumerate(self.evolver.population):
                net.genome.fitness = fitnesses[i]

            # ── 4. Memory push with curiosity ──
            for net in self.evolver.population:
                first_inp = inputs[0] if isinstance(inputs, list) else inputs
                first_tgt = targets[0] if isinstance(targets, list) else targets
                pred = net.forward(first_inp)
                tgt_slice = first_tgt[:len(pred)] if isinstance(first_tgt, np.ndarray) else np.array([0.0])
                err = float(np.mean((pred - tgt_slice) ** 2))
                surprise = 1.0 / (1.0 + err)

                inp_arr = first_inp
                h = int(hash(inp_arr.tobytes())) % (2**31 - 1)
                curiosity = self.curriculum.curiosity_score(h)
                self.curriculum.record_visit(h)

                entry = MemoryEntry(
                    input_hash=h,
                    prediction=float(pred[0]) if len(pred) > 0 else 0.0,
                    error=err,
                    surprise=surprise * curiosity,
                    timestamp=self.total_steps,
                )
                self.memory.push(entry)

            avg_fit = float(np.mean(fitnesses))
            best_idx = int(np.argmax(fitnesses))
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_net = self.evolver.population[best_idx]

            avg_err = float(np.mean([1.0 / (1.0 + f) for f in fitnesses]))
            self.error_log.append(avg_err)
            self.fitness_log.append(avg_fit)
            self.curriculum.update_difficulty(avg_err)

            self.evolver.evolve(fitnesses)
            self.total_steps += 1

            if self.total_steps % 20 == 0 and self.best_net:
                self.best_net.prune_dead_neurons(self.evolver.rng)

    def get_state(self) -> dict:
        g = self.best_net.genome if self.best_net else None
        return {
            'generation': self.evolver.generation,
            'total_steps': self.total_steps,
            'population_size': len(self.evolver.population),
            'best_fitness': round(self.best_fitness, 4),
            'curriculum_difficulty': round(self.curriculum.difficulty, 3),
            'neuron_count': g.neuron_count if g else 0,
            'innovations': len(set(g.innovations)) if g else 0,
            'activations': g.activations[:10] if g else [],
            'last_error': round(self.error_log[-1], 6) if self.error_log else 0,
            'memory_size': len(self.memory.buffer),
        }

    def save_checkpoint(self, path: str):
        tmp_path = path + '.tmp'
        state = {
            'total_steps': self.total_steps,
            'best_fitness': self.best_fitness,
            'last_replay_gen': self.last_replay_gen,
            'error_log': self.error_log[-500:],
            'fitness_log': self.fitness_log[-500:],
        }
        if self.best_net:
            state['best_genome'] = self.best_net.genome.to_dict()
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        with open(tmp_path, 'w') as f:
            json.dump(state, f)
        os.replace(tmp_path, path)

    def load_checkpoint(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path) as f:
            state = json.load(f)
        self.total_steps = state['total_steps']
        self.best_fitness = state['best_fitness']
        self.last_replay_gen = state.get('last_replay_gen', -5)
        self.error_log = state.get('error_log', [])
        self.fitness_log = state.get('fitness_log', [])
        if 'best_genome' in state:
            from dex.core.genome import Genome
            g = Genome.from_dict(state['best_genome'])
            self.best_net = DAGNetwork(g)
        return True
