import argparse
import threading
import time
import os
import sys
import json
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dex.core.genome import reset_innovation_counter
from dex.core.memory import EpisodicMemory
from dex.evolution.genetic import GeneticEvolver
from dex.evolution.curriculum import Curriculum
from dex.training.pipeline import TrainingPipeline
from dex.skills.discovery import SkillDiscoverer
from dex.skills.logger import DailyLogger

# ── Resource Governor ─────────────────────────────────────────────
CPU_CAP = 0.6
RAM_CAP_MB = 2048

_have_psutil = False
try:
    import psutil
    _have_psutil = True
except ImportError:
    pass


def _throttle_if_needed():
    if not _have_psutil:
        return
    proc = psutil.Process(os.getpid())
    try:
        cpu = proc.cpu_percent(interval=0) / 100.0
        mem = proc.memory_info().rss / (1024 * 1024)
        if cpu > CPU_CAP or mem > RAM_CAP_MB:
            factor = max(0.1, min(cpu / CPU_CAP, mem / RAM_CAP_MB))
            sleep_time = (factor - 1.0) * 2.0
            if sleep_time > 0:
                time.sleep(min(sleep_time, 5.0))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


# ── Checkpoint paths ──────────────────────────────────────────────
CHECKPOINT_DIR = 'data/checkpoints'
GENOME_PATH = os.path.join(CHECKPOINT_DIR, 'best_genome.json')
PIPELINE_PATH = os.path.join(CHECKPOINT_DIR, 'pipeline_state.json')
MEMORY_PATH = os.path.join(CHECKPOINT_DIR, 'memory.json')


class DEX:
    def __init__(self, pop_size: int = 20, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.memory = EpisodicMemory(capacity=10000)
        self.evolver = GeneticEvolver(pop_size=pop_size, rng=self.rng)
        self.curriculum = Curriculum(rng=self.rng)
        self.pipeline = TrainingPipeline(self.evolver, self.curriculum, self.memory)
        self.skill_discoverer = SkillDiscoverer()
        self.logger = DailyLogger()
        self.running = False
        self._thread = None
        self._save_counter = 0

    def initialize(self):
        reset_innovation_counter()
        if self.pipeline.load_checkpoint(PIPELINE_PATH):
            print('  Restored from checkpoint')
            print(f'  Resuming at step {self.pipeline.total_steps}')
            self.evolver.generation = self.pipeline.total_steps
            self.evolver.population = []
            for _ in range(self.evolver.pop_size):
                from dex.core.genome import bootstrap_genome
                g = bootstrap_genome(self.rng)
                self.evolver.population.append(type(self.pipeline.best_net)(g))
            return

        print(' Spawning initial population (bootstrapped 3-layer MLPs)...')
        self.evolver.initialize()
        print(f'   Population: {self.evolver.pop_size} genomes')
        print(' DEX is alive. Evolution begins.')

    def train_step(self):
        _throttle_if_needed()
        self.pipeline.step(iterations=5)
        state = self.pipeline.get_state()

        if self.pipeline.best_net:
            _, inputs = self.curriculum.generate_batch(1)
            inp = inputs[0] if isinstance(inputs, list) else inputs
            self.skill_discoverer.observe(self.pipeline.best_net, inp)

        self.logger.log_step(state, self.skill_discoverer.skills)

        if state['total_steps'] % 30 == 0:
            self.skill_discoverer.end_session()

        self._save_counter += 1
        if self._save_counter >= 20:
            self._save_counter = 0
            self._checkpoint()

        if state['total_steps'] % 10 == 0:
            s = state
            discovered = len([sk for sk in self.skill_discoverer.skills.values() if sk.state == 'discovered'])
            print(f'  Gen {s["generation"]:>4} | Step {s["total_steps"]:>6} | '
                  f'Neurons {s["neuron_count"]:>3} | Innovs {s["innovations"]:>3} | '
                  f'Fitness {s["best_fitness"]:.4f} | Diff {s["curriculum_difficulty"]:.2f} | '
                  f'Skills {discovered}', flush=True)

    def _checkpoint(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self.pipeline.save_checkpoint(PIPELINE_PATH)
        if self.memory:
            self.memory.save(MEMORY_PATH)

    def train_loop(self, interval: float = 1.0):
        self.running = True
        while self.running:
            self.train_step()
            time.sleep(interval)

    def start_background(self, interval: float = 1.0):
        self._thread = threading.Thread(target=self.train_loop, args=(interval,), daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        self._checkpoint()
        if self._thread:
            self._thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description='DEX — Self-Evolving Neural Network')
    parser.add_argument('--pop-size', type=int, default=20, help='Population size')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--ui', action='store_true', default=True, help='Launch dashboard')
    parser.add_argument('--port', type=int, default=3000, help='Dashboard port')
    parser.add_argument('--interval', type=float, default=0.5, help='Seconds between training steps')
    parser.add_argument('--reset', action='store_true', help='Reset all checkpoints and start fresh')
    args = parser.parse_args()

    if args.reset:
        import shutil
        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
            print('  Checkpoints cleared.')

    global pop_size
    pop_size = args.pop_size

    dex = DEX(pop_size=args.pop_size, seed=args.seed)
    dex.initialize()

    import dex as dex_module
    dex_module.DEX = dex

    dex.start_background(interval=args.interval)

    def _handle_sig(signum, frame):
        print('\n Shutting down DEX...')
        dex.stop()
        exit(0)

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    if args.ui:
        print(f'\n Dashboard at http://localhost:{args.port}')
        from dex.ui.server import run
        try:
            run(port=args.port)
        except KeyboardInterrupt:
            pass
    else:
        print('\n Training in background. Ctrl+C to stop.')
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            pass

    dex.stop()
    print(' DEX evolution paused. He\'ll remember everything.')


pop_size = 20

if __name__ == '__main__':
    main()
