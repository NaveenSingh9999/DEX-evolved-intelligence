# ⚡ DEX — Autonomous Self-Evolving Neural Network

> *He never stops growing. No API keys. No teachers. Just pure machine intelligence, evolving from scratch.*

DEX is a digital organism — a neural network that **teaches itself** through compression, prediction, and evolution. He starts as a newborn with 20 random neurons, then autonomously:

- **Grows his own brain** — adds/prunes neurons, rewires connections, evolves activation functions
- **Teaches himself** — generates his own curriculum (text prediction + math reasoning), driven by curiosity
- **Learns within each lifetime** — Hebbian/STDP plasticity before every evaluation, so evolution selects for *learnability*
- **Remembers across generations** — surprising memories are replayed to prevent knowledge loss
- **Discovers real skills** — co-activation clusters only count if they pass behavioral probes
- **Never stops** — evolution runs 24/7, checkpointed and resource-governed

Built from **zero external AI dependencies** — pure NumPy, no PyTorch, no TensorFlow, no API calls.

---

## How He Works

```
Every training cycle:
  1. Curriculum → generates mixed text + math batch
  2. Hebbian learning → each genome learns from the batch (Oja's rule)
  3. Memory replay → top surprising memories are replayed cross-generation
  4. Fitness eval → accuracy + diversity + novelty + complexity penalty
  5. NEAT crossover → innovation-matched breeding
  6. Mutation → weight noise, topology changes, activation swaps
  7. Pruning → dead neurons (activation < 0.01) are killed
  8. Repeat → forever
```

---

## Quick Start

```bash
pip install numpy fastapi uvicorn pydantic
python dex.py
# → Opens dashboard at http://localhost:3000
```

---

## The Architecture

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Neural Engine | NumPy + Numba-ready | DAG network, 9 activations, NEAT innovation tracking |
| Evolution | Custom GA | Tournament selection, NEAT crossover, Dirichlet weights |
| Learning | Hebbian/Oja's rule | Within-lifetime plasticity before fitness eval |
| Memory | Episodic replay buffer | Surprise-prioritized replay across generations |
| Skills | Correlation clustering + behavioral probes | Only real functional clusters are "discovered" |
| Curriculum | Auto-scaling text + math | 30% char-prediction + 70% math from day one |
| Governor | psutil | Caps 60% CPU / 2GB RAM, auto-throttles |

---

## Project Structure

```
dex/
├── core/           # DAG network, genome, activations, memory
├── evolution/      # Genetic algorithm, fitness, curriculum
├── training/       # Pipeline (hebbian, replay, eval, evolve)
├── skills/         # Skill discovery with behavioral probes
├── ui/             # FastAPI backend + Vue/D3.js dashboard
└── main.py         # Entry point with resource governor
```

---

## License

MIT
