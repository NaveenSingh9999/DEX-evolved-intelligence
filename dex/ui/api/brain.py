import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix='/api/brain', tags=['brain'])


class BrainState(BaseModel):
    neuron_count: int
    activations: list[str]
    adjacency: list[list[float]]
    fitness: float
    generation: int
    learning_rate: float


@router.get('/state')
async def get_brain_state():
    from dex import DEX
    if DEX is None:
        return {'error': 'DEX not initialized'}
    net = DEX.pipeline.best_net
    if net is None:
        return {'neuron_count': 0, 'activations': [], 'adjacency': [], 'fitness': 0, 'generation': 0, 'learning_rate': 0}
    g = net.genome
    adj = np.nan_to_num(g.adjacency, nan=0.0, posinf=3.0, neginf=-3.0)
    return {
        'neuron_count': g.neuron_count,
        'activations': g.activations,
        'adjacency': adj.tolist(),
        'fitness': 0.0 if np.isnan(g.fitness) else round(g.fitness, 4),
        'generation': DEX.pipeline.evolver.generation,
        'learning_rate': 0.0 if np.isnan(g.learning_rate) else round(g.learning_rate, 6),
    }
