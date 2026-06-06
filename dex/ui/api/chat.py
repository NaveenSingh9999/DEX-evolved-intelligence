from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix='/api/chat', tags=['chat'])


class ChatMessage(BaseModel):
    message: str


@router.post('/send')
async def chat(msg: ChatMessage):
    from dex import DEX
    if DEX is None or DEX.pipeline.best_net is None:
        return {'response': 'DEX is still evolving...', 'confidence': 0}

    import numpy as np
    net = DEX.pipeline.best_net
    x = np.array([hash(msg.message) % 1000 / 1000.0], dtype=np.float32)
    pred = net.forward(x)
    confidence = float(1.0 / (1.0 + abs(pred[0])))

    top_neurons = sorted(
        enumerate(net.genome.activations),
        key=lambda t: abs(hash(t[1])) % 100,
        reverse=True
    )[:3]

    return {
        'response': f'*activates {", ".join(a for _, a in top_neurons)}* → {pred[0]:.4f}',
        'confidence': round(confidence, 3),
        'neurons_fired': [a for _, a in top_neurons],
    }
