import math
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from dex.evolution.curriculum import _CHAR_MAP

router = APIRouter(prefix='/api/chat', tags=['chat'])

_REV_MAP = {v: k for k, v in _CHAR_MAP.items()}
_VOCAB = list(_CHAR_MAP.keys())


class ChatMessage(BaseModel):
    message: str


def _text_to_vec(text: str, size: int = 4) -> np.ndarray:
    chars = [c.lower() for c in text if c.lower() in _CHAR_MAP]
    if not chars:
        chars = ['a']
    vec = np.array([_CHAR_MAP.get(c, 0.0) for c in chars[:size]], dtype=np.float32)
    if len(vec) < size:
        vec = np.pad(vec, (0, size - len(vec)))
    return vec


def _vec_to_char(v: float) -> str:
    closest = min(_REV_MAP.keys(), key=lambda k: abs(k - v))
    return _REV_MAP.get(closest, '?')


@router.post('/send')
async def chat(msg: ChatMessage):
    from dex import DEX
    if DEX is None or DEX.pipeline.best_net is None:
        return {'response': 'DEX is still evolving...', 'confidence': 0}

    net = DEX.pipeline.best_net
    input_vec = _text_to_vec(msg.message, size=4)

    generated = []
    x = input_vec.copy()
    for _ in range(min(len(msg.message) + 10, 30)):
        pred = net.forward(x)
        next_char = _vec_to_char(float(pred[0]))
        generated.append(next_char)
        x = np.roll(x, -1)
        x[-1] = float(pred[0])

    response = ''.join(generated).strip()
    if not response:
        response = '...'

    loss = float(np.mean((pred - input_vec[:1]) ** 2))
    confidence = round(1.0 / (1.0 + loss), 3)

    return {
        'response': response,
        'confidence': confidence,
        'neurons': net.genome.neuron_count,
    }
