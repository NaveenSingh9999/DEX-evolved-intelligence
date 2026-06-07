import numpy as np

_sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -100, 100)))

ACTIVATIONS = {
    'relu':    lambda x: np.maximum(0, x),
    'tanh':    lambda x: np.tanh(x),
    'sigmoid': _sigmoid,
    'swish':   lambda x: x * _sigmoid(x),
    'gelu':    lambda x: x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
    'sin':     lambda x: np.sin(x),
    'abs':     lambda x: np.abs(x),
    'square':  lambda x: x**2,
    'identity': lambda x: x,
}

ACTIVATION_NAMES = list(ACTIVATIONS.keys())

# Derivatives for backprop
_DERIVATIVES = {
    'relu':    lambda x: np.where(x > 0, 1.0, 0.0),
    'tanh':    lambda x: 1 - np.tanh(x)**2,
    'sigmoid': lambda x: (s := _sigmoid(x)) * (1 - s),
    'swish':   lambda x: (s := _sigmoid(x)) * (1 + x * (1 - s)),
    'gelu':    lambda x: 0.5 * (1 + np.tanh(t := np.sqrt(2/np.pi) * (x + 0.044715*x**3))) + (0.5 * x * (1 - np.tanh(t)**2) * np.sqrt(2/np.pi) * (1 + 0.134145*x**2)),
    'sin':     lambda x: np.cos(x),
    'abs':     lambda x: np.where(x >= 0, 1.0, -1.0),
    'square':  lambda x: 2 * x,
    'identity': lambda x: np.ones_like(x),
}


def apply(act_name: str, x: np.ndarray) -> np.ndarray:
    return ACTIVATIONS.get(act_name, ACTIVATIONS['relu'])(x)


def derivative(act_name: str, x: np.ndarray) -> np.ndarray:
    return _DERIVATIVES.get(act_name, _DERIVATIVES['relu'])(x)


def random_activation(rng: np.random.Generator) -> str:
    return rng.choice(ACTIVATION_NAMES)
