import numpy as np

ACTIVATIONS = {
    'relu':   lambda x: np.maximum(0, x),
    'tanh':   lambda x: np.tanh(x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -100, 100))),
    'swish':  lambda x: x * (1 / (1 + np.exp(-np.clip(x, -100, 100)))),
    'gelu':   lambda x: x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
    'sin':    lambda x: np.sin(x),
    'abs':    lambda x: np.abs(x),
    'square': lambda x: x**2,
    'identity': lambda x: x,
}

ACTIVATION_NAMES = list(ACTIVATIONS.keys())


def apply(act_name: str, x: np.ndarray) -> np.ndarray:
    return ACTIVATIONS.get(act_name, ACTIVATIONS['relu'])(x)


def random_activation(rng: np.random.Generator) -> str:
    return rng.choice(ACTIVATION_NAMES)
