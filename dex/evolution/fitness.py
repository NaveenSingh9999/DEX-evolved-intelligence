import numpy as np
from dex.core.dag_network import DAGNetwork


def evaluate_fitness(net: DAGNetwork, inputs: list[np.ndarray], targets: list[np.ndarray]) -> float:
    total_error = 0.0
    n = max(len(inputs), 1)
    for x, y in zip(inputs, targets):
        pred = net.forward(x)
        err = float(np.mean((pred - y) ** 2))
        total_error += err

    mse = total_error / n
    accuracy = 1.0 / (1.0 + mse)
    complexity_penalty = 0.0001 * net.genome.neuron_count
    age_bonus = min(0.1, net.genome.age * 0.001)
    novelty = _novelty_bonus(net)

    weights = net.genome.dirichlet_weights
    w_sum = weights.sum()
    if w_sum > 0:
        weights = weights / w_sum

    fitness = (
        weights[0] * accuracy
        - weights[1] * complexity_penalty
        + weights[2] * age_bonus
        + weights[3] * novelty
    )
    return float(fitness)


def evaluate_diversity(net: DAGNetwork, all_nets: list[DAGNetwork]) -> float:
    similarity_sum = 0.0
    my_innovs = set(net.genome.innovations)
    for other in all_nets:
        other_innovs = set(other.genome.innovations)
        if not my_innovs and not other_innovs:
            continue
        jaccard = len(my_innovs & other_innovs) / len(my_innovs | other_innovs)
        similarity_sum += jaccard
    avg_sim = similarity_sum / max(len(all_nets), 1)
    return 1.0 - avg_sim


def _novelty_bonus(net: DAGNetwork) -> float:
    act_set = set(net.genome.activations)
    variety = len(act_set) / 9.0
    return variety * 0.05
