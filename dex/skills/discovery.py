import numpy as np
from collections import defaultdict
from dex.core.dag_network import DAGNetwork

MIN_INPUTS_FOR_SKILL = 500
MIN_SESSIONS_FOR_SKILL = 3
CONFIDENCE_THRESHOLD = 0.6
PROBE_CONSISTENCY_THRESHOLD = 0.4


class Skill:
    def __init__(self, name: str, neuron_ids: list[int], strength: float = 0.0):
        self.name = name
        self.neuron_ids = neuron_ids
        self.strength = strength
        self.age = 0
        self.accuracy = 0.0
        self.input_count = 0
        self.sessions_seen = set()
        self.confidence = 0.0
        self.state = 'dormant'
        self.probe_consistency = 0.0
        self.last_probe_pass = False

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'neuron_count': len(self.neuron_ids),
            'strength': round(self.strength, 3),
            'age': self.age,
            'accuracy': round(self.accuracy, 3),
            'input_count': self.input_count,
            'sessions': len(self.sessions_seen),
            'confidence': round(self.confidence, 3),
            'state': self.state,
            'probe_consistency': round(self.probe_consistency, 3),
        }


class SkillDiscoverer:
    def __init__(self):
        self.skills: dict[str, Skill] = {}
        self.activation_history: list[list[float]] = []
        self.emergence_counter = 0
        self.session_counter = 0
        self.current_session_inputs: dict[str, int] = defaultdict(int)
        self._probe_buffer: dict[str, list[np.ndarray]] = defaultdict(list)

    def observe(self, net: DAGNetwork, input_data: np.ndarray) -> dict:
        g = net.genome
        n = g.neuron_count
        adj = g.adjacency
        state = np.zeros(n, dtype=np.float32)
        input_dim = input_data.shape[-1]
        state[:input_dim] = input_data.ravel()[:n]

        for _ in range(n):
            new_state = state.copy()
            for i in range(n):
                incoming = adj[:, i] * state
                new_state[i] = float(np.sum(incoming))
            state = new_state

        self.activation_history.append(state.tolist())
        if len(self.activation_history) > 1000:
            self.activation_history.pop(0)

        discovered = self._detect_skills(n)

        for name in discovered:
            self.current_session_inputs[name] += 1
            self._probe_buffer[name].append(state.copy())

        return discovered

    def end_session(self):
        self.session_counter += 1
        for name, skill in self.skills.items():
            count = self.current_session_inputs.get(name, 0)
            skill.input_count += count
            if count >= MIN_INPUTS_FOR_SKILL:
                skill.sessions_seen.add(self.session_counter)
            self._run_behavioral_probe(name, skill)
            self._update_confidence(skill)
        self.current_session_inputs.clear()

    def _run_behavioral_probe(self, name: str, skill: Skill):
        """Test if a skill cluster actually produces consistent output.
        Feed probe inputs through the skill's neurons and measure activation consistency."""
        if name not in self._probe_buffer or len(self._probe_buffer[name]) < 10:
            skill.last_probe_pass = False
            return

        traces = self._probe_buffer[name][-50:]
        neuron_ids = skill.neuron_ids[:min(len(skill.neuron_ids), len(traces[0]))]

        if len(neuron_ids) < 2:
            skill.last_probe_pass = False
            return

        patterns = np.array([t[neuron_ids] for t in traces if len(t) > max(neuron_ids)])
        if patterns.shape[0] < 5:
            skill.last_probe_pass = False
            return

        corr = np.corrcoef(patterns.T)
        corr = np.nan_to_num(corr)
        if corr.shape[0] < 2:
            skill.last_probe_pass = False
            return

        consistency = float(np.mean(np.abs(corr[np.triu_indices_from(corr, k=1)])))
        skill.probe_consistency = consistency
        skill.last_probe_pass = consistency >= PROBE_CONSISTENCY_THRESHOLD

    def _update_confidence(self, skill: Skill):
        input_ok = skill.input_count >= MIN_INPUTS_FOR_SKILL
        session_ok = len(skill.sessions_seen) >= MIN_SESSIONS_FOR_SKILL
        strength_factor = min(1.0, skill.strength * 2)
        probe_factor = 0.3 if skill.last_probe_pass else -0.2
        skill.confidence = float(
            strength_factor * (0.3 if input_ok else 0.05)
            + (0.2 if session_ok else 0.05)
            + probe_factor
        )
        skill.confidence = round(max(0.0, min(1.0, skill.confidence)), 3)

        if (skill.confidence >= CONFIDENCE_THRESHOLD and input_ok and session_ok
                and skill.last_probe_pass and skill.probe_consistency >= PROBE_CONSISTENCY_THRESHOLD):
            skill.state = 'discovered'
        elif skill.input_count > 0:
            skill.state = 'emerging'
        else:
            skill.state = 'dormant'

    def _detect_skills(self, n: int) -> dict:
        if len(self.activation_history) < 50:
            return {}

        acts = np.array(self.activation_history[-50:])
        co_occurrence = np.corrcoef(acts.T)
        co_occurrence = np.nan_to_num(co_occurrence)

        clusters = self._cluster_neurons(co_occurrence, n)
        discovered = {}

        for cid, neurons in clusters.items():
            name = self._name_cluster(neurons, n)
            if name not in self.skills:
                self.skills[name] = Skill(name, neurons)
                self.emergence_counter += 1
            skill = self.skills[name]
            skill.neuron_ids = neurons
            skill.age += 1
            skill.strength = float(np.mean(np.abs(co_occurrence[neurons[0], neurons])))
            discovered[name] = skill

        return discovered

    def _cluster_neurons(self, corr: np.ndarray, n: int, threshold: float = 0.3) -> dict:
        visited = set()
        clusters = {}
        cid = 0
        for i in range(n):
            if i in visited:
                continue
            group = [i]
            visited.add(i)
            for j in range(i + 1, n):
                if j not in visited and abs(corr[i, j]) > threshold:
                    group.append(j)
                    visited.add(j)
            if len(group) >= 2:
                clusters[cid] = group
                cid += 1
        return clusters

    def _name_cluster(self, neurons: list[int], total_n: int) -> str:
        region = neurons[0] / max(total_n, 1)
        if region < 0.25:
            return f'pattern_{self.emergence_counter}'
        elif region < 0.5:
            return f'sequence_{self.emergence_counter}'
        elif region < 0.75:
            return f'abstract_{self.emergence_counter}'
        else:
            return f'output_{self.emergence_counter}'
