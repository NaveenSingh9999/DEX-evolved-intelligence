import json
import os
from datetime import datetime


class DailyLogger:
    def __init__(self, log_dir: str = 'data/logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.daily: dict = self._load_today()

    def _load_today(self) -> dict:
        path = self._today_path()
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {
            'date': datetime.now().isoformat(),
            'neurons_added': 0,
            'neurons_pruned': 0,
            'new_skills': [],
            'fitness_start': 0,
            'fitness_end': 0,
            'error_avg': 0,
            'milestones': [],
        }

    def _today_path(self) -> str:
        return os.path.join(self.log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.json")

    def log_step(self, state: dict, skills: dict):
        today = self._load_today()
        today['fitness_end'] = state['best_fitness']
        today['error_avg'] = state['last_error']

        new_skill_names = [s.name for s in skills.values() if s.age <= 1]
        for name in new_skill_names:
            if name not in today['new_skills']:
                today['new_skills'].append(name)

        if today['fitness_start'] == 0:
            today['fitness_start'] = state['best_fitness']

        with open(self._today_path(), 'w') as f:
            json.dump(today, f, indent=2)

    def get_today_summary(self) -> dict:
        return self._load_today()

    def get_history(self, days: int = 7) -> list:
        files = sorted(os.listdir(self.log_dir))[-days:]
        history = []
        for fname in files:
            path = os.path.join(self.log_dir, fname)
            with open(path) as f:
                history.append(json.load(f))
        return history
