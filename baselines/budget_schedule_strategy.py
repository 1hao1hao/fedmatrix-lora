import numpy as np

from .base_strategy import BaseStrategy


class BudgetScheduleStrategy(BaseStrategy):
    """Deterministic non-RL budget policy that spends the budget across rounds."""

    def __init__(self, total_budget, max_rounds, per_round_cap, min_ratio=0.1, max_ratio=1.0):
        self.total_budget = float(total_budget)
        self.max_rounds = max(1, int(max_rounds))
        self.per_round_cap = max(1.0, float(per_round_cap))
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)

    def predict(self, observation, deterministic=True):
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        remaining_ratio = float(obs[3]) if obs.size > 3 else 1.0
        round_ratio = float(obs[4]) if obs.size > 4 else 0.0

        remaining_budget = max(0.0, remaining_ratio * self.total_budget)
        current_round = int(round_ratio * self.max_rounds)
        remaining_rounds = max(1, self.max_rounds - current_round)

        planned_budget = remaining_budget / remaining_rounds
        action_value = planned_budget / self.per_round_cap
        action_value = np.clip(action_value, self.min_ratio, self.max_ratio)
        return np.array([action_value], dtype=np.float32), None
