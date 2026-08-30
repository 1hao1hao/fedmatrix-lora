import numpy as np
from .base_strategy import BaseStrategy

class RandomStrategy(BaseStrategy):
    def __init__(self, low=0.1, high=1.0, shape=(1, 1), seed=42):
        """
        :param low: 动作下限 (预算比例)
        :param high: 动作上限
        :param shape: 动作维度
        """
        self.low = low
        self.high = high
        self.shape = shape
        # [核心修复] 使用独立的 RandomState，不干扰全局 np.random
        self.rng = np.random.RandomState(seed)

    def predict(self, observation, deterministic=True):
        # 使用 self.rng 而不是 np.random
        random_budget_ratio = self.rng.uniform(low=self.low, high=self.high, size=self.shape)
        action = random_budget_ratio.astype(np.float32)
        return action, None