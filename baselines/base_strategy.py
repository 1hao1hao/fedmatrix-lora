class BaseStrategy:
    def predict(self, observation, deterministic=True):
        """
        统一接口，模拟 SB3 模型的 predict 方法。
        :param observation: 环境观测值
        :param deterministic: 是否确定性 (对随机策略可能无效，但为了接口一致需保留)
        :return: (action, state) - state 通常为 None
        """
        raise NotImplementedError("必须在子类中实现 predict 方法")