# 文件: models/scheduling.py
"""Scheduling primitives shared by federated server/client code."""

import collections
import numpy as np


Configuration = collections.namedtuple(
    'Configuration',
    ['client_id', 'value', 'cost', 'delay', 'memory', 'assignment']
)


class SchedulingMixin:
    def _solve_mckp_with_dp(self, configs_by_client: dict, budget: int) -> list:
        """
        [新] 使用动态规划 (DP) 精确求解多选背包问题 (MCKP)。

        :param configs_by_client: 按 client_id 分组的配置字典。
        :param budget: 总预算 W (即 L_k)。
        :return: 最终被选中的最优配置列表。
        """
        client_ids = sorted(configs_by_client.keys())
        num_clients = len(client_ids)

        max_possible_total_cost = sum(
            max((config.cost for config in configs), default=0)
            for configs in configs_by_client.values()
        )

        budget = int(min(budget, max_possible_total_cost))

        if budget <= 0:
            return []

        dp = np.full((num_clients + 1, budget + 1), -1.0)
        path = [[None for _ in range(budget + 1)] for _ in range(num_clients + 1)]

        dp[0, :] = 0.0

        for i in range(1, num_clients + 1):
            client_id = client_ids[i - 1]
            client_configs = configs_by_client[client_id]

            for w in range(budget + 1):
                dp[i, w] = dp[i - 1, w]
                path[i][w] = None

                for config in client_configs:
                    cost = int(config.cost)
                    value = config.value

                    if w >= cost:
                        new_value = dp[i - 1, w - cost] + value

                        if new_value > dp[i, w]:
                            dp[i, w] = new_value
                            path[i][w] = config

        final_selection = []
        w = budget
        for i in range(num_clients, 0, -1):
            chosen_config = path[i][w]
            if chosen_config is not None:
                final_selection.append(chosen_config)
                w -= int(chosen_config.cost)

        return final_selection

    def integrated_scheduling(self, all_configs, L_k):
        if not all_configs:
            return [], 0.0

        feasible_delays = sorted(list(set(c.delay for c in all_configs)))
        best_J, best_final_configs, best_T_max_decision = -np.inf, [], 0

        for T_max_candidate in feasible_delays:
            P_T_max = [c for c in all_configs if c.delay <= T_max_candidate]
            configs_by_client = collections.defaultdict(list)
            for c in P_T_max:
                configs_by_client[c.client_id].append(c)

            final_configs_for_T_max = self._solve_mckp_with_dp(configs_by_client, L_k)

            V_total = sum(c.value for c in final_configs_for_T_max)
            J_current = V_total - self.lambda_prime * T_max_candidate
            if J_current > best_J:
                best_J, best_final_configs, best_T_max_decision = J_current, final_configs_for_T_max, T_max_candidate

        return best_final_configs, best_T_max_decision
