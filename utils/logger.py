# 文件: utils/logger.py

import os
import pandas as pd
from datetime import datetime


class MetricsLogger:
    def __init__(self, log_dir="logs"):
        """
        初始化日志记录器。
        :param log_dir: 存放日志文件的目录。
        """
        self.log_dir = log_dir
        self.round_data = []
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def log_round_data(self, round_num, performance, avg_flops, avg_memory,
                       avg_comm_cost, avg_time, remaining_budget, consumed_budget):
        """
        记录一轮联邦学习的所有关键指标。
        """
        log_entry = {
            'Round': round_num,
            'Performance': performance,
            'Avg_TFLOPs': avg_flops,
            'Avg_Memory_GB': avg_memory,
            'Avg_Comm_Cost_MB': avg_comm_cost,
            'Avg_Wall_Time_Sec': avg_time,
            'Remaining_Budget': remaining_budget,
            'Consumed_Budget_Round': consumed_budget
        }
        self.round_data.append(log_entry)
        print(f"[Logger] 已记录第 {round_num} 轮数据。性能: {performance:.4f}")

    def save_to_csv(self, dataset_name, distribution, episode_num):
        """
        将当前 episode 收集的所有数据保存到 CSV 文件。
        """
        if not self.round_data:
            print("[Logger] 没有数据可保存。")
            return

        # 创建一个唯一的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{distribution.replace('/', '_')}_episode_{episode_num}_{timestamp}.csv"
        filepath = os.path.join(self.log_dir, filename)

        # 转换为 pandas DataFrame 并保存
        df = pd.DataFrame(self.round_data)
        df.to_csv(filepath, index=False)

        print(f"[Logger] Episode {episode_num} 的数据已成功保存到: {filepath}")

    def reset(self):
        """
        清空缓存，为下一个 episode 做准备。
        """
        self.round_data = []