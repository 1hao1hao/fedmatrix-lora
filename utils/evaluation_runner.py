import os
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  # [新增]
import random # 确保导入
import torch  # 确保导入

class EvaluationRunner:
    def __init__(self, env, args, run_id=None):
        self.env = env
        self.args = args

        # 如果外部传了 run_id，就用外部的（确保一组实验在同一个文件夹）
        # 如果没传，就按以前的逻辑自己生成
        self.timestamp = run_id if run_id else datetime.now().strftime('%Y%m%d-%H%M%S')

        # 创建统一的评估结果目录 (CSV)
        self.base_save_dir = f"evaluation_results/{args.dataset}/{self.timestamp}"
        os.makedirs(self.base_save_dir, exist_ok=True)

        # [新增] 创建 TensorBoard 的根目录
        # 注意：我们将评估日志放在 eval_tensorboard_... 目录下，与训练日志分开，以免混淆
        self.tb_base_dir = f"eval_tensorboard_{args.dataset}/{self.timestamp}"

    def run(self, strategy_model, strategy_name, seed):
        """
        运行单次对比实验
        :param strategy_model: 拥有 .predict() 方法的预算策略对象
        :param strategy_name: 策略名称 (用于文件名)
        :param seed: 随机种子 (核心控制变量)
        """
        print(f"\n>>> [对比实验] 正在评估: {strategy_name} (Seed={seed}) <<<")

        # ====================================================
        # [核心修复] 强制重置所有全局随机数生成器
        # 确保 FedEnv._setup_server 中调用的 np.random 生成完全一样的客户端属性
        # ====================================================
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # ====================================================
        # [新增] 初始化该策略专用的 TensorBoard Writer
        # 例如: eval_tensorboard_cifar100/2026.../Random_FIM
        log_dir = os.path.join(self.tb_base_dir, strategy_name)
        writer = SummaryWriter(log_dir=log_dir)

        # 1. 设置种子
        # VecEnv 使用 .seed() 方法，而不是在 reset() 中传参
        try:
            self.env.seed(seed)
        except AttributeError:
            # 如果万一传入的是原生 Gym 环境，尝试原生方式
            pass

        # 2. 重置环境
        try:
            obs, info = self.env.reset(seed=seed)
        except TypeError:
            obs = self.env.reset()
            info = {}

        records = []
        overhead_records = []
        terminated = False
        truncated = False

        # 3. 实验循环
        while not (terminated or truncated):
            # 统一调用 .predict() 接口
            # 预算策略 predict 返回 (action, state)
            action, _ = strategy_model.predict(obs, deterministic=True)

            # 4. 环境步进
            # 自动检测是返回 4 个值 (VecEnv) 还是 5 个值 (Gymnasium)
            step_result = self.env.step(action)

            if len(step_result) == 4:
                # --- SB3 VecEnv 格式 (obs, rewards, dones, infos) ---
                obs, rewards, dones, infos = step_result

                # 因为我们设置了 n_envs=1，所以取列表的第一个元素
                # 注意：dones 在 VecEnv 中合并了 terminated 和 truncated
                terminated = dones[0]
                truncated = False # VecEnv 通常会自动处理截断并 reset，我们主要看 dones

                # 提取 info (也是列表)
                info = infos[0]

            elif len(step_result) == 5:
                # --- Gymnasium 格式 (obs, reward, terminated, truncated, info) ---
                obs, reward, terminated, truncated, info = step_result

            else:
                raise ValueError(f"未知的环境返回格式长度: {len(step_result)}")

            # 2. [核心修改] 全面记录数据
            current_round = info.get("current_round", 0)
            # 获取累积指标
            cum_time = info.get("cumulative_time_sec", 0.0)
            cum_budget = info.get("cumulative_budget_used", 0)
            loss = info.get("loss", 0.0)
            raw_lm_loss = info.get("raw_lm_loss", loss)
            rouge = info.get("rouge_l", 0.0)
            accuracy = info.get("accuracy", 0.0)
            # 5. 记录数据
            # 从 info 中提取所有关键指标
            row = {
                "Round": current_round,
                "Loss": loss,
                "Raw_LM_Loss": raw_lm_loss,
                "Rouge_l":rouge,
                "Accuracy": accuracy,
                # [新增] 记录单轮和累积
                "T_Actual": info.get("T_actual", 0.0),
                "T_Comp_Avg": info.get("T_comp_avg", 0.0), # 新增
                "T_Comm_Avg": info.get("T_comm_avg", 0.0), # 新增
                "Cumulative_Time": cum_time,

                "Consumed_Budget": info.get("consumed_budget_round", 0),
                "Cumulative_Budget": cum_budget,

                "Remaining_Budget": info.get("remaining_budget", 0),

                # [新增] 硬件效率指标
                "Avg_Flops": info.get("avg_flops", 0.0),
                "Avg_Memory_GB": info.get("avg_memory_gb", 0.0),
                "Avg_Comm_MB": info.get("avg_comm_cost_mb", 0.0),

                "Action_Value": float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
            }
            records.append(row)
            # [Overhead instrumentation] Keep stage timing in a separate CSV so the
            # historical eval CSV schema remains unchanged.
            overhead_row = info.get("overhead_metrics", {})
            if overhead_row:
                overhead_row = overhead_row.copy()
                overhead_row.setdefault("method", strategy_name)
                overhead_row.setdefault("dataset", self.args.dataset)
                overhead_row.setdefault("round", current_round)
                overhead_records.append(overhead_row)

            # 3. [核心修改] 写入 TensorBoard (分门别类)
            # Performance Group
            writer.add_scalar("Main/Loss", row["Loss"], current_round)
            writer.add_scalar("Main/Raw_LM_Loss", row["Raw_LM_Loss"], current_round)
            writer.add_scalar("Main/T_Actual", row["T_Actual"], current_round)
            writer.add_scalar("Main/Accuracy", accuracy, current_round)
            writer.add_scalar("Main/Rouge-L", rouge, current_round)

            # Action Group
            writer.add_scalar("Debug/Action_Value", row["Action_Value"], current_round)
            writer.add_scalar("Debug/Remaining_Budget", row["Remaining_Budget"], current_round)

            # [新增视图 1]: X轴 = 累计预算 (Analysis_Budget)
            writer.add_scalar("Analysis_Budget/Loss_vs_CumBudget", row["Loss"], int(cum_budget))

            # [新增视图 2]: X轴 = 累计时间 (Analysis_Time)
            # 注意: add_scalar 的 step 通常接受整数，我们将秒转为整数
            writer.add_scalar("Analysis_Time/Loss_vs_CumTime_Sec", row["Loss"], int(cum_time))
            writer.add_scalar("Analysis_Time/Rouge-L_vs_CumTime_Sec", rouge, int(cum_time))
            print(f"  [{strategy_name}] Round {row['Round']}: Loss={row['Loss']:.4f},rouge-l={rouge:.4f} T_Actual={row['T_Actual']:.2f}s")

            # 1. 基础视图 (X轴=轮次)
            writer.add_scalar("Latency/T_Actual_Round", row["T_Actual"], current_round)
            # 堆叠显示计算和通信延迟，便于分析瓶颈
            writer.add_scalars("Latency/Breakdown", {
                "Avg_Computation": row["T_Comp_Avg"],
                "Avg_Communication": row["T_Comm_Avg"]
            }, current_round)

        # ==========================================================
        # 将提前结束的曲线向右水平补齐 (Forward-fill)
        # ==========================================================
        if len(records) > 0:
            last_row = records[-1]
            current_r = int(last_row["Round"])
            max_r = self.args.max_rounds

            while current_r < max_r:
                current_r += 1
                pad_row = last_row.copy()
                pad_row["Round"] = current_r

                # 既然已经停止训练，本轮的消耗增量必须清零
                pad_row["T_Actual"] = 0.0
                pad_row["Consumed_Budget"] = 0
                pad_row["Avg_Memory_GB"] = 0.0
                pad_row["Avg_Comm_MB"] = 0.0
                pad_row["Avg_Flops"] = 0.0

                records.append(pad_row)

                # 补齐 TensorBoard 曲线
                writer.add_scalar("Main/Loss", pad_row["Loss"], current_r)
                writer.add_scalar("Main/Raw_LM_Loss", pad_row["Raw_LM_Loss"], current_r)
                if "Accuracy" in pad_row:
                    writer.add_scalar("Main/Accuracy", pad_row["Accuracy"], current_r)
                if pad_row["Rouge_l"] > 0:
                    writer.add_scalar("Main/Rouge-L", pad_row["Rouge_l"], current_r)

        # 4. 保存 CSV
        csv_path = os.path.join(self.base_save_dir, f"eval_{strategy_name}.csv")
        # 按照特定顺序保存列，方便阅读
        columns_order = ["Round", "Loss", "Raw_LM_Loss", "Rouge_l", "Accuracy",
                         "T_Actual","T_Comp_Avg","T_Comm_Avg", "Cumulative_Time",
                         "Consumed_Budget", "Cumulative_Budget", "Remaining_Budget",
                         "Avg_Memory_GB", "Avg_Comm_MB", "Avg_Flops", "Action_Value"]

        df = pd.DataFrame(records)
        # 确保列存在再排序
        valid_columns = [c for c in columns_order if c in df.columns]
        df = df[valid_columns]

        df.to_csv(csv_path, index=False)
        print(f">>> {strategy_name} 数据已保存: {csv_path}")
        # [Overhead instrumentation] Write only actual executed rounds; forward-filled
        # eval rows are intentionally not duplicated as synthetic overhead measurements.
        if overhead_records:
            overhead_csv_path = os.path.join(self.base_save_dir, f"overhead_{strategy_name}.csv")
            overhead_columns_order = [
                "round", "dataset", "method",
                "profiling_time_sec", "config_generation_time_sec",
                "server_scheduling_time_sec", "client_training_time_sec",
                "aggregation_time_sec", "evaluation_time_sec", "round_total_time_sec",
                "num_clients", "num_selected_clients", "total_lora_candidates_L",
                "avg_configs_per_client", "max_configs_per_client",
                "global_budget_Ck", "consumed_budget", "delay_candidate_count_D",
                "global_budget_W",
            ]
            overhead_df = pd.DataFrame(overhead_records)
            valid_overhead_columns = [c for c in overhead_columns_order if c in overhead_df.columns]
            overhead_df = overhead_df[valid_overhead_columns]
            overhead_df.to_csv(overhead_csv_path, index=False)
            print(f">>> {strategy_name} overhead 数据已保存: {overhead_csv_path}")
        writer.close()
