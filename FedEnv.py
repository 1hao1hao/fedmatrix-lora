# 文件: FedEnv.py
# 负责维护像 self.previous_loss 这样的跨轮次状态。它调用 Server，获取本轮结果，并更新自己的状态。
"""
定义一个名为 FedEnv 的类，它继承自 gym.Env。
这个类将 Server 封装成按轮推进的联邦微调实验环境，
负责定义观测空间、预算动作空间、step() 函数（执行一轮 FL）和 reset() 函数（开始新的 FL 实验）。
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# 导入我们自己项目中的 Server 和 Client 类
from models.algorithm import Server, Client
from utils.data_utils import  distribute_data
import copy # 确保导入
import torch # 确保 FedEnv.py 头部有这个导入

class FedEnv(gym.Env):
    """
    一个将联邦学习服务器封装为 Gym 环境的自定义类。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, args, tokenizer, train_dataset, test_dataset, val_loader, small_val_loader,val_text_labels, global_model, allocation_strategy='Random_FIM'):
        super(FedEnv, self).__init__()

        # --- 1. 存储从主进程接收的所有资源 ---
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.initial_global_model = global_model  # <--- 使用这个统一的变量名
        self.val_loader = val_loader
        self.small_val_loader = small_val_loader  # <--- [核心修复] 保存到 self
        self.val_text_labels = val_text_labels
        self.episode_num = 0
        self.allocation_strategy = allocation_strategy # [保存策略]

        # 分别跟踪 episode 内的步数和总步数
        self.episode_step = 0
        self.total_steps = 0  # 用于打印

        self.episode_length = args.epochs  # 将 args.epochs 视为 episode 的长度

        # [新增] 初始化累积时间变量
        self.cumulative_time = 0.0
        # 1. 定义状态空间 (State Space)
        # S_k = [L(k-1), C_consumed(k-1), T_actual(k-1), B_rem_ratio, k_ratio]
        # 使用 Box 空间，定义每个维度的上下限
        # 我们对值进行归一化，便于统一不同预算策略的输入尺度。
        self.observation_space = spaces.Box(
            low=np.array([0.0] * 5, dtype=np.float32),  # 确保是 float32
            high=np.array([1.0] * 5, dtype=np.float32), # 确保是 float32
            shape=(5,),
            dtype=np.float32
        )

        # 2. 定义动作空间 (Action Space)
        # A_k = [C^k, T_max]
        # C^k: LoRA 模块总成本, T_max: 最大容忍延迟
        # 同样进行归一化，输出范围在 [0, 1] 之间
        self.action_space = spaces.Box(
            low=np.array([0.1], dtype=np.float32),  # <--- 关键修改
            high=np.array([1.0], dtype=np.float32), # <--- 关键修改
            shape=(1,),  # 形状为 (1,)
            dtype=np.float32
        )

        # 3. 环境内部状态
        self.total_budget = args.total_budget
        self.remaining_budget = self.total_budget
        self.current_round = 0
        self.previous_loss = 10.0 # 给一个较大的初始 Loss

        # [新增] 记录上一次的准确率，防止跳崖
        self.previous_accuracy = 0.0
        self.previous_rouge = 0.0
        # 初始化内部状态变量
        self._state = None
        self.current_step = 0
        # self.total_fl_rounds = args.epochs

        # 定义动作的真实范围，用于反归一化
        self.C_max_per_round = args.num_clients * args.client_l_max # 单轮最大可能消耗
        self.T_max_global = args.fixed_t_max # 使用超参数中的固定延迟

        # [核心修复] 1. 保存一份干净的备份，绝对不要修改它
        # 这里的 global_model 是从 main 传进来的，我们要把它冻结并在 CPU 上备份
        self.clean_initial_model = copy.deepcopy(global_model)
        self.clean_initial_model.to('cpu') # 放在内存里省显存

    # --------------------------3.搭建联邦学习系统 (_setup_server)--------------------------
    def _setup_server(self):
        """初始化或重置联邦学习服务器和客户端"""
        # 1. 动态检测环境 GPU
        num_gpus = torch.cuda.device_count()
        print(f"--- [环境检测] 发现 {num_gpus} 个可用 GPU ---")

        # [优化 A] 母版模型始终留在 CPU 内存中
        # 理由：避免占用宝贵的 GPU 显存。客户端训练时再从 CPU copy 到指定的 GPU。
        # 这样 GPU 0 不会因为存了一份不用的母版而比其他 GPU 更容易 OOM。
        current_run_model = copy.deepcopy(self.clean_initial_model)
        current_run_model.to('cpu')

        # 2. 解析内存配置 (保持不变)
        try:
            mem_levels = [float(m) for m in self.args.mem_levels_gb.split(':')]
            mem_ratios = [int(r) for r in self.args.mem_dist_ratio.split(':')]
            assert len(mem_levels) == len(mem_ratios)
        except (ValueError, AssertionError):
            raise ValueError("内存水平格式错误")

        num_clients = self.args.num_clients
        total_ratio = sum(mem_ratios)

        # 计算客户端分布 (保持不变)
        num_clients_per_level = []
        for ratio in mem_ratios:
            num = round(num_clients * ratio / total_ratio)
            num_clients_per_level.append(int(num))

        diff = num_clients - sum(num_clients_per_level)
        if diff != 0:
            max_idx = np.argmax(num_clients_per_level)
            num_clients_per_level[max_idx] += diff

        client_memory_allocations = []
        for i, num in enumerate(num_clients_per_level):
            client_memory_allocations.extend([mem_levels[i]] * num)

        np.random.shuffle(client_memory_allocations)

        print("\n--- [环境设置] 正在创建异构客户端 ---")
        print(f"内存水平 (GB): {mem_levels}")
        print(f"客户端数量/水平: {num_clients_per_level}")
        print("-----------------------------------------")

        # 数据处理
        client_data_indices = distribute_data(self.train_dataset, self.args.num_clients, self.args)

        clients = []
        # 3. 客户端初始化 (保持轮询分配逻辑)
        for i in range(self.args.num_clients):
            client_memory = client_memory_allocations[i]

            # 轮询分配 GPU
            if num_gpus > 0:
                target_device_id = i % num_gpus
            else:
                target_device_id = -1

            client = Client(
                client_id=i,
                M_max=client_memory,
                T_layer_avg=np.random.uniform(self.args.delay_min, self.args.delay_max),
                L_min=self.args.client_l_min,
                L_max=self.args.client_l_max,
                total_layers=self.args.total_lora_layers,
                model=current_run_model,  # 传入 CPU 模型
                dataset=self.train_dataset,
                data_indices=client_data_indices[i],
                args=self.args,
                device_id=target_device_id # 告诉客户端它该去哪个 GPU
            )
            clients.append(client)

        # 4. Server 初始化与设备分配
        # [优化 B] 优先使用用户指定的 args.gpu，否则使用 GPU 0
        if torch.cuda.is_available() and self.args.gpu != -1:
            server_device_id = self.args.gpu
        elif num_gpus > 0:
            server_device_id = 0
        else:
            server_device_id = -1

        self.server = Server(
            clients=clients,
            total_layers=self.args.total_lora_layers,
            lambda_prime=self.args.lambda_prime,
            global_model=current_run_model, # 传入 CPU 模型，Server 内部可以自己处理
            val_loader=self.val_loader,
            small_val_loader=self.small_val_loader,
            val_text_labels=self.val_text_labels,
            tokenizer=self.tokenizer,
            args=self.args
        )

        # 强制修正 Server 的设备 (Server 的评估操作将在这里进行)
        self.server.device = torch.device(f"cuda:{server_device_id}" if server_device_id >= 0 else "cpu")

        # 如果 Server 初始化时没有把模型搬到 GPU，这里可以手动搬一下，或者在 evaluate 时搬
        # 为了安全，建议让 Server 持有一份 GPU 上的评估模型 (evaluation model)
        # 但注意不要修改传入的 current_run_model (因为那是给 Clients 用的母版)
        # Server 内部的 self.pristine_model 会在评估时 to(device)，所以这里只需设置 device 属性即可。

        self.previous_loss = 10.0

    def _normalize_state(self, loss_proxy, C_consumed, T_actual, B_rem, k):
        """将原始状态值归一化到 [0, 1] 范围"""
        # 状态的第一个值现在是 loss_proxy
        # [修改] Loss 的归一化
        # 假设 Loss 最大不会超过 10.0 (对于 CrossEntropy/NLL 通常在 0.5~8.0 之间)
        MAX_LOSS_CAP = 1.0 if getattr(self.args, "dataset", None) == "samsum" else 10.0
        norm_loss_proxy = np.clip(loss_proxy / MAX_LOSS_CAP, 0, 1)
        # norm_loss_proxy = np.clip(loss_proxy, 0, 1) # 1-performance 自然在 [0,1] 范围
        norm_C = np.clip(C_consumed / self.C_max_per_round, 0, 1)
        norm_T = np.clip(T_actual / self.T_max_global, 0, 1)
        norm_B = np.clip(B_rem / self.total_budget, 0, 1)
        norm_k = np.clip(k / self.args.max_rounds, 0, 1)
        return np.array([norm_loss_proxy, norm_C, norm_T, norm_B, norm_k], dtype=np.float32)

    # 添加一个空的 seed 方法以实现向后兼容
    def seed(self, seed=None):
        """
        这个方法是为了兼容某些仍然会调用 .seed() 的旧版辅助函数。
        在 Gymnasium 中，真正的种子设置发生在 reset() 方法中。
        所以这里我们什么都不用做。
        """
        pass

    def reset(self, seed=None, options=None):
        """重置环境到一个初始状态"""
        super().reset(seed=seed)

        print("====== [FedEnv] 环境重置，开始新的联邦学习流程 ======")
        self._setup_server()
        self.current_step = 0

        # 重置所有状态
        self.remaining_budget = self.total_budget
        self.current_round = 0
        self.previous_loss = 10.0 # 给一个较大的初始 Loss
        # [新增] 重置累积时间
        self.cumulative_time = 0.0

        # [新增] 重置时清零
        self.previous_accuracy = 0.0
        self.previous_rouge = 0.0

        # 注意：这里不再是 1 - performance
        initial_loss_proxy = self.previous_loss
        # 定义初始状态 S_0 = [L(-1), C(-1), T(-1), B_rem(0), k=0]
        initial_state_raw = (initial_loss_proxy, 0, 0, self.remaining_budget, 0)
        self._state = self._normalize_state(*initial_state_raw)


        info = {}  # Gym v26 之后 reset 返回 info

        # [核心修改] 在 reset 时，重置累积奖励和长度计数器
        self.current_round_reward = 0.0
        self.current_round_len = 0

        return self._state, info

    def step(self, action):
        """
        执行一个时间步（即一轮联邦学习）
        :param action: 归一化预算动作 [norm_C, norm_T]，也兼容只给预算的标量动作。
        :return: (next_state, reward, terminated, truncated, info)
        """
        # 更新步数计数器
        self.episode_step += 1 #总步数
        self.total_steps += 1 # 回合内步数

        # [新增] 计算当前是第几轮 (从1开始计数，方便阅读和取模)
        current_round_idx = self.current_round + 1
        # 修改打印内容
        print(
            f"\n{'=' * 10} [FedEnv] 总步数: {self.total_steps} (Episode 内: {self.episode_step}/{self.episode_length})(剩余预算: {self.remaining_budget}) {'='*10}")

        # 1. 反归一化动作，得到真实的 C^k 和 T_max
        # 兼容标量预算策略和数组形式的预算/延迟动作。
        if np.ndim(action) > 0:
            norm_C = float(action[0])
        else:
            norm_C = float(action)
        # 1. 计算本轮实际参与的客户端数量
        num_selected_clients = max(1, int(self.args.num_clients))

        # 2. 计算本轮理论上可能消耗的最大预算
        max_possible_cost_this_round = num_selected_clients * self.args.client_l_max

        # 3. 确定本轮预算的有效上限
        # 上限不能超过理论最大消耗，也不能超过剩余的总预算
        effective_budget_cap = max_possible_cost_this_round

        # 4. 根据归一化动作和有效上限，计算计划预算
        C_plan = min(int(norm_C * effective_budget_cap), self.remaining_budget)
        # 代理决定的是本轮"计划"消耗的预算
        T_max = self.args.fixed_t_max

        # === [核心修改] 决定是否评估 ===
        # 1. 定期检查: 每 5 轮 (5, 10, 15...)
        is_periodic_check = (current_round_idx % 5 == 1)

        # 2. 达到最大轮次: 必须评估
        is_last_round = (current_round_idx >= self.args.max_rounds)
        # 3. 预算预警: 如果剩余预算可能不足以支撑下一轮 (假设下一轮所有客户端都用满层数)
        max_possible_cost_per_round = self.args.num_clients * self.args.client_l_max*0.8
        is_budget_critical = self.remaining_budget < (max_possible_cost_per_round * 1.0)
        # 综合判断
        do_evaluate = is_periodic_check or is_last_round or is_budget_critical

        if do_evaluate:
            print(f"[环境决策] 本轮触发评估 (条件: 定期={is_periodic_check}, 尾轮={is_last_round}, 预算低={is_budget_critical})")

        # 2. 调用服务器执行
        # 将 Server 的返回结果扩展，以包含所有指标
        print(f"该轮次C_Plan: {C_plan} ，T_max: {T_max}，previous_performance: {self.previous_loss:.4f}")
        # [修改] 接收 Server 返回的详细指标
        (C_consumed, T_actual, avg_comp_latency, avg_comm_latency, new_loss,
         avg_flops, avg_memory, avg_comm_cost, rouge_score, accuracy, raw_lm_loss) = self.server.run_federated_round(
            C_plan, T_max, self.previous_loss, do_evaluate=do_evaluate,
            allocation_strategy=self.allocation_strategy
        )
        # [Overhead instrumentation] Pull passive timing metrics from Server without changing
        # the server return tuple consumed by the existing training/evaluation logic.
        overhead_metrics = getattr(self.server, "last_overhead_metrics", {}).copy()
        # ==============================================================
        # 如果本轮没有任何消耗（预算不足），说明全局模型根本没更新
        # 此时必须无条件继承上一轮的准确率，而不是接受底层的 0.0
        # ==============================================================
        if C_consumed == 0:
            accuracy = self.previous_accuracy
            rouge_score = self.previous_rouge
        else:
            self.previous_accuracy = accuracy
            self.previous_rouge = rouge_score

        new_performance = 0
        # [核心修改] 奖励计算：Loss 降低是好事
        # 如果 new_loss < previous_loss，loss_reduction 为正
        performance_improvement = self.previous_loss - new_loss

        # 3. 更新剩余预算
        self.remaining_budget -= C_consumed
        self.current_round += 1

        # 累积时间 (T_actual 即为本轮延迟)
        self.cumulative_time += T_actual
        # 累积消耗预算 (总预算 - 剩余预算)
        cumulative_budget_used = self.total_budget - self.remaining_budget
        # 4. [新] 计算奖励
        # 奖励函数超参数
        alpha, beta, gamma = 10.0, 0.005, 0.01
        R_success, R_fail = 100, -100

        # 中间奖励
        reward = alpha * performance_improvement- beta * C_consumed - gamma * T_actual

        # 5. [新] 检查结束条件 (基于性能指标)
        terminated = False
        print(f"[FedEnv] 当前性能指标（损失）: {new_loss:.4f} (目标: <= {self.args.performance_target})")

        if self.remaining_budget <= 0:
            print("[FedEnv] 预算耗尽，Episode 结束。")
            terminated = True
        elif self.current_round >= self.args.max_rounds:
            print("[FedEnv] 达到最大轮次，Episode 结束。")
            terminated = True

        # 如果是最后一步，增加最终奖励
        if terminated:
            if new_loss <= self.args.performance_target:
                reward += R_success
                print(f"[FedEnv] 成功达到目标，获得最终奖励: +{R_success}")
            else:
                reward += R_fail
                print(f"[FedEnv] 未能达到目标，受到最终惩罚: {R_fail}")

        truncated = False

        # 6. 构建下一个状态
        self.previous_loss = new_loss
        loss_proxy = new_loss

        # self.previous_performance = new_performance
        # loss_proxy = 1.0 - new_performance
        next_state_raw = (loss_proxy, C_consumed, T_actual, self.remaining_budget, self.current_round)
        state = self._normalize_state(*next_state_raw)

        # [核心修改] 创建包含所有自定义指标的 info 字典
        info = {
            "current_round": current_round_idx, # <--- 记录当前轮次
            "loss": new_loss,
            "raw_lm_loss": raw_lm_loss,
            "performance": new_performance, # 保留兼容性
            "rouge_l": rouge_score, # 记录到 info
            "accuracy": accuracy,
            # --- 延迟指标 ---
            "T_actual": T_actual,                 # 本轮总延迟 (Max)
            "T_comp_avg": avg_comp_latency,       # 本轮平均计算延迟
            "T_comm_avg": avg_comm_latency,       # 本轮平均通信延迟
            "cumulative_time_sec": self.cumulative_time, # 累计总延迟 (横轴)

            # --- 硬件与效率指标 ---
            "avg_flops": avg_flops,         # 平均计算量
            "avg_memory_gb": avg_memory,    # 平均内存占用
            "avg_comm_cost_mb": avg_comm_cost, # 平均通信开销

            "consumed_budget_round": C_consumed,

            # --- [新增] 累积指标 ---
            "cumulative_budget_used": cumulative_budget_used, # 累计预算 (层)
            "remaining_budget": self.remaining_budget,
            "evaluated_this_step": do_evaluate, # 记录本轮是否进行了评估
            "overhead_metrics": overhead_metrics
        }

        # 只有在 episode 结束时，才需要记录最终的 episode 级信息
        if terminated:
            info["episode"] = {
                "r": self.current_round_reward + reward, # 累积奖励
                "l": self.current_round_len + 1,        # 累积长度
            }

        # 更新累积奖励和长度
        self.current_round_reward += reward
        self.current_round_len += 1

        # [修复] 必须返回 Gymnasium 标准的 5 个值，不要手动合并 done
        return state, float(reward), terminated, truncated, info

    def render(self, mode='human', close=False):
        # 简单打印当前状态信息
        print(f"  Step: {self.current_step}")
        print(f"  Loss: {self.previous_loss:.4f}")
        print(f"  Reward: {self._state[4]:.4f}")
