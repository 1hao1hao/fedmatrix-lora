# 文件: models/client.py
"""Federated client implementation."""

from collections import OrderedDict
import collections
import copy
import gc
import re

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from models.lora_utils import apply_lora_to_model, find_lora_candidates, infer_lora_candidate_metadata
from models.scheduling import Configuration
from utils.data_utils import custom_collate_fn_for_generation


class Client:
    def __init__(self, client_id, M_max, T_layer_avg, L_min, L_max, total_layers,
                 model, dataset, data_indices, args, device_id=0):
        self.client_id = client_id
        # TODO 训练层的平均时间需要计算推导
        # 硬件约束 内存 随机算子
        self.M_max = M_max
        # 训练单个层的平均时间 随机算子
        self.T_layer_avg = T_layer_avg
        # 训练层数的上下限(超参数)
        self.L_min = L_min
        self.L_max = L_max
        # 模型中可微调的 LoRA 总层数 (BERT-base 有 24 个)
        self.total_layers = total_layers
        self.args = args
        # [核心修改] 根据传入的 device_id 设置设备
        if device_id >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")

        # 持有模型、数据和训练组件
        self.model = model  # 这是一个共享的基模型引用
        self.dataset = dataset # 完整的训练数据集对象。
        self.data_indices = data_indices # 本地数据列表
        self.optimizer = None  # 将在训练时初始化


        # --- 1. 物理感知内存模型 (保持物理常数，控制变量) ---
        # 假设 ViT-Base / OPT-1.3B 规模
        # 为了让实验更有趣，我们可以让不同层的内存略有不同 (模拟 Transformer 不同层的架构差异)
        # 但必须是固定的，不依赖随机噪声
        base_mem = 6 * 1024 * 1024 # 6MB

        # 构造一个确定的正弦波波动，模拟不同层激活值的微小差异
        # 这样既不是全平的，也不是乱随机的，且对所有实验一致
        layer_indices = np.arange(total_layers)
        variation = np.sin(layer_indices) * (0.5 * 1024 * 1024) # +/- 0.5MB

        self.sigma_d = {j: base_mem + variation[j] for j in range(total_layers)}
        self.sigma_s = {j: 1 * 1024 * 1024 for j in range(total_layers)}

        # 基础常量
        self.eta = 4
        self.zeta = 2
        self.phi_theta = {j: 12288 for j in range(total_layers)}
        self.C_p = 440 * 1024 * 1024
        self.C_c = 100 * 1024 * 1024

        # 激活值大小应该是固定的
        # Batch Size * Sequence Length * Hidden Size * Layer Factor
        # 假设 seq_len=128 (NLP) 或 patch_num=197 (ViT)
        # 一个典型的 Layer 激活值大约在 6MB 左右 (对于 bs=1)
        # 我们这里设定一个基准值，然后加上一点点层级差异(模拟不同层的缓存策略)
        base_activation = 6 * 1024 * 1024 # 6MB
        # 动态激活 (Forward + Backward stored activations)
        self.sigma_d = {j: base_activation for j in range(total_layers)}
        # 静态激活 (Temporary buffers)
        self.sigma_s = {j: 1 * 1024 * 1024 for j in range(total_layers)} # 1MB

        # 陈旧度计数器 (Staleness / Starvation Counter)
        # 记录每一层有多少轮没有被训练了。初始为 0。
        self.staleness = np.zeros(total_layers)

        # 探索系数 alpha
        # 用于平衡 "梯度利用(Exploitation)" 和 "抗饥饿(Exploration)"
        # 0.1 表示每闲置 1 轮，增加等同于 0.1 单位梯度范数的评分
        self.alpha_starvation = 0
        # --- 参数模拟结束 ---
        # === [新增] 物理环境模拟 ===
        # 1. 计算相关的物理量
        # n_i: 本地样本数
        self.num_samples = len(data_indices)
        # b: 本地 Batch Size
        self.batch_size = args.local_bs
        # E: 本地 Epoch
        self.local_ep = args.local_ep
        # 计算本地一轮训练的总 Batch 数: (n_i * E) / b
        self.total_batches = (self.num_samples * self.local_ep) / self.batch_size

        # T_layer_avg 在此处被定义为: "单层、单个Batch" 的平均计算时间
        # 为了复用你之前传入的 T_layer_avg (它是随机生成的 0.8~2.0s)，我们假设那个值是 "处理完本地所有数据所需的单层总时间" 的基准
        # 或者为了严谨，我们直接使用传入的参数作为 "单层总计算时间"
        self.comp_time_per_layer_total = T_layer_avg

        # 2. 通信相关的物理量
        # 模拟上传带宽 BW_i (MB/s)。
        # 假设带宽在 5 MB/s 到 50 MB/s 之间随机 (可根据需要调整)
        self.bandwidth = np.random.uniform(5.0, 50.0)

        # 估算单个 LoRA 层的参数大小 (MB)
        # 假设 Hidden=768, Rank=8, float32 (4 bytes).
        # Size = 2 * d * r * 4 bytes
        # ViT-Base d=768. Size approx 0.05 MB per layer.
        self.lora_layer_size_mb = (2 * 768 * 8 * 4) / (1024 * 1024)
        self.score_ema_mean = None
        self.score_ema_sq_mean = None
        self.score_ema_count = 0
        self.last_raw_scores = None
        self.last_ema_scores = None
        self.last_depth_prior = None
        self.last_position_signal = 0.0
        self.last_position_weight = 0.0
        self.last_position_direction = "none"
        self.last_agreement_scores = None
        self.last_final_utility = None
        self.last_config_generation_summary = {}
        self.candidate_modules = find_lora_candidates(self.model)[:self.total_layers]
        self.candidate_metadata = [
            infer_lora_candidate_metadata(name) for name in self.candidate_modules
        ]

    def _depth_power_extra(self) -> float:
        distribution = str(getattr(self.args, "distribution", "")).lower()
        severe_extra = float(getattr(self.args, "depth_severe_extra", 2.0))
        non_iid_extra = float(getattr(self.args, "depth_non_iid_extra", 1.0))

        if distribution.startswith("dirichlet:"):
            try:
                alpha = float(distribution.split(":", 1)[1])
            except ValueError:
                alpha = None
            if alpha is not None and alpha <= 0.1:
                return severe_extra
            return non_iid_extra
        if "patho" in distribution:
            return severe_extra
        return 0.0

    def _utility_mode(self) -> str:
        return str(getattr(self.args, "score_utility_mode", "raw") or "raw").lower()

    def _utility_mode_enabled(self) -> bool:
        return self._utility_mode() != "raw"

    def _position_calibration_mode(self) -> str:
        return str(getattr(self.args, "position_calibration_mode", "legacy") or "legacy").lower()

    def _resource_ratio(self) -> float:
        try:
            mem_levels = [
                float(item)
                for item in str(getattr(self.args, "mem_levels_gb", "")).split(":")
                if item.strip()
            ]
        except ValueError:
            mem_levels = []
        if not mem_levels:
            return 0.0
        mem_min = min(mem_levels)
        mem_max = max(mem_levels)
        return float(np.clip((float(self.M_max) - mem_min) / max(mem_max - mem_min, 1e-8), 0.0, 1.0))

    def _fah_quant_bits(self) -> int:
        # FAH-QLoRA-style proxy: stronger clients can use 8-bit bases,
        # constrained clients use 4-bit bases. Actual base-model quantized
        # kernels are not instantiated in this code path.
        return 8 if self._resource_ratio() >= 0.65 else 4

    def _fah_rank_for_round(self) -> int:
        progress = float(getattr(self, "current_round", 0)) / max(1, int(getattr(self.args, "max_rounds", 1)) - 1)
        if progress < 0.30:
            average_rank = 16
        elif progress < 0.70:
            average_rank = 8
        else:
            average_rank = 4

        tiers = [2, 4, 8, 16]
        idx = tiers.index(average_rank)

        resource_ratio = self._resource_ratio()
        try:
            delay_min = float(getattr(self.args, "delay_min", 0.8))
            delay_max = float(getattr(self.args, "delay_max", 2.0))
            speed_ratio = 1.0 - ((float(self.T_layer_avg) - delay_min) / max(delay_max - delay_min, 1e-8))
            speed_ratio = float(np.clip(speed_ratio, 0.0, 1.0))
        except (TypeError, ValueError):
            speed_ratio = 0.5

        capability = 0.6 * resource_ratio + 0.4 * speed_ratio
        if capability >= 0.75:
            idx += 1
        elif capability < 0.35:
            idx -= 1

        return tiers[int(np.clip(idx, 0, len(tiers) - 1))]

    def calculate_fah_memory(self, assignment: list) -> float:
        quant_scale = self._fah_quant_bits() / 16.0
        original_base = self.C_p
        try:
            self.C_p = original_base * quant_scale
            return self.calculate_memory(assignment)
        finally:
            self.C_p = original_base

    def calculate_fah_latency_components(self, assignment):
        t_comp, t_comm, _ = self.calculate_latency_components(assignment)
        quant_bits = self._fah_quant_bits()
        quant_compute_multiplier = 1.08 if quant_bits == 4 else 1.03
        quant_comm_multiplier = quant_bits / 16.0
        t_comp = t_comp * quant_compute_multiplier
        t_comm = t_comm * quant_comm_multiplier
        return t_comp, t_comm, t_comp + t_comm

    def get_depth_prior(self, scale: str = "one_ten") -> np.ndarray:
        """Return a monotonic depth prior using the current depth policy."""
        if self.total_layers <= 0:
            return np.array([], dtype=float)
        is_sentence_pair = self.args.dataset in ['qqp', 'mnli', 'qnli']
        depth_power = 1.0
        if is_sentence_pair:
            depth_power += 1.5
        depth_power += self._depth_power_extra()
        depth = np.arange(self.total_layers, dtype=float) / max(1, self.total_layers - 1)
        prior = np.power(1.0 + depth, depth_power)
        prior_min = float(np.min(prior))
        prior_max = float(np.max(prior))
        if prior_max > prior_min:
            prior = (prior - prior_min) / (prior_max - prior_min)
        else:
            prior = np.zeros_like(prior)
        if scale == "one_ten":
            prior = prior * 9.0 + 1.0
        self.last_depth_prior = prior.copy()
        return prior

    def _sanitize_scores(self, scores) -> np.ndarray:
        return np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def _stable_ema_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        beta = float(getattr(self.args, "score_ema_beta", 0.8))
        beta = min(max(beta, 0.0), 0.999999)
        raw_scores = self._sanitize_scores(raw_scores)

        if self.score_ema_mean is None or len(self.score_ema_mean) != len(raw_scores):
            self.score_ema_mean = raw_scores.copy()
            self.score_ema_sq_mean = np.square(raw_scores)
            self.score_ema_count = 1
        else:
            self.score_ema_mean = beta * self.score_ema_mean + (1.0 - beta) * raw_scores
            self.score_ema_sq_mean = beta * self.score_ema_sq_mean + (1.0 - beta) * np.square(raw_scores)
            self.score_ema_count += 1

        eps = float(getattr(self.args, "score_stability_eps", 1e-8))
        ema_var = np.maximum(self.score_ema_sq_mean - np.square(self.score_ema_mean), 0.0)
        stable_score = self.score_ema_mean / np.sqrt(ema_var + eps)
        return self._sanitize_scores(stable_score)

    def _block_coherence_bonus(self, assignment: list, layer_idx: int) -> float:
        if not getattr(self.args, "use_block_coherence", False):
            return 0.0
        weight = float(getattr(self.args, "block_coherence_weight", 0.0))
        qkv_weight = float(getattr(self.args, "qkv_completeness_weight", 0.0))
        if (weight <= 0 and qkv_weight <= 0) or layer_idx >= len(self.candidate_metadata):
            return 0.0
        meta = self.candidate_metadata[layer_idx]
        if meta.get("group_type") not in {"attention", "mlp"}:
            return 0.0
        same_group_count = 0
        selected_attention_types = set()
        for idx, active in enumerate(assignment):
            if idx == layer_idx or active <= 0 or idx >= len(self.candidate_metadata):
                continue
            other = self.candidate_metadata[idx]
            if (
                other.get("block_id") == meta.get("block_id")
                and other.get("group_type") == meta.get("group_type")
            ):
                same_group_count += 1
                if other.get("module_type") in {"q", "k", "v"}:
                    selected_attention_types.add(other.get("module_type"))

        bonus = weight * same_group_count
        module_type = meta.get("module_type")
        if qkv_weight > 0 and module_type in {"q", "k", "v"}:
            completed_types = selected_attention_types | {module_type}
            if len(completed_types) == 2:
                bonus += qkv_weight
            elif len(completed_types) >= 3:
                bonus += 2.0 * qkv_weight
        return bonus

    def _cifar100_coverage_bonus(self, assignment: list, layer_idx: int) -> float:
        if (
            getattr(self.args, "dataset", "") != "cifar100"
            or getattr(self, "current_allocation_strategy", "") != "Random_FIM"
            or getattr(self.args, "cifar100_prior_mode", "legacy_prior") != "legacy_prior"
            or layer_idx >= len(self.candidate_metadata)
        ):
            return 0.0
        meta = self.candidate_metadata[layer_idx]
        block_ids = sorted({
            int(item.get("block_id", 0))
            for item in self.candidate_metadata[:self.total_layers]
        })
        if not block_ids:
            return 0.0
        block_pos = {block_id: pos for pos, block_id in enumerate(block_ids)}
        block_id = int(meta.get("block_id", 0))
        block_depth = block_pos.get(block_id, 0) / max(1, len(block_ids) - 1)
        band_count = 3
        band = min(band_count - 1, int(block_pos.get(block_id, 0) * band_count / max(1, len(block_ids))))

        active_bands = set()
        same_block_active = 0
        same_group_active = 0
        selected_attention_types = set()
        attention_type_counts = {"q": 0, "k": 0, "v": 0}
        active_blocks = set()
        for idx, active in enumerate(assignment):
            if active <= 0 or idx >= len(self.candidate_metadata):
                continue
            other = self.candidate_metadata[idx]
            other_block = int(other.get("block_id", 0))
            active_blocks.add(other_block)
            other_band = min(band_count - 1, int(block_pos.get(other_block, 0) * band_count / max(1, len(block_ids))))
            active_bands.add(other_band)
            other_module_type = other.get("module_type")
            if other.get("group_type") == "attention" and other_module_type in attention_type_counts:
                attention_type_counts[other_module_type] += 1
            if other_block == block_id:
                same_block_active += 1
                if other.get("group_type") == meta.get("group_type"):
                    same_group_active += 1
                    if other_module_type in {"q", "k", "v"}:
                        selected_attention_types.add(other_module_type)

        bonus = 0.0
        if band not in active_bands:
            bonus += 0.35
        if same_block_active == 0:
            if len(active_blocks) < 3:
                bonus += 0.40
            elif len(active_blocks) < 5 and (block_depth <= 0.15 or block_depth >= 0.85):
                bonus += 0.20
            else:
                bonus -= 1.10
        elif same_block_active <= 2:
            bonus += 0.65
        else:
            bonus += 0.25

        if block_depth <= 0.15:
            bonus += 0.55
        elif block_depth >= 0.85:
            bonus += 0.25

        module_type = meta.get("module_type")
        if module_type in {"q", "k", "v"}:
            min_count = min(attention_type_counts.values())
            if attention_type_counts[module_type] == min_count:
                bonus += 1.80
            if module_type == "q":
                bonus += 1.20
            elif module_type == "k":
                bonus += 0.65
            elif module_type == "v":
                bonus -= 0.45
            if attention_type_counts[module_type] >= min_count + 1:
                bonus -= 1.25 * (attention_type_counts[module_type] - min_count)
            if attention_type_counts["v"] > attention_type_counts["q"] + 1 and module_type == "v":
                bonus -= 1.50

            completed_types = selected_attention_types | {module_type}
            if len(completed_types) == 2:
                bonus += 0.90
            elif len(completed_types) >= 3:
                bonus += 1.60
        return bonus

    def _use_cifar100_qkv_bundles(self) -> bool:
        return False

    def _attention_block_groups(self) -> dict:
        groups = collections.defaultdict(dict)
        for idx, meta in enumerate(self.candidate_metadata[:self.total_layers]):
            if meta.get("group_type") != "attention":
                continue
            module_type = meta.get("module_type")
            if module_type in {"q", "k", "v"}:
                groups[int(meta.get("block_id", 0))][module_type] = idx
        return {
            block_id: [items[kind] for kind in ("q", "k", "v") if kind in items]
            for block_id, items in groups.items()
            if len(items) == 3
        }

    def _cifar100_bundle_value(self, assignment: list, block_groups: dict, block_ids: list) -> float:
        scores = self._sanitize_scores(self.scores)
        value = float(sum(scores[idx] for idx, active in enumerate(assignment) if active > 0 and idx < len(scores)))
        selected_blocks = []
        for block_id, indices in block_groups.items():
            if all(assignment[idx] > 0 for idx in indices):
                selected_blocks.append(block_id)
        value += 3.0 * len(selected_blocks)
        if selected_blocks and block_ids:
            band_count = 3
            pos = {block_id: offset for offset, block_id in enumerate(block_ids)}
            bands = {
                min(band_count - 1, int(pos.get(block_id, 0) * band_count / max(1, len(block_ids))))
                for block_id in selected_blocks
            }
            value += 1.0 * len(bands)
        return value

    def _generate_cifar100_qkv_bundle_configurations(self, T_deadline: float) -> list:
        print(f"  [客户端 {self.client_id}] 使用 CIFAR100 q/k/v bundle 配置生成...")
        scores = self._sanitize_scores(self.scores)
        block_groups = self._attention_block_groups()
        block_ids = sorted(block_groups)
        if not block_ids:
            return []

        c_time_max = int(T_deadline // self.T_layer_avg)
        c_max_global = min(self.L_max, c_time_max)
        if c_max_global < self.L_min:
            return []

        block_score = {
            block_id: float(sum(scores[idx] for idx in indices) + np.mean([scores[idx] for idx in indices]))
            for block_id, indices in block_groups.items()
        }
        ranked_blocks = sorted(block_ids, key=lambda block_id: block_score[block_id], reverse=True)

        def uniform_blocks(count):
            if count >= len(block_ids):
                return list(block_ids)
            positions = sorted(set(np.linspace(0, len(block_ids) - 1, count, dtype=int).tolist()))
            cursor = 0
            while len(positions) < count and cursor < len(block_ids):
                if cursor not in positions:
                    positions.append(cursor)
                cursor += 1
            return [block_ids[pos] for pos in sorted(positions[:count])]

        def anchored_blocks(count):
            chosen = []
            for block_id in [block_ids[0], block_ids[-1]]:
                if block_id not in chosen:
                    chosen.append(block_id)
                if len(chosen) >= count:
                    return chosen
            for block_id in ranked_blocks:
                if block_id not in chosen:
                    chosen.append(block_id)
                if len(chosen) >= count:
                    break
            return chosen

        def make_assignment(chosen_blocks, extra_count=0):
            assignment = [0] * self.total_layers
            for block_id in chosen_blocks:
                for idx in block_groups.get(block_id, []):
                    assignment[idx] = 1
            if extra_count > 0:
                ranked_extra = sorted(range(self.total_layers), key=lambda idx: scores[idx], reverse=True)
                added = 0
                for idx in ranked_extra:
                    if assignment[idx] > 0:
                        continue
                    assignment[idx] = 1
                    if self.calculate_memory(assignment) > self.M_max:
                        assignment[idx] = 0
                        continue
                    added += 1
                    if added >= extra_count:
                        break
            return assignment

        configs = []
        seen = set()
        memory_rejects = 0
        transition_checks = 0
        max_bundle_count = max(1, min(len(block_ids), c_max_global // 3))
        for count in range(1, max_bundle_count + 1):
            candidate_sets = [
                ranked_blocks[:count],
                uniform_blocks(count),
                anchored_blocks(count),
                block_ids[:count],
                block_ids[-count:],
            ]
            for chosen in candidate_sets:
                if not chosen:
                    continue
                base_cost = 3 * len(chosen)
                max_extra = min(2, max(0, c_max_global - base_cost))
                for extra_count in range(max_extra + 1):
                    assignment = make_assignment(chosen, extra_count=extra_count)
                    key = tuple(assignment)
                    if key in seen:
                        continue
                    seen.add(key)
                    transition_checks += 1
                    active_count = sum(1 for value in assignment if value > 0)
                    if active_count < self.L_min or active_count > c_max_global:
                        continue
                    mem = self.calculate_memory(assignment)
                    if mem > self.M_max:
                        memory_rejects += 1
                        continue
                    _, _, delay = self.calculate_latency_components(assignment)
                    if delay > T_deadline:
                        continue
                    configs.append(Configuration(
                        client_id=self.client_id,
                        value=self._cifar100_bundle_value(assignment, block_groups, block_ids),
                        cost=float(active_count),
                        delay=delay,
                        memory=mem,
                        assignment=tuple(assignment),
                    ))

        final_configs = self._prune_configurations(configs)
        self.last_config_generation_summary = {
            "generated_configs": len(configs),
            "pruned_configs": len(final_configs),
            "memory_rejects": memory_rejects,
            "transition_checks": transition_checks,
            "c_time_max": c_time_max,
            "c_max_global": c_max_global,
        }
        print(f"  [客户端 {self.client_id}] 生成了 {len(final_configs)} 个 CIFAR100 bundle 配置。")
        return final_configs

    def _prepare_input_batch(self, batch):
        """Move tensor-like model inputs to the client device."""
        allowed_keys = ['input_ids', 'attention_mask', 'pixel_values', 'labels']
        input_batch = {}
        for k, v in batch.items():
            if k not in allowed_keys:
                continue
            if isinstance(v, torch.Tensor):
                input_batch[k] = v.to(self.device)
            elif isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], torch.Tensor):
                    input_batch[k] = torch.stack(v, dim=1).to(self.device)
                elif isinstance(v[0], (int, float)):
                    input_batch[k] = torch.tensor(v).to(self.device)
                elif isinstance(v[0], list):
                    input_batch[k] = torch.tensor(v).to(self.device)
        return input_batch

    def calculate_memory(self, assignment: list) -> float:
        """
        [Rank感知版] 根据分配方案估算显存峰值。
        完美兼容 LEGEND 的异构秩 (Rank)。显存消耗随 Rank 线性缩放。
        """
        base_memory_bytes = self.C_p + self.C_c
        chosen_indices =[j for j, is_assigned in enumerate(assignment) if is_assigned > 0]

        if not chosen_indices:
            return base_memory_bytes / (1024 ** 3)

        optimizer_and_dynamic_activation_mem_bytes = 0
        for j in chosen_indices:
            # 自动解析 Rank (如果为 1，则代表基准 Rank 8)
            rank = 8 if assignment[j] == 1 else assignment[j]

            # 计算缩放因子 (以 Rank=8 为 1.0)
            scale_factor = rank / 8.0

            # 优化器状态和动态激活显存均随 Rank 线性缩放
            term1 = self.zeta * (self.phi_theta[j] * scale_factor)
            term2 = self.eta * (self.sigma_d[j] * scale_factor)
            optimizer_and_dynamic_activation_mem_bytes += term1 + term2

        if getattr(self.args, "memory_model", "position") == "additive":
            # Ablation: ignore the suffix activation term and approximate memory additively.
            static_activation_mem_bytes = sum(self.eta * self.sigma_s[j] for j in chosen_indices)
        else:
            # 静态激活显存 (与网络深度有关，与 Rank 关系较小，保持原样)
            min_chosen_idx = min(chosen_indices)
            static_activation_mem_bytes = sum(self.eta * self.sigma_s[j] for j in range(min_chosen_idx, self.total_layers))

        batch_size = self.args.local_bs
        total_dynamic_part_bytes = 24 * (optimizer_and_dynamic_activation_mem_bytes + static_activation_mem_bytes)

        return (base_memory_bytes + total_dynamic_part_bytes) / (1024 ** 3)

    def _prune_configurations(self, configs):
            """
            [极速修剪] 基于支配性原理。
            对于每一个 Cost (层数)，只保留 Score 最高的那一个配置。
            这是无损压缩，不会降低 DP 求解的最优性。
            """
            if not configs:
                return []

            # 字典映射: cost -> max_value_config
            # 这种查找方式是 O(N) 的，非常快
            best_configs_map = {}

            for conf in configs:
                cost = conf.cost
                # 如果当前 cost 还没记录，或者当前 conf 的分数更高，则更新
                if cost not in best_configs_map:
                    best_configs_map[cost] = conf
                else:
                    if conf.value > best_configs_map[cost].value:
                        best_configs_map[cost] = conf

            # 转回列表并按 cost 排序 (DP 算法通常喜欢有序输入)
            pruned_configs = sorted(list(best_configs_map.values()), key=lambda x: x.cost)

            return pruned_configs

    def _can_use_fast_additive_configurations(self) -> bool:
        """
        Exact fast path for the main binary rank-8 matrix scheduler.

        It is valid only when the transition utility is additive by matrix and
        the selected-matrix dynamic memory term is constant. Optional coherence
        or coverage bonuses make the objective depend on the existing partial
        assignment, so those cases must keep the general DP.
        """
        if self._use_cifar100_qkv_bundles():
            return False
        if float(getattr(self.args, "marginal_memory_penalty", 0.0)) > 0.0:
            return False
        if float(getattr(self.args, "marginal_latency_penalty", 0.0)) > 0.0:
            return False
        if bool(getattr(self.args, "use_block_coherence", False)) and (
            float(getattr(self.args, "block_coherence_weight", 0.0)) > 0.0
            or float(getattr(self.args, "qkv_completeness_weight", 0.0)) > 0.0
        ):
            return False
        if (
            getattr(self.args, "dataset", "") == "cifar100"
            and getattr(self, "current_allocation_strategy", "") == "Random_FIM"
            and getattr(self.args, "cifar100_prior_mode", "legacy_prior") == "legacy_prior"
        ):
            return False

        dynamic_terms = [
            self.zeta * self.phi_theta[j] + self.eta * self.sigma_d[j]
            for j in range(self.total_layers)
        ]
        if not dynamic_terms:
            return False
        if not np.allclose(dynamic_terms, dynamic_terms[0]):
            return False

        if getattr(self.args, "memory_model", "position") == "additive":
            static_terms = [self.eta * self.sigma_s[j] for j in range(self.total_layers)]
            if static_terms and not np.allclose(static_terms, static_terms[0]):
                return False
        return True

    def _fast_binary_memory(self, count: int, min_idx: int, dynamic_per_matrix: float, static_suffix) -> float:
        base_memory_bytes = self.C_p + self.C_c
        if getattr(self.args, "memory_model", "position") == "additive":
            static_bytes = count * self.eta * self.sigma_s[0]
        else:
            static_bytes = static_suffix[min_idx]
        total_dynamic_part_bytes = 24 * (count * dynamic_per_matrix + static_bytes)
        return (base_memory_bytes + total_dynamic_part_bytes) / (1024 ** 3)

    def _generate_fast_additive_configurations(self, T_deadline: float) -> list:
        """
        Exact O(L*C) configuration generation for the main additive scheduler.

        For a fixed earliest selected matrix m and selected count c, the memory
        is fixed by (m, c). The best value is therefore matrix m plus the top
        c-1 scores in the suffix m+1..L-1.
        """
        print(f"  [客户端 {self.client_id}] 使用快速可加配置生成... (内存上限: {self.M_max:.2f}GB, 时间上限: {T_deadline:.2f}s)")

        v = self._sanitize_scores(self.scores)
        c_time_max = int(T_deadline // self.T_layer_avg)
        c_max_global = min(self.L_max, c_time_max, self.total_layers)
        transition_checks = 0
        memory_rejects = 0
        print(f"  [客户端 {self.client_id}] 可分配矩阵上限: {c_max_global}, 延迟允许矩阵上限: {c_time_max}, 硬件预设上限：{self.L_max}")

        if c_max_global < self.L_min or c_max_global <= 0:
            self.last_config_generation_summary = {
                "generated_configs": 0,
                "pruned_configs": 0,
                "memory_rejects": 0,
                "transition_checks": 0,
                "c_time_max": c_time_max,
                "c_max_global": c_max_global,
                "fast_additive": True,
            }
            print(f"  [客户端 {self.client_id}] 生成了 0 个快速配置。")
            return []

        dynamic_per_matrix = self.zeta * self.phi_theta[0] + self.eta * self.sigma_d[0]
        static_suffix = [0.0] * (self.total_layers + 1)
        for idx in range(self.total_layers - 1, -1, -1):
            static_suffix[idx] = static_suffix[idx + 1] + self.eta * self.sigma_s[idx]

        best_by_cost = {}
        top_suffix_indices = []
        max_suffix = max(0, c_max_global - 1)

        def insert_top_suffix(idx):
            if max_suffix <= 0:
                return
            idx_score = float(v[idx])
            pos = 0
            while pos < len(top_suffix_indices):
                other = top_suffix_indices[pos]
                other_score = float(v[other])
                if other_score < idx_score or (other_score == idx_score and other > idx):
                    break
                pos += 1
            top_suffix_indices.insert(pos, idx)
            if len(top_suffix_indices) > max_suffix:
                top_suffix_indices.pop()

        for min_idx in range(self.total_layers - 1, -1, -1):
            selected_indices = [min_idx]
            value = float(v[min_idx])

            for count in range(1, c_max_global + 1):
                if count > 1:
                    suffix_pos = count - 2
                    if suffix_pos >= len(top_suffix_indices):
                        break
                    next_idx = top_suffix_indices[suffix_pos]
                    selected_indices.append(next_idx)
                    value += float(v[next_idx])

                if count < self.L_min:
                    continue

                transition_checks += 1
                mem_needed = self._fast_binary_memory(count, min_idx, dynamic_per_matrix, static_suffix)
                if mem_needed > self.M_max:
                    memory_rejects += 1
                    continue

                previous = best_by_cost.get(count)
                if previous is None or value > previous[0]:
                    best_by_cost[count] = (value, tuple(selected_indices), mem_needed)

            insert_top_suffix(min_idx)

        configs = []
        for count in sorted(best_by_cost):
            value, selected_indices, mem_needed = best_by_cost[count]
            assignment = [0] * self.total_layers
            for idx in selected_indices:
                assignment[idx] = 1
            _, _, estimated_latency = self.calculate_latency_components(count)
            configs.append(Configuration(
                client_id=self.client_id,
                value=value,
                cost=float(count),
                delay=estimated_latency,
                memory=mem_needed,
                assignment=tuple(assignment)
            ))

        self.last_config_generation_summary = {
            "generated_configs": len(configs),
            "pruned_configs": len(configs),
            "memory_rejects": memory_rejects,
            "transition_checks": transition_checks,
            "c_time_max": c_time_max,
            "c_max_global": c_max_global,
            "fast_additive": True,
        }
        print(f"  [客户端 {self.client_id}] 生成了 {len(configs)} 个快速配置。")
        return configs

    def generate_configurations(self, T_deadline: float) -> list:
        """
        核心功能：基于位置敏感的Knapsack动态规划，生成所有可行的配置。
        该实现正确处理了内存约束，其中内存计算需要完整的配置列表。
        """
        if self._use_cifar100_qkv_bundles():
            return self._generate_cifar100_qkv_bundle_configurations(T_deadline)
        if self._can_use_fast_additive_configurations():
            return self._generate_fast_additive_configurations(T_deadline)

        print(f"  [客户端 {self.client_id}] 开始生成配置... (内存上限: {self.M_max:.2f}GB, 时间上限: {T_deadline:.2f}s)")

        v = self._sanitize_scores(self.scores)
        memory_penalty_weight = float(getattr(self.args, "marginal_memory_penalty", 0.0))
        latency_penalty_weight = float(getattr(self.args, "marginal_latency_penalty", 0.0))
        use_marginal_penalty = memory_penalty_weight > 0.0 or latency_penalty_weight > 0.0
        use_block_coherence = bool(getattr(self.args, "use_block_coherence", False)) and float(
            getattr(self.args, "block_coherence_weight", 0.0)
        ) > 0.0

        c_time_max = int(T_deadline // self.T_layer_avg)
        c_max_global = min(self.L_max, c_time_max)
        transition_checks = 0
        memory_rejects = 0
        print(f"  [客户端 {self.client_id}] 可分配矩阵上限: {c_max_global}, 延迟允许矩阵上限: {c_time_max}, 硬件预设上限：{self.L_max}")

        # --- 步骤 1: DP 状态定义和初始化 ---
        dp_shape = (self.total_layers + 1, c_max_global + 1, self.total_layers + 2)

        ## MODIFIED: 修复 ValueError
        ## REASON: np.full 在 dtype=object 时对元组 fill_value 的广播行为有问题。
        ##         正确的做法是先创建空数组，再用 .fill() 方法填充。
        dp = np.empty(dp_shape, dtype=object)
        dp.fill((-np.inf, None))

        # 基础状态：考虑了0层，选了0层，价值为0，配置为空
        # 使用 tuple 存储 assignment，因为它是不可变的
        initial_assignment = tuple([0] * self.total_layers)
        dp[0, 0, self.total_layers] = (0, initial_assignment)

        # --- 步骤 2: DP 转移方程 ---
        for j in range(1, self.total_layers + 1):  # 迭代层 1..l
            layer_idx = j - 1  # 实际的层索引是 0-indexed

            # --- 步骤 1: 继承 (不选择层 j) ---
            # 首先，假设我们不选择第 j 层，那么所有状态都从 j-1 继承而来
            dp[j] = dp[j - 1].copy()

            # --- 步骤 2: 更新 (选择层 j) ---
            # 现在，我们尝试选择第 j 层，看是否能得到更好的结果
            for c in range(1, c_max_global + 1):  # 选择 j，所以 c 至少为 1
                # 遍历所有可能的前序状态 (j-1, c-1, m_idx_old)
                for m_idx_old in range(self.total_layers + 2):
                    prev_val, prev_assignment = dp[j - 1, c - 1, m_idx_old]

                    # 如果前序状态不可达，则跳过
                    if prev_val == -np.inf:
                        continue

                    # 构造新的配置
                    new_assignment_list = list(prev_assignment)
                    new_assignment_list[layer_idx] = 1
                    new_assignment = tuple(new_assignment_list)

                    # --- 核心修正：在转移时检查内存约束 ---
                    mem_needed = self.calculate_memory(list(new_assignment))
                    transition_checks += 1
                    if mem_needed > self.M_max:
                        memory_rejects += 1
                        continue  # 内存超限，此路不通

                    # 确定新状态的 m_idx
                    # Case 1: j 是第一层被选择的 (前序状态是空集)
                    if m_idx_old == self.total_layers:
                        new_m_idx = layer_idx
                    # Case 2: j 不是第一层被选择的 (最小索引保持不变)
                    else:
                        new_m_idx = m_idx_old

                    transition_gain = float(v[layer_idx])
                    if use_marginal_penalty:
                        old_assignment_list = list(prev_assignment)
                        old_mem = self.calculate_memory(old_assignment_list)
                        _, _, old_latency = self.calculate_latency_components(old_assignment_list)
                        _, _, new_latency = self.calculate_latency_components(list(new_assignment))
                        delta_memory = max(mem_needed - old_mem, 0.0)
                        delta_latency = max(new_latency - old_latency, 0.0)
                        transition_gain -= memory_penalty_weight * delta_memory / max(self.M_max, 1e-8)
                        transition_gain -= latency_penalty_weight * delta_latency / max(T_deadline, 1e-8)

                    if use_block_coherence:
                        transition_gain += self._block_coherence_bonus(list(prev_assignment), layer_idx)
                    transition_gain += self._cifar100_coverage_bonus(list(prev_assignment), layer_idx)

                    # 计算新价值
                    new_val = prev_val + transition_gain

                    # 如果新价值更优，则更新DP表
                    if new_val > dp[j, c, new_m_idx][0]:
                        dp[j, c, new_m_idx] = (new_val, new_assignment)

        # --- 步骤 3: 构造最终配置集合 K_i ---
        # 由于 assignment 已存储，不再需要回溯 (path)
        configs = []
        final_dp_slice = dp[self.total_layers]

        for c in range(self.L_min, c_max_global + 1):
            for m_idx in range(self.total_layers + 1):
                value, assignment = final_dp_slice[c, m_idx]

                if value > -np.inf:
                    # 再次确认最终配置的属性
                    final_mem = self.calculate_memory(list(assignment))

                    # [统一规范] 将总秩转换为“标准 rank=8 的模块数量”，作为成本(Cost)
                    # 因为在你的 DP 里 assignment 只有 0 和 1，所以这里算出来刚好等于激活的矩阵数
                    normalized_cost = sum((a if a > 1 else 8) for a in assignment if a > 0) / 8.0
                    num_active_matrices = sum(1 for a in assignment if a > 0)

                    # [精准修复] 直接传入 assignment 列表，让底层函数自己解析秩和维度
                    _, _, estimated_latency = self.calculate_latency_components(list(assignment))

                    # 做一个安全检查 (L_min 和 c_max_global 的判断依据改为标准化成本或矩阵数)
                    if final_mem <= self.M_max and self.L_min <= num_active_matrices <= c_max_global:
                        conf = Configuration(
                            client_id=self.client_id,
                            value=value,
                            cost=normalized_cost, # 统一使用标准化 cost
                            delay=estimated_latency,
                            memory=final_mem,
                            assignment=assignment
                        )
                        configs.append(conf)

        # 去重，因为不同路径可能产生相同配置但价值不同，我们保留价值最高的
        # A dictionary comprehension is a pythonic way to do this
        best_configs = {}
        for c in configs:
            if c.assignment not in best_configs or c.value > best_configs[c.assignment].value:
                best_configs[c.assignment] = c

        unique_configs = list(best_configs.values())

        # print(f"  [客户端 {self.client_id}] 生成了 {len(unique_configs)} 个独特的有效配置。")
        final_configs = self._prune_configurations(unique_configs)
        self.last_config_generation_summary = {
            "generated_configs": len(unique_configs),
            "pruned_configs": len(final_configs),
            "memory_rejects": memory_rejects,
            "transition_checks": transition_checks,
            "c_time_max": c_time_max,
            "c_max_global": c_max_global,
        }
        print(f"  [客户端 {self.client_id}] 生成了 {len(final_configs)} 个有效配置。")
        return final_configs

    def train(self, config: Configuration, global_lora_weights: OrderedDict) -> OrderedDict:
        # [HPC 优化]
        num_workers = max(0, int(getattr(self.args, "dataloader_num_workers", 4)))
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": True,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = False
            loader_kwargs["prefetch_factor"] = 2

        # 1. 准备数据
        client_dataset = Subset(self.dataset, list(self.data_indices))
        dataloader = DataLoader(client_dataset, batch_size=self.args.local_bs, shuffle=True,
                                collate_fn=custom_collate_fn_for_generation,
                                **loader_kwargs
                                )

        # 2. 全量加载模型
        model_copy = copy.deepcopy(self.model)

        # 3. 筛选与解冻逻辑
        all_candidate_modules = find_lora_candidates(model_copy)

        target_module_names = [
            all_candidate_modules[i]
            for i, val in enumerate(config.assignment)
            if val > 0 and i < len(all_candidate_modules)  # <--- [修复 1] 改为 val > 0，完美兼容各种 Rank
        ]

        # [修复 2] 真正根据下发的配置构建异构 Rank 模型，而不是用 full_assignment!
        lora_model = apply_lora_to_model(model_copy, config.assignment).to(self.device)

        #[修复 3] 加载全局权重，极致鲁棒的张量对齐 (Tensor Alignment)
        try:
            local_state = lora_model.state_dict()
            for k, global_tensor in global_lora_weights.items():
                if k in local_state:
                    local_tensor = local_state[k]
                    if global_tensor.shape == local_tensor.shape:
                        local_tensor.copy_(global_tensor)
                    else:
                        # [极致鲁棒机制]：找到本地和全局在每个维度上的"最小公共尺寸"
                        common_shape = tuple(min(sg, sl) for sg, sl in zip(global_tensor.shape, local_tensor.shape))

                        # 生成切片索引
                        slices_g = tuple(slice(0, dim) for dim in common_shape)
                        slices_l = tuple(slice(0, dim) for dim in common_shape)

                        # 将全局张量的公共部分，精确覆盖到本地张量的对应位置
                        # 如果本地 Rank 更大(例如 9)，那么多出来的那 1 维将保持随机初始化 (这是 HeteroFL 的标准做法)
                        local_tensor[slices_l].copy_(global_tensor[slices_g])
        except Exception as e:
            print(f"  [Error] Client {self.client_id} 加载异构权重失败: {e}")

        for param in lora_model.parameters():
            param.requires_grad = False

        # [调试开关] 强制打印第一个客户端的详细信息
        force_debug = (self.client_id == 0)
        # if force_debug:
        #     print(f"\n  [Debug Client 0] --- 开始参数解冻检查 ---")
        #     print(f"  [Debug Client 0] 目标层 (前3个): {target_module_names[:3]}")

        # 再解冻特定层
        trainable_count = 0 # [诊断] 统计可训练参数
        total_count = 0

        for name, param in lora_model.named_parameters():
            total_count += param.numel()
            should_unfreeze = False

            # A. 解冻分类头 / 语言模型头 / Embedding
            # OPT/Llama 通常叫 lm_head, embed_tokens
            if any(k in name for k in ['classifier', 'score', 'modules_to_save', 'lm_head', 'embed_tokens', 'embed_out']):
                should_unfreeze = True
            # B. 解冻 LoRA 层
            elif "lora_" in name:
                for target in target_module_names:
                    if target in name:
                        should_unfreeze = True
                        # # [强制调试] 打印一次成功的匹配，证明逻辑通了
                        # if force_debug and "layers.0." in name and "q_proj" in name:
                        #     print(f"  [Debug Client 0] 成功匹配: Target='{target}' -> Param='{name}'")
                        break

            if should_unfreeze:
                param.requires_grad = True
                trainable_count += param.numel()

        # [强制调试] 打印参数统计
        if force_debug or trainable_count == 0:
            ratio = 100 * trainable_count / total_count if total_count > 0 else 0
            # print(f"  [Debug Client {self.client_id}] 可训练参数: {trainable_count} ({ratio:.4f}%)")
            if trainable_count == 0:
                print(f"  [严重警告] Client {self.client_id} 参数全冻结！检查 target_module_names!")

        # 重置陈旧度
        for i, assigned in enumerate(config.assignment):
            if assigned == 1:
                self.staleness[i] = 0

        try:
            # 4. 训练
            lora_model.train()
            optimizer = AdamW(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=self.args.lr)

            # [AMP 修改] 使用 bfloat16 彻底解决 DeBERTa 溢出问题
            use_amp = self.device.type == 'cuda'
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))
            # [新增调试容器]
            initial_loss = None
            final_loss = None

            # [核心加速修复] 限制本地训练的最大 Batch 数量
            # 防止大数据集(如 QQP)导致单轮训练无休止，同时防止严重的客户端漂移(Client Drift)
            max_local_steps = 50  # 建议设为 50 或 100

            for epoch in range(self.args.local_ep):
                for i, batch in enumerate(dataloader):
                    if i >= max_local_steps:
                        break  # 达到步数上限，直接提前结束本轮训练！

                    input_batch = {}
                    for k, v in batch.items():
                        if k in ['prompts', 'text_labels', 'text']:
                            continue # 跳过字符串
                        if isinstance(v, torch.Tensor):
                            input_batch[k] = v.to(self.device)
                        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                            input_batch[k] = torch.stack(v).to(self.device)

                        optimizer.zero_grad()

                    # [AMP 修改 2] 注入 dtype
                    with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                        outputs = lora_model(**input_batch)
                        loss = outputs.loss

                    # [记录 Loss]
                    current_loss_val = outputs.loss.item() # 记录原始 loss
                    if initial_loss is None:
                        initial_loss = current_loss_val
                    final_loss = current_loss_val

                    # [AMP 标准步骤]
                    scaler.scale(loss).backward()

                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

            # [新增调试打印] 训练结束后，打印 Loss 变化
            # 只在 loss 上升或者 Client 0 时打印，避免刷屏
            if final_loss > initial_loss or self.client_id == 0:
                print(f"  [Debug Client {self.client_id}] Local Loss: {initial_loss:.4f} -> {final_loss:.4f} (Delta: {final_loss - initial_loss:.4f})")
                if final_loss > initial_loss + 0.5:
                    print(f"  [严重警告] 客户端 {self.client_id} 本地训练发散！建议降低 LR。")

            # 5. 提取参数
            trained_weights = OrderedDict()
            for name, param in lora_model.named_parameters():
                if param.requires_grad:
                    trained_weights[name] = param.detach().cpu()

            return trained_weights
        except Exception as e:
            print(f"Client {self.client_id} Training Error: {e}")
            import traceback
            traceback.print_exc() # 打印堆栈以便排查
            return None
        finally:
            # [核心修改] 无论训练成功与否，强制清理显存
            del lora_model
            del model_copy
            if 'optimizer' in locals():
                del optimizer

            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def _build_uniform_matrix_assignment(self, budget_limit: float = float('inf'), shared_limit=None) -> list:
            """Select a uniformly spaced matrix subset under budget and memory limits."""
            max_matrices = min(self.total_layers, int(budget_limit))
            if shared_limit is not None:
                max_matrices = min(max_matrices, int(shared_limit))
            if max_matrices <= 0:
                return [0] * self.total_layers

            for count in range(max_matrices, 0, -1):
                if count >= self.total_layers:
                    indices = list(range(self.total_layers))
                else:
                    indices = sorted(set(np.linspace(0, self.total_layers - 1, count, dtype=int).tolist()))
                    cursor = 0
                    while len(indices) < count and cursor < self.total_layers:
                        if cursor not in indices:
                            indices.append(cursor)
                        cursor += 1
                    indices = sorted(indices[:count])

                assignment = [0] * self.total_layers
                for idx in indices:
                    assignment[idx] = 1
                if self.calculate_memory(assignment) <= self.M_max:
                    return assignment
            return [0] * self.total_layers

    def _build_uniform_physical_assignment(self, budget_limit: float = float('inf'), shared_limit=None) -> list:
            """Select uniformly spaced physical blocks, not individual matrices."""
            import collections

            layer_groups = collections.defaultdict(list)
            for idx, meta in enumerate(self.candidate_metadata[:self.total_layers]):
                layer_groups[int(meta.get("block_id", 0))].append(idx)
            sorted_layer_ids = sorted(layer_groups)
            if not sorted_layer_ids:
                return [0] * self.total_layers

            effective_limit = float(budget_limit)
            if shared_limit is not None:
                effective_limit = min(effective_limit, float(shared_limit))

            def cost(assign_list):
                return sum((a if a > 1 else 8) for a in assign_list if a > 0) / 8.0

            for count in range(len(sorted_layer_ids), 0, -1):
                if count >= len(sorted_layer_ids):
                    chosen_layers = list(sorted_layer_ids)
                else:
                    positions = sorted(set(np.linspace(0, len(sorted_layer_ids) - 1, count, dtype=int).tolist()))
                    cursor = 0
                    while len(positions) < count and cursor < len(sorted_layer_ids):
                        if cursor not in positions:
                            positions.append(cursor)
                        cursor += 1
                    chosen_layers = [sorted_layer_ids[pos] for pos in sorted(positions[:count])]

                assignment = [0] * self.total_layers
                for layer_id in chosen_layers:
                    for idx in layer_groups[layer_id]:
                        assignment[idx] = 1
                if cost(assignment) <= effective_limit and self.calculate_memory(assignment) <= self.M_max:
                    return assignment
            return [0] * self.total_layers

    def _build_prefix_physical_assignment(self, budget_limit: float = float('inf'), shared_limit=None) -> list:
            """Select a fixed prefix of physical blocks under budget and memory limits."""
            import collections

            layer_groups = collections.defaultdict(list)
            for idx, meta in enumerate(self.candidate_metadata[:self.total_layers]):
                layer_groups[int(meta.get("block_id", 0))].append(idx)
            sorted_layer_ids = sorted(layer_groups)
            if not sorted_layer_ids:
                return [0] * self.total_layers

            effective_limit = float(budget_limit)
            if shared_limit is not None:
                effective_limit = min(effective_limit, float(shared_limit))

            def cost(assign_list):
                return sum((a if a > 1 else 8) for a in assign_list if a > 0) / 8.0

            for count in range(len(sorted_layer_ids), 0, -1):
                chosen_layers = sorted_layer_ids[:count]
                assignment = [0] * self.total_layers
                for layer_id in chosen_layers:
                    for idx in layer_groups[layer_id]:
                        assignment[idx] = 1
                if cost(assignment) <= effective_limit and self.calculate_memory(assignment) <= self.M_max:
                    return assignment
            return [0] * self.total_layers

    def _build_random_matrix_assignment(self, budget_limit: float = float('inf')) -> list:
            """Randomly select LoRA matrices under budget and memory limits."""
            max_matrices = min(self.total_layers, int(budget_limit))
            if max_matrices <= 0:
                return [0] * self.total_layers

            seed = int(getattr(self.args, "seed", 1))
            round_k = int(getattr(self, "current_round", 0))
            rng = np.random.default_rng(seed + 1009 * round_k + 9176 * int(self.client_id))
            indices = list(range(self.total_layers))
            rng.shuffle(indices)

            assignment = [0] * self.total_layers
            chosen = 0
            for idx in indices:
                if chosen >= max_matrices:
                    break
                trial = assignment.copy()
                trial[idx] = 1
                if self.calculate_memory(trial) <= self.M_max:
                    assignment = trial
                    chosen += 1
            return assignment

    def generate_fixed_assignment_config(self, assignment: list, value=None) -> Configuration:
            """Wrap a server-provided assignment as a client configuration."""
            assignment = list(assignment)
            cost_normalized = sum((a if a > 1 else 8) for a in assignment if a > 0) / 8.0
            mem = self.calculate_memory(assignment)
            _, _, delay = self.calculate_latency_components(assignment)
            if mem > self.M_max:
                assignment = [0] * self.total_layers
                cost_normalized = 0.0
                mem = self.calculate_memory(assignment)
                delay = 0.0
            self.last_config_generation_summary = {
                "generated_configs": 1,
                "pruned_configs": 1 if cost_normalized > 0 else 0,
                "memory_rejects": 1 if cost_normalized == 0 else 0,
                "transition_checks": 1,
                "c_time_max": "",
                "c_max_global": "",
            }
            return Configuration(
                client_id=self.client_id,
                value=cost_normalized if value is None else value,
                cost=cost_normalized,
                delay=delay,
                memory=mem,
                assignment=tuple(assignment)
            )

    def generate_heuristic_config(self, strategy: str, budget_limit: float = float('inf')) -> Configuration:
            """
            根据指定的启发式策略生成 LoRA 层配置
            将物理层基线限制为基于 Layer-wise 的粗粒度分配。
            矩阵级方法使用 Matrix-wise 的候选分配。
            :param strategy: 'EL', 'MH', 'MS', 'FedRA'
            """
            from models.lora_utils import find_lora_candidates
            import re
            import collections

            # 初始化全 0 (都不选)
            assignment = [0] * self.total_layers
            memory_override = None
            delay_override = None

            # === 1. 解析物理层，将属于同一层的 LoRA 矩阵打包分组 ===
            candidate_modules = find_lora_candidates(self.model)
            layer_groups = collections.defaultdict(list)

            for idx, name in enumerate(candidate_modules):
                if idx >= self.total_layers: break # 防止越界
                # 匹配 HuggingFace 模型常见的物理层命名 (兼容 BERT, ViT, Llama, OPT, GPT)
                # 例如 "encoder.layer.0.attention.query", "model.decoder.layers.1.q_proj"
                match = re.search(r'(?:layers?|blocks?|h)\.(\d+)', name)
                if not match:
                    # 兼容后备正则匹配 (查找被点包围的数字)
                    match = re.search(r'\.(\d+)\.', name)

                layer_id = int(match.group(1)) if match else 0
                layer_groups[layer_id].append(idx)

            # 获得排序后的物理层 ID 列表 (例如:[0, 1, 2, ..., 11])
            sorted_layer_ids = sorted(list(layer_groups.keys()))

            # --- 辅助函数：计算当前 cost (模块数量) ---
            def get_current_cost(assign_list):
                return sum((a if a > 1 else 8) for a in assign_list if a > 0) / 8.0

            def build_depth_rank_assignment(rank_candidates, memory_fn):
                """Allocate higher ranks to deeper physical layers before spreading rank-2 everywhere."""
                rank_candidates = sorted(set(rank_candidates), reverse=True)
                if not rank_candidates:
                    return [0] * self.total_layers

                for rank in rank_candidates:
                    if rank < 8:
                        continue
                    candidate_assignment = [rank] * self.total_layers
                    if memory_fn(candidate_assignment) <= self.M_max and get_current_cost(candidate_assignment) <= budget_limit:
                        return candidate_assignment

                current_assignment = [0] * self.total_layers
                for rank in rank_candidates:
                    for layer_id in reversed(sorted_layer_ids):
                        indices = layer_groups[layer_id]
                        trial_assignment = list(current_assignment)
                        for idx in indices:
                            trial_assignment[idx] = max(trial_assignment[idx], rank)
                        if memory_fn(trial_assignment) <= self.M_max and get_current_cost(trial_assignment) <= budget_limit:
                            current_assignment = trial_assignment

                return current_assignment

            # === 2. 启发式策略逻辑 (以"物理层"为最小分配单元) ===
                    # 2. 策略分发
            if strategy == 'FedAvg_LoRA_homogeneous' or strategy == 'FedAvg-LoRA':
                full_assignment = [1] * self.total_layers
                if getattr(self.args, "dataset", "") == "cifar100":
                    assignment = self._build_random_matrix_assignment(budget_limit)
                elif self.calculate_memory(full_assignment) <= self.M_max and get_current_cost(full_assignment) <= budget_limit:
                    assignment = full_assignment
                elif getattr(self.args, "baseline_granularity", "matrix") == "physical":
                    if getattr(self.args, "fedavg_physical_policy", "uniform") == "prefix":
                        assignment = self._build_prefix_physical_assignment(budget_limit)
                    else:
                        assignment = self._build_uniform_physical_assignment(budget_limit)
                else:
                    assignment = self._build_uniform_matrix_assignment(budget_limit)

            elif strategy == 'Uniform_allocation':
                if getattr(self.args, "dataset", "") == "cifar100":
                    assignment = self._build_uniform_matrix_assignment(budget_limit)
                elif getattr(self.args, "baseline_granularity", "matrix") == "physical":
                    assignment = self._build_uniform_physical_assignment(budget_limit)
                else:
                    assignment = self._build_uniform_matrix_assignment(budget_limit)

            elif strategy in ['Physical_FIM', 'Physical_Taylor']:
                layer_values = {}
                for layer_id, indices in layer_groups.items():
                    layer_values[layer_id] = sum(self.scores[idx] for idx in indices if idx < len(self.scores))
                ranked_layer_ids = sorted(layer_values, key=layer_values.get, reverse=True)

                current_assignment = [0] * self.total_layers
                for layer_id in ranked_layer_ids:
                    indices = layer_groups[layer_id]
                    for idx in indices:
                        current_assignment[idx] = 1
                    if self.calculate_memory(current_assignment) > self.M_max or get_current_cost(current_assignment) > budget_limit:
                        for idx in indices:
                            current_assignment[idx] = 0
                        continue
                assignment = current_assignment

            elif strategy == 'Fed-HeLLo':
                # 1. 物理层级 FIM 聚合
                layer_fim = {}
                for layer_id, indices in layer_groups.items():
                    layer_fim[layer_id] = sum(self.scores[idx] for idx in indices if idx < len(self.scores))

                anchor_count = max(0, int(getattr(self.args, "fedhello_anchor_layers", 0)))
                anchor_layer_ids = sorted_layer_ids[:anchor_count]
                anchor_layer_set = set(anchor_layer_ids)
                current_assignment = [0] * self.total_layers
                for layer_id in anchor_layer_ids:
                    trial_assignment = list(current_assignment)
                    for idx in layer_groups[layer_id]:
                        trial_assignment[idx] = 1
                    if (
                        self.calculate_memory(trial_assignment) <= self.M_max
                        and get_current_cost(trial_assignment) <= budget_limit
                    ):
                        current_assignment = trial_assignment
                    else:
                        break

                layer_ids = [
                    layer_id for layer_id in layer_groups.keys()
                    if layer_id not in anchor_layer_set
                ]
                fim_values = np.array([layer_fim[lid] for lid in layer_ids])

                # 2. 概率分布构建 (Selection Probability)
                # 步骤 A: 线性归一化 (Min-Max Scaling) 防止 Softmax 溢出
                if np.max(fim_values) > np.min(fim_values):
                    fim_norm = (fim_values - np.min(fim_values)) / (np.max(fim_values) - np.min(fim_values))
                else:
                    fim_norm = np.zeros_like(fim_values)

                # 步骤 B: 引入温度系数的 Softmax (Temperature Scaling)
                tau = 1.0  # 温度参数: tau 越小越贪婪(Exploitation)，越大越平均(Exploration)
                exp_preds = np.exp(fim_norm / tau)
                fim_probs = exp_preds / np.sum(exp_preds) # 确保 sum(p_j) = 1

                # 3. 异构采样逻辑 (Heterogeneous Sampling)
                # 根据构建的概率分布 P，无放回地随机抽取物理层
                sampled_layer_ids = np.random.choice(
                    layer_ids,
                    size=len(layer_ids),
                    replace=False,
                    p=fim_probs
                )

                # 4. 在可行锚定层之后按原规则填充剩余预算
                for layer_id in sampled_layer_ids:
                    indices = layer_groups[layer_id]
                    # 尝试将抽中的层在掩码中设为 1
                    for idx in indices: current_assignment[idx] = 1

                    # 检查客户端资源能力 c_i (内存与公平均摊预算)
                    if self.calculate_memory(current_assignment) > self.M_max or get_current_cost(current_assignment) > budget_limit:
                        # 资源不足，将该层回退为 0
                        for idx in indices: current_assignment[idx] = 0
                        # Fed-HeLLo 允许在装不下大层时，继续遍历尝试装入后续被抽中的较小层
                        continue

                assignment = current_assignment

            elif strategy == 'LEGEND':
                # ==========================================================
                # LEGEND 算法 (2025 SOTA) 严格实现
                # 核心1: 基于设备能力确定最大微调深度 k_i
                # 核心2: 全局秩采用等差数列 (Inc 分布，越深秩越高)
                # ==========================================================

                current_assignment = [0] * self.total_layers

                # 2. 秩分布策略 (Inc Distribution)
                # 论文设定：越靠近输出的深层，Rank 越大。
                # 使用等差数列，公差 lambda = 2。最深层赋予 Rank=16。
                current_rank = 10
                lambda_step = 1

                # 从深层(靠后)往前倒推分配
                for layer_id in reversed(sorted_layer_ids):
                    indices = layer_groups[layer_id]
                    for idx in indices:
                        current_assignment[idx] = current_rank

                    # 硬件约束安全检查
                    if self.calculate_memory(current_assignment) > self.M_max or get_current_cost(current_assignment) > budget_limit:
                        for idx in indices:
                            current_assignment[idx] = 0
                        break

                    # 根据等差数列，向浅层递减秩大小，保底秩为 2
                    current_rank = max(4, current_rank - lambda_step)

                assignment = current_assignment

            elif strategy == 'HETLoRA':
                # HETLoRA-style baseline: assign a client-specific LoRA rank
                # to the same candidate set. Value 1 is reserved in this
                # codebase for default rank-8 selection, so the runnable rank
                # tiers use explicit values >= 2.
                try:
                    mem_levels = [
                        float(item)
                        for item in str(getattr(self.args, "mem_levels_gb", "")).split(":")
                        if item.strip()
                    ]
                except ValueError:
                    mem_levels = []
                if mem_levels:
                    mem_min = min(mem_levels)
                    mem_max = max(mem_levels)
                    memory_ratio = (float(self.M_max) - mem_min) / max(mem_max - mem_min, 1e-8)
                else:
                    memory_ratio = 0.0

                if memory_ratio >= 0.80:
                    preferred_rank = 16
                elif memory_ratio >= 0.55:
                    preferred_rank = 8
                elif memory_ratio >= 0.30:
                    preferred_rank = 4
                else:
                    preferred_rank = 2

                rank_candidates = [rank for rank in [preferred_rank, 8, 4, 2] if rank <= preferred_rank]
                current_assignment = build_depth_rank_assignment(rank_candidates, self.calculate_memory)

                assignment = current_assignment

            elif strategy == 'FAH_QLoRA':
                # FAH-QLoRA-style proxy baseline. It reproduces the scheduling
                # axis of FAH-QLoRA inside this codebase: heterogeneous
                # quantization-aware memory/latency estimates plus dynamic and
                # client-conditioned LoRA rank assignment. The actual frozen
                # base model remains the normal HF model because per-client
                # q-bit kernels are not part of the current training backend.
                preferred_rank = self._fah_rank_for_round()
                rank_candidates = sorted({rank for rank in [preferred_rank, 8, 4, 2] if rank <= preferred_rank}, reverse=True)
                current_assignment = build_depth_rank_assignment(rank_candidates, self.calculate_fah_memory)

                assignment = current_assignment
                memory_override = self.calculate_fah_memory(assignment)
                _, _, delay_override = self.calculate_fah_latency_components(assignment)

            elif strategy == 'EL':
                # Baseline-Exclusive Learning: 尝试全选
                all_ones = [1] * self.total_layers
                mem_needed = self.calculate_memory(all_ones)
                if mem_needed <= self.M_max and get_current_cost(all_ones) <= budget_limit:
                    assignment = all_ones
                else:
                    assignment = [0] * self.total_layers

            elif strategy == 'MH':
                # Baseline-Memory Hogger: 优先分配浅层的物理层 (Layer 0, 1, 2...)
                current_assignment = [0] * self.total_layers
                for layer_id in sorted_layer_ids:
                    indices = layer_groups[layer_id]
                    # 尝试打包激活该物理层下的【所有】 LoRA 矩阵
                    for idx in indices:
                        current_assignment[idx] = 1

                    # 检查加完这一整个物理层后，内存是否超限
                    #[核心修改] 同时检查内存和公平预算
                    if self.calculate_memory(current_assignment) > self.M_max or get_current_cost(current_assignment) > budget_limit:
                        for idx in indices: current_assignment[idx] = 0
                        break
                assignment = current_assignment

            elif strategy == 'MS':
                # Baseline-Memory Saver: 优先分配深层的物理层 (Layer N-1, N-2...)
                current_assignment =[0] * self.total_layers
                for layer_id in reversed(sorted_layer_ids):
                    indices = layer_groups[layer_id]
                    # 尝试打包激活该物理层下的【所有】 LoRA 矩阵
                    for idx in indices:
                        current_assignment[idx] = 1

                    # [核心修改] 同时检查内存和公平预算
                    if self.calculate_memory(current_assignment) > self.M_max or get_current_cost(current_assignment) > budget_limit:
                        for idx in indices: current_assignment[idx] = 0
                        break
                assignment = current_assignment

            elif strategy == 'FedRA':
                # FedRA: 随机选择物理层进行分配
                shuffled_layer_ids = list(sorted_layer_ids)
                np.random.shuffle(shuffled_layer_ids)

                current_assignment = [0] * self.total_layers
                for layer_id in shuffled_layer_ids:
                    indices = layer_groups[layer_id]
                    # 尝试打包激活该物理层下的【所有】 LoRA 矩阵
                    for idx in indices:
                        current_assignment[idx] = 1

                    if self.calculate_memory(current_assignment) > self.M_max or get_current_cost(current_assignment) > budget_limit:
                        for idx in indices: current_assignment[idx] = 0
                        continue # FedRA 允许跳过装不下的，看后面的小层
                assignment = current_assignment

            else:
                raise ValueError(f"未知的策略: {strategy}")

            # === 3. 规范化 Cost 计算 ===
            # 将总秩转换为“标准 rank=8 的模块数量”，实现全算法预算衡量的统一
            cost_normalized = sum((a if a > 1 else 8) for a in assignment if a > 0) / 8.0
            mem = memory_override if memory_override is not None else self.calculate_memory(assignment)
            if delay_override is not None:
                accurate_delay = delay_override
            else:
                _, _, accurate_delay = self.calculate_latency_components(assignment)
            self.last_config_generation_summary = {
                "generated_configs": 1,
                "pruned_configs": 1 if cost_normalized > 0 else 0,
                "memory_rejects": 1 if cost_normalized == 0 else 0,
                "transition_checks": 1,
                "c_time_max": "",
                "c_max_global": "",
            }

            return Configuration(
                client_id=self.client_id,
                value=cost_normalized,
                cost=cost_normalized,
                delay=accurate_delay,
                memory=mem,
                assignment=tuple(assignment)
            )

    def calculate_local_gradients(self, global_lora_weights, scoring_method='taylor', apply_depth_scaling=True):
        """
        [全面升级] 利用本地数据计算每一层 LoRA 矩阵的重要性。
        支持 'taylor' (泰勒展开) 和 'fim' (Fisher信息矩阵)。
        """
        # 1. 临时构建全量 LoRA 模型
        from models.lora_utils import apply_lora_to_model, find_lora_candidates

        model_copy = copy.deepcopy(self.model)
        full_assignment = [1] * self.total_layers
        probe_model = apply_lora_to_model(model_copy, full_assignment)

        # 加载当前的全局权重
        probe_model.load_state_dict(global_lora_weights, strict=False)
        probe_model.to(self.device)
        probe_model.eval() # 保持 Eval 模式，但开启梯度

        # 激活梯度
        for n, p in probe_model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # 2. 获取一个本地 Batch
        temp_dataset = Subset(self.dataset, list(self.data_indices)[:self.args.local_bs])
        temp_loader = DataLoader(temp_dataset, batch_size=self.args.local_bs,
                                 shuffle=False,
                                 collate_fn=custom_collate_fn_for_generation)

        try:
            batch = next(iter(temp_loader))
        except StopIteration:
            return np.zeros(self.total_layers) # 无数据情况

        # 数据移动与过滤
        input_batch = self._prepare_input_batch(batch)
        if 'input_ids' not in input_batch and 'pixel_values' not in input_batch:
            return np.zeros(self.total_layers)

        # 3. 前向与反向传播
        probe_model.zero_grad()
        outputs = probe_model(**input_batch)
        loss = outputs.loss
        loss.backward()

        # 4. [核心升级] 聚合重要性评分
        candidate_modules = self.candidate_modules or find_lora_candidates(self.model)
        layer_scores = np.zeros(self.total_layers)

        for name, param in probe_model.named_parameters():
            if "lora_" not in name or param.grad is None:
                continue

            for layer_idx, module_name in enumerate(candidate_modules):
                if module_name in name:
                    # ==========================================
                    # 评分策略计算核心
                    # ==========================================
                    if scoring_method == 'taylor':
                        # 方法一：泰勒一阶展开 (Taylor Score)
                        # |W * dW| 的绝对值求和，并加入小的 FIM 稳定项。
                        # LoRA 常见近零初始化会让纯 Taylor 在早期轮次退化为近零分数。
                        taylor_score = (param.detach() * param.grad.detach()).abs().sum().item()
                        fim_score = param.grad.detach().pow(2).sum().item()
                        score = taylor_score + float(getattr(self.args, "taylor_fim_blend", 0.05)) * fim_score
                    elif scoring_method == 'fim':
                        # 方法二：Fisher 信息矩阵对角近似 (FIM-HLA)
                        # [核心修正] 严格对齐论文：梯度的 L2 范数的平方。
                        # 必须使用 .sum() 而不是 .mean()，以反映整个矩阵对模型梯度的总贡献量。
                        score = param.grad.detach().pow(2).sum().item()
                    else:
                        score = param.grad.data.norm(2).item()

                    if apply_depth_scaling:
                        is_sentence_pair = self.args.dataset in ['qqp', 'mnli', 'qnli']
                        depth_power = 1.0
                        if is_sentence_pair:
                            depth_power += 1.5
                        depth_power += self._depth_power_extra()
                        normalized_depth = layer_idx / max(1, self.total_layers - 1)
                        depth_weight = (1.0 + normalized_depth) ** depth_power
                        score = score * depth_weight

                    layer_scores[layer_idx] += score
                    break

        # 5. 立即清理显存
        probe_model.zero_grad()
        del probe_model, model_copy, batch, input_batch, outputs, loss
        torch.cuda.empty_cache()

        return layer_scores

    def update_scores(self, global_lora_weights, scoring_method='fim', apply_depth_scaling=True):
        """
        [解除封印版] 完全基于全局绝对梯度的重要性评分。
        直接反映哪个客户端的哪个矩阵最需要被更新！
        """
        # 1. 计算原始绝对评分 (Taylor 或 FIM)
        calibration_mode = self._position_calibration_mode()
        if calibration_mode == "fixed_deep":
            local_depth_scaling = True
        elif calibration_mode in {"none", "adaptive"}:
            local_depth_scaling = False
        else:
            local_depth_scaling = apply_depth_scaling and not self._utility_mode_enabled()
        raw_scores = self.calculate_local_gradients(
            global_lora_weights,
            scoring_method,
            apply_depth_scaling=local_depth_scaling,
        )
        raw_scores = self._sanitize_scores(raw_scores)
        self.last_raw_scores = raw_scores.copy()

        if getattr(self.args, "use_score_ema", False):
            ema_scores = self._stable_ema_scores(raw_scores)
        else:
            ema_scores = raw_scores.copy()
            if self.score_ema_mean is None or len(ema_scores) != self.total_layers:
                self.score_ema_mean = ema_scores.copy()
                self.score_ema_sq_mean = np.square(ema_scores)
                self.score_ema_count = 1
        self.last_ema_scores = self._sanitize_scores(ema_scores).copy()

        if self._utility_mode_enabled():
            self.scores = self.last_ema_scores.copy()
        else:
            # 旧行为：保留 calculate_local_gradients 已经产生的 depth-scaled raw score。
            self.scores = raw_scores

    def calculate_latency_components(self, assignment):
        """
        [Rank感知版] 精确计算延迟分量
        完美兼容 LEGEND 的异构秩 (Rank)。通信与计算时间均随 Rank 线性缩放。
        """
        # 向后兼容标量输入 (以防某些代码传了 cost 整数进来)
        if isinstance(assignment, (int, float)):
            num_active_matrices = int(assignment)
            total_rank = num_active_matrices * 8
        else:
            # 提取整个模型的总 Rank 数量
            total_rank = sum((a if a > 1 else 8) for a in assignment if a > 0)

        if total_rank == 0:
            return 0.0, 0.0, 0.0

        # 1. 计算延迟 T_comp (计算量与 Rank 严格成正比)
        # self.comp_time_per_layer_total 是基于 Rank=8 的基准测量值
        t_comp = self.comp_time_per_layer_total * (total_rank / 8.0)

        # 2. 通信延迟 T_comm (参数量与 Rank 严格成正比)
        # self.lora_layer_size_mb 是 Rank=8 的单矩阵基准大小
        total_data_mb = (total_rank / 8.0) * self.lora_layer_size_mb * 2  # 乘以 2 代表上下行双向
        t_comm = total_data_mb / self.bandwidth

        t_total = t_comp + t_comm
        return t_comp, t_comm, t_total
