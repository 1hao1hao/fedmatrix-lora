# 文件: models/server.py
"""Federated server implementation."""

import collections
import concurrent.futures
import copy
import csv
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.lora_utils import (
    apply_lora_to_model,
    average_lora_weights,
    get_lora_state_dict,
    find_lora_candidates,
    infer_lora_candidate_metadata,
)
from models.metrics import evaluate_accuracy, evaluate_macro_f1, evaluate_model_loss, evaluate_rouge_l
from models.scheduling import SchedulingMixin
from utils.data_utils import custom_collate_fn_for_generation


class Server(SchedulingMixin):
    def __init__(self, clients, total_layers, lambda_prime, global_model, val_loader, small_val_loader,val_text_labels, tokenizer,args):
        self.clients = {c.client_id: c for c in clients}
        self.lambda_prime = lambda_prime
        self.round_k = 0
        self.total_layers = total_layers
        self.args = args

        # --- 核心修改：分离模型架构和权重 ---
        # 1. 冻结并保存基座模型 (永远不变)
        self.pristine_model = copy.deepcopy(global_model)
        for param in self.pristine_model.parameters():
            param.requires_grad = False # 确保基座不参与任何梯度计算
        self.pristine_model.to('cpu')

        # 2. 初始化全局 LoRA 权重 (这是我们要更新的东西)
        # 我们需要先创建一个临时的 LoRA 模型来获取初始的 state_dict 结构
        from models.lora_utils import apply_lora_to_model, get_lora_state_dict
        # 假设初始所有层都开启 LoRA (或者全0初始化)
        # 这里为了获得字典结构，我们临时应用一次全量 LoRA
        initial_assignment = [1] * total_layers
        temp_lora_model = apply_lora_to_model(copy.deepcopy(self.pristine_model), initial_assignment)

        # 保存初始的全局 LoRA 权重
        self.global_lora_weights = get_lora_state_dict(temp_lora_model)

        # 清理内存
        del temp_lora_model

        # 2. 只维护和更新全局模型的权重字典 (state_dict)
        self.global_model_weights = self.pristine_model.state_dict()
        # 存储验证集加载器和设备信息
        self.val_loader = val_loader
        self.small_val_loader = small_val_loader # <--- [核心修复] 接收并保存
        self.val_text_labels = val_text_labels
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.tokenizer = tokenizer

        self.args = args
        self.candidate_modules = find_lora_candidates(self.pristine_model)[:self.total_layers]
        self.candidate_metadata = [
            infer_lora_candidate_metadata(name) for name in self.candidate_modules
        ]
        self.trace_enabled = bool(getattr(args, "enable_trace", False))
        self.log_selection_analysis = bool(getattr(args, "log_selection_analysis", False))
        self.log_update_cosine = bool(getattr(args, "log_update_cosine", False))
        run_id = getattr(args, "run_id", "unknown_run")
        default_trace_dir = os.path.join("evaluation_results", args.dataset, run_id, "traces")
        self.trace_dir = getattr(args, "trace_dir", "") or default_trace_dir
        self.current_agreement_scores = np.zeros(self.total_layers, dtype=float)
        self.client_score_component_cache = {}
        if self.trace_enabled or self.log_selection_analysis or self.log_update_cosine:
            os.makedirs(self.trace_dir, exist_ok=True)

    def _append_trace_rows(self, trace_name, fieldnames, rows):
        if not self.trace_enabled or not rows:
            return
        path = os.path.join(self.trace_dir, f"{trace_name}.csv")
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def _append_analysis_rows(self, trace_name, fieldnames, rows, enabled):
        if not enabled or not rows:
            return
        os.makedirs(self.trace_dir, exist_ok=True)
        path = os.path.join(self.trace_dir, f"{trace_name}.csv")
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def _assignment_summary(self, assignment):
        active = [idx for idx, value in enumerate(assignment) if value > 0]
        return {
            "selected_count": len(active),
            "active_indices": " ".join(str(idx) for idx in active),
            "earliest_index": min(active) if active else "",
            "latest_index": max(active) if active else "",
        }

    def _write_score_trace(self, allocation_strategy, selected_clients, score_source, scoring_method="", global_max=""):
        rows = []
        for client in selected_clients:
            raw_scores = self.client_raw_scores_cache.get(client.client_id, None)
            for matrix_idx in range(self.total_layers):
                rows.append({
                    "experiment_id": getattr(self.args, "experiment_id", ""),
                    "run_id": getattr(self.args, "run_id", ""),
                    "dataset": getattr(self.args, "dataset", ""),
                    "distribution": getattr(self.args, "distribution", ""),
                    "method": allocation_strategy,
                    "round": self.round_k,
                    "client_id": client.client_id,
                    "matrix_index": matrix_idx,
                    "module_name": self.candidate_modules[matrix_idx] if matrix_idx < len(self.candidate_modules) else "",
                    "depth_norm": matrix_idx / max(1, self.total_layers - 1),
                    "score_source": score_source,
                    "scoring_method": scoring_method,
                    "raw_score": raw_scores[matrix_idx] if raw_scores is not None and matrix_idx < len(raw_scores) else "",
                    "normalized_score": client.scores[matrix_idx] if hasattr(client, "scores") and matrix_idx < len(client.scores) else "",
                    "global_max": global_max,
                })
        self._append_trace_rows(
            f"score_trace_{allocation_strategy}",
            [
                "experiment_id", "run_id", "dataset", "distribution", "method", "round",
                "client_id", "matrix_index", "module_name", "depth_norm",
                "score_source", "scoring_method", "raw_score", "normalized_score", "global_max",
            ],
            rows,
        )

    def _write_memory_trace(self, allocation_strategy, selected_clients, configs_per_client):
        rows = []
        for client in selected_clients:
            summary = getattr(client, "last_config_generation_summary", {}).copy()
            rows.append({
                "experiment_id": getattr(self.args, "experiment_id", ""),
                "run_id": getattr(self.args, "run_id", ""),
                "dataset": getattr(self.args, "dataset", ""),
                "distribution": getattr(self.args, "distribution", ""),
                "method": allocation_strategy,
                "round": self.round_k,
                "client_id": client.client_id,
                "memory_model": getattr(self.args, "memory_model", "position"),
                "M_max": client.M_max,
                "T_layer_avg": client.T_layer_avg,
                "generated_configs": summary.get("generated_configs", ""),
                "pruned_configs": summary.get("pruned_configs", ""),
                "memory_rejects": summary.get("memory_rejects", ""),
                "transition_checks": summary.get("transition_checks", ""),
                "c_time_max": summary.get("c_time_max", ""),
                "c_max_global": summary.get("c_max_global", ""),
                "configs_seen_by_server": configs_per_client.get(client.client_id, ""),
            })
        self._append_trace_rows(
            f"memory_trace_{allocation_strategy}",
            [
                "experiment_id", "run_id", "dataset", "distribution", "method", "round",
                "client_id", "memory_model", "M_max", "T_layer_avg",
                "generated_configs", "pruned_configs", "memory_rejects", "transition_checks",
                "c_time_max", "c_max_global", "configs_seen_by_server",
            ],
            rows,
        )

    def _write_scheduler_trace(self, allocation_strategy, all_configs, final_allocation, C_k, T_actual):
        selected_keys = {(c.client_id, c.assignment) for c in final_allocation}
        rows = []
        for config_idx, config in enumerate(all_configs):
            summary = self._assignment_summary(config.assignment)
            rows.append({
                "experiment_id": getattr(self.args, "experiment_id", ""),
                "run_id": getattr(self.args, "run_id", ""),
                "dataset": getattr(self.args, "dataset", ""),
                "distribution": getattr(self.args, "distribution", ""),
                "method": allocation_strategy,
                "round": self.round_k,
                "client_id": config.client_id,
                "config_index": config_idx,
                "value": config.value,
                "cost": config.cost,
                "delay": config.delay,
                "memory": config.memory,
                "selected_count": summary["selected_count"],
                "earliest_index": summary["earliest_index"],
                "latest_index": summary["latest_index"],
                "active_indices": summary["active_indices"],
                "selected": int((config.client_id, config.assignment) in selected_keys),
                "budget_Ck": C_k,
                "T_actual": T_actual,
                "lambda_prime": self.lambda_prime,
            })
        self._append_trace_rows(
            f"scheduler_trace_{allocation_strategy}",
            [
                "experiment_id", "run_id", "dataset", "distribution", "method", "round",
                "client_id", "config_index", "value", "cost", "delay", "memory",
                "selected_count", "earliest_index", "latest_index", "active_indices",
                "selected", "budget_Ck", "T_actual", "lambda_prime",
            ],
            rows,
        )

    def _write_selection_trace(self, allocation_strategy, final_allocation):
        rows = []
        for config in final_allocation:
            client = self.clients[config.client_id]
            for matrix_idx, rank_value in enumerate(config.assignment):
                if rank_value <= 0:
                    continue
                rows.append({
                    "experiment_id": getattr(self.args, "experiment_id", ""),
                    "run_id": getattr(self.args, "run_id", ""),
                    "dataset": getattr(self.args, "dataset", ""),
                    "distribution": getattr(self.args, "distribution", ""),
                    "method": allocation_strategy,
                    "round": self.round_k,
                    "client_id": config.client_id,
                    "client_memory_gb": client.M_max,
                    "matrix_index": matrix_idx,
                    "module_name": self.candidate_modules[matrix_idx] if matrix_idx < len(self.candidate_modules) else "",
                    "depth_norm": matrix_idx / max(1, self.total_layers - 1),
                    "rank_value": rank_value,
                    "config_value": config.value,
                    "config_cost": config.cost,
                    "config_delay": config.delay,
                    "config_memory": config.memory,
                })
        self._append_trace_rows(
            f"selection_trace_{allocation_strategy}",
            [
                "experiment_id", "run_id", "dataset", "distribution", "method", "round",
                "client_id", "client_memory_gb", "matrix_index", "module_name",
                "depth_norm", "rank_value", "config_value", "config_cost",
                "config_delay", "config_memory",
            ],
            rows,
        )

    def _normalize_to_score_scale(self, values, low=1.0, high=10.0):
        arr = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size == 0:
            return arr
        arr_min = float(np.min(arr))
        arr_max = float(np.max(arr))
        if arr_max <= arr_min:
            return np.full_like(arr, low, dtype=float)
        return (arr - arr_min) / (arr_max - arr_min) * (high - low) + low

    def _compute_agreement_scores(self, score_vectors):
        if not score_vectors:
            return np.zeros(self.total_layers, dtype=float)
        matrix = np.vstack([
            np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            for scores in score_vectors
        ])
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        eps = float(getattr(self.args, "agreement_eps", 1e-8))
        agreement = mean / (std + eps)
        return self._normalize_to_score_scale(agreement, low=0.0, high=1.0)

    def _build_client_utility_scores(self, client, base_scores, global_max, agreement_scores, allocation_strategy=""):
        mode = str(getattr(self.args, "score_utility_mode", "raw") or "raw").lower()
        calibration_mode = str(
            getattr(self.args, "position_calibration_mode", "legacy") or "legacy"
        ).lower()
        raw = np.nan_to_num(np.asarray(base_scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        client.last_position_signal = 0.0
        client.last_position_weight = 0.0
        client.last_position_direction = "none"
        if self._is_cifar100_coverage_fim(allocation_strategy):
            final = self._build_cifar100_coverage_fim_scores(client, raw, global_max)
            client.last_final_utility = final.copy()
            client.last_agreement_scores = np.zeros_like(final)
            client.last_depth_prior = final.copy()
            return final
        if allocation_strategy == "Lite_FIM_WeakDepth":
            raw_min = float(np.min(raw)) if raw.size else 0.0
            raw_max = float(np.max(raw)) if raw.size else 0.0
            if raw_max > raw_min:
                norm_fim = (raw - raw_min) / (raw_max - raw_min)
            else:
                norm_fim = np.zeros_like(raw)
            depth_prior = client.get_depth_prior(scale="zero_one") if hasattr(client, "get_depth_prior") else np.zeros_like(raw)
            final = norm_fim + 0.15 * depth_prior
            final = np.nan_to_num(final, nan=0.0, posinf=1.15, neginf=0.0)
            client.last_depth_prior = depth_prior.copy()
            client.last_agreement_scores = np.zeros_like(final)
            client.last_final_utility = final.copy()
            return final
        explicit_adaptive = calibration_mode == "adaptive"
        if mode == "raw" and not explicit_adaptive:
            final = (raw / global_max) * 9.0 + 1.0
            client.last_final_utility = final.copy()
            client.last_agreement_scores = np.zeros_like(final)
            if hasattr(client, "get_depth_prior"):
                client.last_depth_prior = client.get_depth_prior(scale="one_ten")
            return final

        norm_score = (raw / global_max) * 9.0 + 1.0
        final = norm_score.copy()
        depth_prior = client.get_depth_prior(scale="one_ten") if hasattr(client, "get_depth_prior") else np.ones_like(norm_score)
        use_depth_mix = explicit_adaptive or (
            calibration_mode == "legacy"
            and (
                mode in {"depth_mix", "full"}
                or bool(getattr(self.args, "adaptive_depth_mix", False))
                or float(getattr(self.args, "depth_prior_weight", 0.0)) > 0.0
            )
        )
        if use_depth_mix:
            if explicit_adaptive or bool(getattr(self.args, "adaptive_depth_mix", False)):
                depth_prior, prior_weight, signal = self._build_adaptive_position_prior(client, raw)
                lambda_i = 1.0 - prior_weight
                client.last_position_signal = signal
                client.last_position_weight = prior_weight
                client.last_position_direction = "deep" if signal > 0 else "shallow" if signal < 0 else "none"
            else:
                lambda_i = 1.0 - float(getattr(self.args, "depth_prior_weight", 0.0))
            lambda_i = min(max(lambda_i, 0.0), 1.0)
            final = lambda_i * norm_score + (1.0 - lambda_i) * depth_prior

        use_agreement = mode in {"agreement", "full"} or float(getattr(self.args, "agreement_weight", 0.0)) > 0.0
        agreement_prior = np.asarray(agreement_scores, dtype=float)
        if use_agreement:
            final = final + float(getattr(self.args, "agreement_weight", 0.0)) * agreement_prior

        final = np.nan_to_num(final, nan=0.0, posinf=10.0, neginf=0.0)
        client.last_depth_prior = depth_prior.copy()
        client.last_agreement_scores = agreement_prior.copy()
        client.last_final_utility = final.copy()
        return final

    def _build_adaptive_position_prior(self, client, profile_scores):
        """Build a shallow- or deep-oriented prior from the existing one-batch profile."""
        scores = np.nan_to_num(np.asarray(profile_scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if scores.size == 0:
            return scores, 0.0, 0.0

        metadata = list(getattr(client, "candidate_metadata", []))[: len(scores)]
        if len(metadata) != len(scores):
            depths = np.linspace(0.0, 1.0, len(scores), dtype=float)
            block_scores = scores
        else:
            block_to_indices = collections.defaultdict(list)
            for idx, meta in enumerate(metadata):
                block_to_indices[int(meta.get("block_id", idx))].append(idx)
            block_ids = sorted(block_to_indices)
            if len(block_ids) < 2:
                return np.ones_like(scores), 0.0, 0.0
            block_pos = {block_id: pos for pos, block_id in enumerate(block_ids)}
            depths = np.asarray([
                block_pos[int(meta.get("block_id", idx))] / max(1, len(block_ids) - 1)
                for idx, meta in enumerate(metadata)
            ], dtype=float)
            # Averaging within a block prevents a Q/K/V scale imbalance from
            # being mistaken for a shallow-versus-deep preference.
            block_scores = np.asarray([
                float(np.mean(scores[block_to_indices[block_id]])) for block_id in block_ids
            ], dtype=float)

        band_fraction = float(getattr(self.args, "position_band_fraction", 1.0 / 3.0))
        band_fraction = min(max(band_fraction, 1.0 / max(2, len(block_scores))), 0.5)
        band_size = max(1, int(np.ceil(len(block_scores) * band_fraction)))
        shallow_mean = float(np.mean(block_scores[:band_size]))
        deep_mean = float(np.mean(block_scores[-band_size:]))
        eps = float(getattr(self.args, "score_stability_eps", 1e-8))
        signal = (deep_mean - shallow_mean) / (abs(deep_mean) + abs(shallow_mean) + eps)

        temperature = max(float(getattr(self.args, "position_signal_temperature", 0.15)), eps)
        confidence = float(np.tanh(abs(signal) / temperature))
        max_weight = float(getattr(self.args, "position_prior_max_weight", 0.45))
        prior_weight = min(max(max_weight, 0.0), 1.0) * confidence
        if abs(signal) <= eps or prior_weight <= eps:
            return np.ones_like(scores), 0.0, 0.0

        directional_depth = depths if signal > 0 else 1.0 - depths
        prior = 1.0 + 9.0 * directional_depth
        return prior, prior_weight, float(signal)

    def _is_cifar100_coverage_fim(self, allocation_strategy=""):
        return (
            getattr(self.args, "dataset", "") == "cifar100"
            and allocation_strategy == "Random_FIM"
            and getattr(self.args, "cifar100_prior_mode", "legacy_prior") == "legacy_prior"
        )

    def _build_cifar100_coverage_fim_scores(self, client, raw, global_max):
        """Blend FIM with a CIFAR100/ViT shallow-to-mid visual adaptation prior."""
        if raw.size == 0:
            return raw
        global_score = (raw / max(float(global_max), 1e-8)) * 9.0 + 1.0
        block_score = np.full_like(global_score, 5.5, dtype=float)
        anchor_score = np.full_like(global_score, 5.5, dtype=float)
        block_to_indices = {}
        for idx, meta in enumerate(getattr(client, "candidate_metadata", [])[: len(raw)]):
            block_id = int(meta.get("block_id", 0))
            block_to_indices.setdefault(block_id, []).append(idx)

        for indices in block_to_indices.values():
            values = raw[indices]
            lo = float(np.min(values))
            hi = float(np.max(values))
            if hi > lo:
                block_score[indices] = ((values - lo) / (hi - lo)) * 9.0 + 1.0

        block_ids = sorted(block_to_indices)
        block_pos = {block_id: pos for pos, block_id in enumerate(block_ids)}
        for idx, meta in enumerate(getattr(client, "candidate_metadata", [])[: len(raw)]):
            block_id = int(meta.get("block_id", 0))
            if len(block_ids) > 1:
                depth = block_pos.get(block_id, 0) / max(1, len(block_ids) - 1)
            else:
                depth = idx / max(1, len(raw) - 1)
            shallow = np.power(1.0 - depth, 1.35)
            deep = np.power(depth, 2.0)
            # CIFAR100/ViT benefits from shallow visual adaptation plus a light
            # final-block anchor, matching the stable structure seen in Uniform.
            anchor_score[idx] = 1.0 + 9.0 * (0.78 * shallow + 0.22 * deep)

        progress = min(max(self.round_k / max(1, int(getattr(self.args, "max_rounds", 1))), 0.0), 1.0)
        anchor_weight = 0.34 * (1.0 - progress) + 0.16 * progress
        final = (0.55 - 0.5 * anchor_weight) * global_score
        final += (0.45 - 0.5 * anchor_weight) * block_score
        final += anchor_weight * anchor_score
        return np.nan_to_num(final, nan=1.0, posinf=10.0, neginf=1.0)

    def _write_selection_analysis(self, allocation_strategy, selected_clients, final_allocation):
        if not self.log_selection_analysis:
            return
        final_by_client = {config.client_id: config for config in final_allocation}
        rows = []
        for client in selected_clients:
            config = final_by_client.get(client.client_id)
            assignment = list(config.assignment) if config is not None else [0] * self.total_layers
            base_assignment = [0] * self.total_layers
            base_mem = client.calculate_memory(base_assignment)
            _, _, base_latency = client.calculate_latency_components(base_assignment)

            for matrix_idx in range(self.total_layers):
                selected = int(matrix_idx < len(assignment) and assignment[matrix_idx] > 0)
                if selected and config is not None:
                    without = assignment.copy()
                    without[matrix_idx] = 0
                    old_mem = client.calculate_memory(without)
                    _, _, old_latency = client.calculate_latency_components(without)
                    estimated_memory = config.memory
                    estimated_latency = config.delay
                    memory_delta = max(estimated_memory - old_mem, 0.0)
                    latency_delta = max(estimated_latency - old_latency, 0.0)
                else:
                    singleton = [0] * self.total_layers
                    singleton[matrix_idx] = 1
                    estimated_memory = client.calculate_memory(singleton)
                    _, _, estimated_latency = client.calculate_latency_components(singleton)
                    memory_delta = max(estimated_memory - base_mem, 0.0)
                    latency_delta = max(estimated_latency - base_latency, 0.0)

                meta = self.candidate_metadata[matrix_idx] if matrix_idx < len(self.candidate_metadata) else {}
                rows.append({
                    "experiment_id": getattr(self.args, "experiment_id", ""),
                    "run_id": getattr(self.args, "run_id", ""),
                    "dataset": getattr(self.args, "dataset", ""),
                    "distribution": getattr(self.args, "distribution", ""),
                    "method": allocation_strategy,
                    "round": self.round_k,
                    "client_id": client.client_id,
                    "matrix_idx": matrix_idx,
                    "module_name": meta.get("module_name", self.candidate_modules[matrix_idx] if matrix_idx < len(self.candidate_modules) else ""),
                    "block_id": meta.get("block_id", ""),
                    "module_type": meta.get("module_type", "unknown"),
                    "group_type": meta.get("group_type", "unknown"),
                    "depth_norm": matrix_idx / max(1, self.total_layers - 1),
                    "raw_score": client.last_raw_scores[matrix_idx] if getattr(client, "last_raw_scores", None) is not None and matrix_idx < len(client.last_raw_scores) else "",
                    "ema_score": client.last_ema_scores[matrix_idx] if getattr(client, "last_ema_scores", None) is not None and matrix_idx < len(client.last_ema_scores) else "",
                    "depth_prior": client.last_depth_prior[matrix_idx] if getattr(client, "last_depth_prior", None) is not None and matrix_idx < len(client.last_depth_prior) else "",
                    "position_signal": getattr(client, "last_position_signal", 0.0),
                    "position_weight": getattr(client, "last_position_weight", 0.0),
                    "position_direction": getattr(client, "last_position_direction", "none"),
                    "agreement_score": client.last_agreement_scores[matrix_idx] if getattr(client, "last_agreement_scores", None) is not None and matrix_idx < len(client.last_agreement_scores) else "",
                    "final_utility": client.last_final_utility[matrix_idx] if getattr(client, "last_final_utility", None) is not None and matrix_idx < len(client.last_final_utility) else "",
                    "selected": selected,
                    "memory_delta": memory_delta,
                    "latency_delta": latency_delta,
                    "estimated_memory": estimated_memory,
                    "estimated_latency": estimated_latency,
                })
        self._append_analysis_rows(
            f"selection_analysis_{allocation_strategy}",
            [
                "experiment_id", "run_id", "dataset", "distribution", "method", "round",
                "client_id", "matrix_idx", "module_name", "block_id", "module_type",
                "group_type", "depth_norm", "raw_score", "ema_score", "depth_prior",
                "position_signal", "position_weight", "position_direction", "agreement_score",
                "final_utility", "selected", "memory_delta",
                "latency_delta", "estimated_memory", "estimated_latency",
            ],
            rows,
            self.log_selection_analysis,
        )

    def _flatten_update_delta(self, local_weights):
        chunks = []
        for key in sorted(local_weights.keys()):
            if key not in self.global_lora_weights:
                continue
            local_tensor = local_weights[key].detach().float().cpu()
            global_tensor = self.global_lora_weights[key].detach().float().cpu()
            common_shape = tuple(min(a, b) for a, b in zip(local_tensor.shape, global_tensor.shape))
            if not common_shape:
                continue
            slices = tuple(slice(0, dim) for dim in common_shape)
            chunks.append((local_tensor[slices] - global_tensor[slices]).reshape(-1))
        if not chunks:
            return None
        return torch.cat(chunks)

    def _write_update_cosine_trace(self, allocation_strategy, local_updates):
        if not self.log_update_cosine:
            return
        vectors = []
        client_ids = []
        for weights, config in local_updates:
            vec = self._flatten_update_delta(weights)
            if vec is not None and vec.numel() > 0:
                vectors.append(vec)
                client_ids.append(config.client_id)
        cosines = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                a, b = vectors[i], vectors[j]
                common = min(a.numel(), b.numel())
                if common <= 0:
                    continue
                denom = torch.norm(a[:common]) * torch.norm(b[:common])
                if denom.item() <= 1e-12:
                    continue
                cosines.append(float(torch.dot(a[:common], b[:common]) / denom))
        if cosines:
            arr = np.asarray(cosines, dtype=float)
            row = {
                "experiment_id": getattr(self.args, "experiment_id", ""),
                "run_id": getattr(self.args, "run_id", ""),
                "dataset": getattr(self.args, "dataset", ""),
                "distribution": getattr(self.args, "distribution", ""),
                "method": allocation_strategy,
                "round": self.round_k,
                "num_clients": len(vectors),
                "num_pairs": len(cosines),
                "client_ids": " ".join(str(cid) for cid in client_ids),
                "cosine_mean": float(np.mean(arr)),
                "cosine_std": float(np.std(arr)),
                "cosine_min": float(np.min(arr)),
                "cosine_max": float(np.max(arr)),
            }
        else:
            row = {
                "experiment_id": getattr(self.args, "experiment_id", ""),
                "run_id": getattr(self.args, "run_id", ""),
                "dataset": getattr(self.args, "dataset", ""),
                "distribution": getattr(self.args, "distribution", ""),
                "method": allocation_strategy,
                "round": self.round_k,
                "num_clients": len(vectors),
                "num_pairs": 0,
                "client_ids": " ".join(str(cid) for cid in client_ids),
                "cosine_mean": "",
                "cosine_std": "",
                "cosine_min": "",
                "cosine_max": "",
            }
        self._append_analysis_rows(
            f"update_cosine_{allocation_strategy}",
            [
                "experiment_id", "run_id", "dataset", "distribution", "method", "round",
                "num_clients", "num_pairs", "client_ids", "cosine_mean", "cosine_std",
                "cosine_min", "cosine_max",
            ],
            [row],
            self.log_update_cosine,
        )

    def _canonical_strategy_name(self, strategy_name):
        aliases = {
            "Ours-FIM": "Random_FIM",
            "Ours_FIM": "Random_FIM",
            "Matrix-FIM": "Random_FIM",
            "Matrix_FIM": "Random_FIM",
            "Old_Random_FIM": "Random_FIM",
            "Old_FIM_Depth": "Random_FIM",
            "Pure_FIM": "Random_FIM_NoDepth",
            "Ours-Taylor": "Random_Taylor",
            "Ours_Taylor": "Random_Taylor",
            "Matrix-Taylor": "Random_Taylor",
            "Matrix_Taylor": "Random_Taylor",
            "FedAvg-LoRA": "FedAvg_LoRA_homogeneous",
            "FedAvg_LoRA": "FedAvg_LoRA_homogeneous",
            "Homogeneous_LoRA": "FedAvg_LoRA_homogeneous",
            "Uniform": "Uniform_allocation",
            "HetLoRA": "HETLoRA",
            "HETLoRA-style": "HETLoRA",
            "HetLoRA-style": "HETLoRA",
            "HETLoRA_style": "HETLoRA",
            "HetLoRA_style": "HETLoRA",
            "FAH-QLoRA": "FAH_QLoRA",
            "FAH_QLoRA-style": "FAH_QLoRA",
            "FAH-QLoRA-style": "FAH_QLoRA",
            "No_Local_Profiling": "Random_Scores",
            "NoLocalProfiling": "Random_Scores",
            "No_Depth_FIM": "Random_FIM_NoDepth",
            "NoDepth_FIM": "Random_FIM_NoDepth",
            "No_Depth_Taylor": "Random_Taylor_NoDepth",
            "NoDepth_Taylor": "Random_Taylor_NoDepth",
            "PhysicalLayer_FIM": "Physical_FIM",
            "Physical_Layer_FIM": "Physical_FIM",
            "PhysicalLayer_Taylor": "Physical_Taylor",
            "Physical_Layer_Taylor": "Physical_Taylor",
            "Greedy": "Greedy_ValueCost",
            "Greedy_Scheduling": "Greedy_ValueCost",
        }
        return aliases.get(strategy_name, strategy_name)

    def _matrix_scheduler_strategies(self):
        return {
            "Random_FIM",
            "Lite_FIM_WeakDepth",
            "Random_Taylor",
            "Random_FIM_NoDepth",
            "Random_Taylor_NoDepth",
            "Random_Scores",
            "Random",
            "Greedy_Value",
            "Greedy_ValueCost",
            "Greedy_LatencyAware",
        }

    def _local_profile_strategies(self):
        return {
            "Random_FIM",
            "Lite_FIM_WeakDepth",
            "Random_Taylor",
            "Random_FIM_NoDepth",
            "Random_Taylor_NoDepth",
            "Physical_FIM",
            "Physical_Taylor",
            "Greedy_Value",
            "Greedy_ValueCost",
            "Greedy_LatencyAware",
        }

    def _make_shared_homogeneous_assignment(self, selected_clients, per_client_budget):
        if not selected_clients:
            return [0] * self.total_layers
        shared_cap = min(self.total_layers, int(per_client_budget))
        min_memory = min(client.M_max for client in selected_clients)
        template_client = min(selected_clients, key=lambda client: client.M_max)

        full_assignment = [1] * self.total_layers
        if (
            self.total_layers <= shared_cap
            and template_client.calculate_memory(full_assignment) <= min_memory
        ):
            return full_assignment

        if getattr(self.args, "baseline_granularity", "matrix") == "physical":
            if getattr(self.args, "fedavg_physical_policy", "uniform") == "prefix":
                return template_client._build_prefix_physical_assignment(
                    budget_limit=shared_cap,
                    shared_limit=shared_cap,
                )
            return template_client._build_uniform_physical_assignment(
                budget_limit=shared_cap,
                shared_limit=shared_cap,
            )

        return template_client._build_uniform_matrix_assignment(
            budget_limit=shared_cap,
            shared_limit=shared_cap,
        )

    def _greedy_schedule(self, all_configs, budget, mode="value_cost"):
        chosen = []
        used_clients = set()
        remaining_budget = float(budget)

        def score(config):
            if mode == "value":
                return config.value
            if mode == "latency":
                return config.value / max(config.cost * max(config.delay, 1e-9), 1e-9)
            return config.value / max(config.cost, 1e-9)

        for config in sorted(all_configs, key=score, reverse=True):
            if config.client_id in used_clients:
                continue
            if config.cost <= remaining_budget:
                chosen.append(config)
                used_clients.add(config.client_id)
                remaining_budget -= config.cost

        t_actual = max((config.delay for config in chosen), default=0.0)
        return chosen, t_actual

    def _prepare_input_batch(self, batch):
        """Move tensor-like model inputs to the server device."""
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

    # 在 Server 类中替换或添加此方法
    def _fedavg_aggregation(self, local_updates):
        """
        标准的 FedAvg 聚合：对 LoRA 参数进行加权平均。
        """
        from models.lora_utils import average_lora_weights

        print("\n[服务器] 执行 FedAvg 聚合...")

        weights_list = [update[0] for update in local_updates]
        avg_weights = average_lora_weights(weights_list)

        # --- [DEBUG] 监控权重变化幅度 ---
        total_diff_norm = 0.0
        param_count = 0

        for key, new_val in avg_weights.items():
            if key in self.global_lora_weights:
                old_val = self.global_lora_weights[key].to(new_val.device)

                # [核心修复] 避免计算 norm 时因形状不同(如 r=12 和 r=8)导致相减报错
                if new_val.shape == old_val.shape:
                    diff = (new_val - old_val).norm(2).item()
                else:
                    # 动态对齐尺寸后再算差值
                    common_shape = tuple(min(s1, s2) for s1, s2 in zip(new_val.shape, old_val.shape))
                    slices1 = tuple(slice(0, dim) for dim in common_shape)
                    slices2 = tuple(slice(0, dim) for dim in common_shape)
                    diff = (new_val[slices1] - old_val[slices2]).norm(2).item()

                total_diff_norm += diff
                param_count += 1

        print(f"  [Debug Aggregation] 聚合了 {len(local_updates)} 个客户端。")
        print(f"  [Debug Aggregation] 全局权重变化范数 (L2 Norm): {total_diff_norm:.4f}")

        return avg_weights

    def calculate_global_importance(self, scoring_method='fim'):
        """
        [核心] 服务端全局重要性评估 (Proxy Dataset Profiling)。
        用于 Fed-HeLLo 等需要全局静态视角的基线算法。
        """
        import copy
        import numpy as np
        import torch
        from models.lora_utils import apply_lora_to_model, find_lora_candidates

        # 1. 将基座模型切换至评估模式，并放到对应设备上
        self.pristine_model.to(self.device)
        self.pristine_model.eval()

        # 2. 临时构建一个开启了全部 LoRA 层的探测模型
        full_assignment = [1] * self.total_layers
        probe_model = apply_lora_to_model(copy.deepcopy(self.pristine_model), full_assignment)
        probe_model.load_state_dict(self.global_lora_weights, strict=False)
        probe_model.to(self.device)

        # 3. 激活所有 LoRA 层的梯度，冻结其他基座参数
        for n, p in probe_model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # 4. 获取服务端的一小批代理数据 (Proxy Dataset D_FIM)
        try:
            batch = next(iter(self.small_val_loader))
        except StopIteration:
            self.small_val_loader = iter(self.small_val_loader)
            batch = next(iter(self.small_val_loader))

        input_batch = self._prepare_input_batch(batch)
        if 'input_ids' not in input_batch and 'pixel_values' not in input_batch:
            raise ValueError("Fed-HeLLo FIM profiling batch 缺少 input_ids 或 pixel_values。")

        # 5. 前向传播与反向传播 (仅此一次)
        probe_model.zero_grad()
        outputs = probe_model(**input_batch)
        loss = outputs.loss
        loss.backward()

        # 6. 获取所有候选层的名字
        candidate_modules = find_lora_candidates(self.pristine_model)
        if len(candidate_modules) > self.total_layers:
            candidate_modules = candidate_modules[:self.total_layers]

        layer_scores = np.zeros(self.total_layers)

        # 7. 遍历梯度，累加分数
        for name, param in probe_model.named_parameters():
            if "lora_" not in name or param.grad is None:
                continue
            for layer_idx, module_name in enumerate(candidate_modules):
                if module_name in name:
                    # ==========================================
                    # 严谨的数学计算 (平方和/绝对值和)
                    # ==========================================
                    if scoring_method == 'fim':
                        # Fed-HeLLo 核心公式: 梯度的平方和
                        score = param.grad.detach().pow(2).sum().item()
                    elif scoring_method == 'taylor':
                        score = (param.detach() * param.grad.detach()).abs().sum().item()
                    else:
                        score = param.grad.data.norm(2).item()

                    layer_scores[layer_idx] += score
                    break

        # 8. 彻底释放显存
        del probe_model
        torch.cuda.empty_cache()
        return layer_scores

    def run_federated_round(self, C_k, T_max, current_performance, do_evaluate=True, allocation_strategy='Random_FIM'):
        """
        单轮联邦微调执行函数。
        :param C_k: 本轮全局预算
        :param T_max: 本轮最大延迟
        :param current_performance:
        :param do_evaluate: bool, 本轮是否执行模型评估
        :return: (实际消耗预算, 实际最大延迟, 实际平均延迟, 新的全局损失)
        """
        allocation_strategy = self._canonical_strategy_name(allocation_strategy)
        # [Overhead instrumentation] Best-effort wall-clock timers only observe existing stages.
        # They are stored on the server and never participate in allocation, training, or aggregation logic.
        round_wall_start = time.perf_counter()
        overhead_metrics = {
            "profiling_time_sec": 0.0,
            "config_generation_time_sec": 0.0,
            "server_scheduling_time_sec": 0.0,
            "client_training_time_sec": 0.0,
            "aggregation_time_sec": 0.0,
            "evaluation_time_sec": 0.0,
            "round_total_time_sec": 0.0,
            "num_clients": len(self.clients),
            "num_selected_clients": 0,
            "total_lora_candidates_L": self.total_layers,
            "avg_configs_per_client": 0.0,
            "max_configs_per_client": 0,
            "global_budget_Ck": C_k,
            "consumed_budget": 0,
            "delay_candidate_count_D": 0,
            "global_budget_W": getattr(self.args, "total_budget", 0),
            "method": allocation_strategy,
            "dataset": getattr(self.args, "dataset", ""),
            "round": self.round_k + 1,
        }

        def finalize_overhead(consumed_budget=0):
            overhead_metrics["consumed_budget"] = consumed_budget
            overhead_metrics["round_total_time_sec"] = time.perf_counter() - round_wall_start
            self.last_overhead_metrics = overhead_metrics.copy()

        self.round_k += 1

        print(f"\n{'=' * 30} 开始第 {self.round_k} 轮联邦微调 ({allocation_strategy}) {'=' * 30}")
        print(f"本轮全局预算 L^k = {C_k}, 延迟上限 = {T_max:.2f}s")
        # 客户端选择、配置生成、集成调度
        num_selected = max(1, int(len(self.clients)))
        selected_client_ids = np.random.choice(list(self.clients.keys()), num_selected, replace=False)
        selected_clients_obj = [self.clients[cid] for cid in selected_client_ids]
        overhead_metrics["num_selected_clients"] = len(selected_clients_obj)

        print(f"本轮选择了 {len(selected_clients_obj)} 个客户端: {selected_client_ids}")

        # === 步骤 2: 计算重要性评分与缓存机制 (Overhead Reduction) ===

        # 1. 初始化 Server 端的缓存字典 (保证在类的生命周期内持久化)
        if not hasattr(self, 'client_raw_scores_cache'):
            self.client_raw_scores_cache = {}  # 存放客户端的原始梯度分数
        if not hasattr(self, 'fedhello_fim_scores_cache'):
            self.fedhello_fim_scores_cache = None # 存放 Fed-HeLLo 的全局 FIM
        all_client_configs_for_trace = []

        profiling_start = time.perf_counter()
        if allocation_strategy == 'Fed-HeLLo':
            # [Fed-HeLLo 机制] 服务端评估，每 5 轮更新一次全局 FIM
            if self.fedhello_fim_scores_cache is None or self.round_k % 5 == 1:
                print(f"[Server] 执行 Fed-HeLLo 动态全局 FIM 评估 (使用 Proxy Dataset)...")
                # 调用时请确保该方法在 Server 类中已实现 (之前提供的 calculate_global_importance)
                self.fedhello_fim_scores_cache = self.calculate_global_importance(scoring_method='fim')
            else:
                print(f"[Server] Fed-HeLLo 复用已缓存的全局 FIM 分数 (节省开销)。")

            for client in selected_clients_obj:
                client.scores = self.fedhello_fim_scores_cache
            self._write_score_trace(allocation_strategy, selected_clients_obj, "server_proxy", "fim", "")

        elif allocation_strategy in self._local_profile_strategies():
            scoring_method = 'taylor' if 'Taylor' in allocation_strategy else 'fim'
            apply_depth_scaling = 'NoDepth' not in allocation_strategy
            if allocation_strategy == "Lite_FIM_WeakDepth":
                apply_depth_scaling = False
            calibration_mode = str(
                getattr(self.args, "position_calibration_mode", "legacy") or "legacy"
            ).lower()
            if calibration_mode != "legacy":
                apply_depth_scaling = calibration_mode == "fixed_deep"
            utility_mode = str(getattr(self.args, "score_utility_mode", "raw") or "raw").lower()

            #[核心机制] 设置客户端重新评估的周期 (每 2 轮算一次)
            profile_interval = 2

            need_profiling_clients =[]

            # 判断哪些客户端本轮必须真正进行计算
            for client in selected_clients_obj:
                # 触发条件：1. 缓存里没有它(第一次被选中)； 2. 达到了指定的更新周期
                if client.client_id not in self.client_raw_scores_cache or self.round_k % profile_interval == 1:
                    need_profiling_clients.append(client)

            # 执行按需计算
            if need_profiling_clients:
                print(f"[Server] 通知 {len(need_profiling_clients)}/{len(selected_clients_obj)} 个选中客户端进行本地重要性 profiling (方法: {scoring_method})...")
                for client in need_profiling_clients:
                    # 客户端进行极低成本的 1 个 Batch 的梯度计算
                    client.update_scores(
                        self.global_lora_weights,
                        scoring_method=scoring_method,
                        apply_depth_scaling=apply_depth_scaling,
                    )
                    # [关键修复] 将"原始"绝对分数深拷贝一份存入服务器缓存，防止被后续的归一化污染！
                    self.client_raw_scores_cache[client.client_id] = client.scores.copy()
                    self.client_score_component_cache[client.client_id] = {
                        "raw": getattr(client, "last_raw_scores", client.scores).copy(),
                        "ema": getattr(client, "last_ema_scores", client.scores).copy(),
                    } if hasattr(self, "client_score_component_cache") else {}
            else:
                print(f"[Server] 本轮所有 {len(selected_clients_obj)} 个选中客户端完美命中缓存，本地 profiling 零开销！")

            # ==============================================================
            # 2. 跨客户端全局重要性归一化 (Global Normalization)
            # ==============================================================
            # [关键数学修复]：我们必须从"缓存的原始分数"中提取 global_max
            # 因为如果我们用归一化后的分数去比，尺度就全乱了。
            current_raw_scores = [self.client_raw_scores_cache[client.client_id] for client in selected_clients_obj]
            global_max = max([np.max(scores) for scores in current_raw_scores])

            if global_max <= 1e-9:
                global_max = 1.0  # 防止除零

            agreement_enabled = (
                utility_mode in {"agreement", "full"}
                or float(getattr(self.args, "agreement_weight", 0.0)) > 0.0
            )
            self.current_agreement_scores = self._compute_agreement_scores(current_raw_scores) if agreement_enabled else np.zeros(self.total_layers, dtype=float)

            for client in selected_clients_obj:
                # 取出该客户端的原始/EMA分数
                raw_s = self.client_raw_scores_cache[client.client_id]
                # 将所有人映射到[1.0, 10.0] 区间；raw 模式保持旧逻辑。
                client.scores = self._build_client_utility_scores(
                    client,
                    raw_s,
                    global_max,
                    self.current_agreement_scores,
                    allocation_strategy,
                )
            self._write_score_trace(allocation_strategy, selected_clients_obj, "local_profile", scoring_method, global_max)

        elif allocation_strategy == 'Random_Scores':
            for client in selected_clients_obj:
                client.scores = np.ones(self.total_layers, dtype=float)
            self._write_score_trace(allocation_strategy, selected_clients_obj, "constant", "none", "")
        else:
            print(f"[Server] 策略为 {allocation_strategy}，跳过梯度评分计算。")
        overhead_metrics["profiling_time_sec"] = time.perf_counter() - profiling_start

        # === 步骤 3: 资源分配与调度 ===
        final_allocation =[]
        configs_per_client = []
        if allocation_strategy == 'Matrix_DP_MCKP':
            print(f"本轮全局预算 L^k = {C_k}, 延迟上限 = {T_max:.2f}s")
            all_client_configs =[]
            config_start = time.perf_counter()
            for client in selected_clients_obj:
                client_configs = client.generate_configurations(T_max)
                configs_per_client.append(len(client_configs))
                all_client_configs.extend(client_configs)
            all_client_configs_for_trace = list(all_client_configs)
            overhead_metrics["config_generation_time_sec"] = time.perf_counter() - config_start
            overhead_metrics["delay_candidate_count_D"] = len(set(c.delay for c in all_client_configs))
            scheduling_start = time.perf_counter()
            final_allocation, optimal_T_max = self.integrated_scheduling(all_client_configs, C_k)
            overhead_metrics["server_scheduling_time_sec"] = time.perf_counter() - scheduling_start
            T_actual = optimal_T_max
        else:
            print(f"执行启发式策略: {allocation_strategy}")

            # Matrix-level strategies share the client DP and server MCKP scheduling path.
            if allocation_strategy in self._matrix_scheduler_strategies():
                all_client_configs =[]
                config_start = time.perf_counter()
                for client in selected_clients_obj:
                    client.current_allocation_strategy = allocation_strategy
                    client_configs = client.generate_configurations(T_max)
                    configs_per_client.append(len(client_configs))
                    all_client_configs.extend(client_configs)
                all_client_configs_for_trace = list(all_client_configs)
                overhead_metrics["config_generation_time_sec"] = time.perf_counter() - config_start
                overhead_metrics["delay_candidate_count_D"] = len(set(c.delay for c in all_client_configs))
                scheduling_start = time.perf_counter()
                if allocation_strategy == 'Greedy_Value':
                    final_allocation, optimal_T_max = self._greedy_schedule(all_client_configs, C_k, mode="value")
                elif allocation_strategy == 'Greedy_LatencyAware':
                    final_allocation, optimal_T_max = self._greedy_schedule(all_client_configs, C_k, mode="latency")
                elif allocation_strategy == 'Greedy_ValueCost':
                    final_allocation, optimal_T_max = self._greedy_schedule(all_client_configs, C_k, mode="value_cost")
                else:
                    final_allocation, optimal_T_max = self.integrated_scheduling(all_client_configs, C_k)
                overhead_metrics["server_scheduling_time_sec"] = time.perf_counter() - scheduling_start
                T_actual = optimal_T_max
            else:
                 # 原有 MS, MH, FedRA 逻辑保持不变...
                heuristic_scheduling_start = time.perf_counter()
                temp_allocation =[]
                total_cost = 0

                clients_to_process = list(selected_clients_obj) # 复制一份
                np.random.shuffle(clients_to_process)

                # ==========================================================
                # [公平对比核心]：计算均摊到每个客户端的预算配额
                # 这样基线就不会被 1 个客户端独占，强制在细粒度受限下与你的算法 PK
                # ==========================================================
                budget_per_client = C_k / max(1, len(clients_to_process))
                print(f"[DEBUG] 公平分配，基线单客户端预算上限: {budget_per_client:.2f}")

                shared_assignment = None
                if (
                    allocation_strategy == 'FedAvg_LoRA_homogeneous'
                    and getattr(self.args, "dataset", "") != "cifar100"
                ):
                    shared_assignment = self._make_shared_homogeneous_assignment(clients_to_process, budget_per_client)

                for client in clients_to_process:
                    # 传入 budget_limit
                    config_start = time.perf_counter()
                    client.current_round = self.round_k
                    if shared_assignment is not None:
                        config = client.generate_fixed_assignment_config(shared_assignment)
                    else:
                        config = client.generate_heuristic_config(allocation_strategy, budget_limit=budget_per_client)
                    overhead_metrics["config_generation_time_sec"] += time.perf_counter() - config_start
                    configs_per_client.append(1)
                    all_client_configs_for_trace.append(config)

                    if config.cost > 0:
                        if total_cost + config.cost <= C_k:
                            temp_allocation.append(config)
                            total_cost += config.cost
                final_allocation = temp_allocation
                print(f"[DEBUG] 最终选中的配置数量: {len(final_allocation)}")

                # 计算实际延迟 (取最慢的)
                if final_allocation:
                    T_actual = max(c.delay for c in final_allocation)
                else:
                    T_actual = 0

                optimal_T_max = T_actual
                overhead_metrics["delay_candidate_count_D"] = len(set(c.delay for c in temp_allocation))
                overhead_metrics["server_scheduling_time_sec"] = max(
                    0.0,
                    time.perf_counter() - heuristic_scheduling_start - overhead_metrics["config_generation_time_sec"]
                )
        if configs_per_client:
            overhead_metrics["avg_configs_per_client"] = float(np.mean(configs_per_client))
            overhead_metrics["max_configs_per_client"] = int(max(configs_per_client))
        configs_per_client_by_id = {
            client.client_id: count
            for client, count in zip(selected_clients_obj, configs_per_client)
        }
        self._write_memory_trace(allocation_strategy, selected_clients_obj, configs_per_client_by_id)
        self._write_scheduler_trace(allocation_strategy, all_client_configs_for_trace, final_allocation, C_k, T_actual)
        self._write_selection_trace(allocation_strategy, final_allocation)
        self._write_selection_analysis(allocation_strategy, selected_clients_obj, final_allocation)

        # 在启发式策略分支里
        if allocation_strategy == 'MS' and len(final_allocation) > 0:
            # 随便拿一个客户端看看
            sample_conf = final_allocation[0]
            # 打印被激活层的索引
            active_indices = [i for i, x in enumerate(sample_conf.assignment) if x == 1]
            print(f"[DEBUG MS Check] Client {sample_conf.client_id} active layers: {active_indices}")

            # 如果 all_layers 是排好序的，看看名字
            all_names = find_lora_candidates(self.pristine_model) # 需要临时获取一下
            print(f"Active Layer Names: {[all_names[i] for i in active_indices]}")

        if not final_allocation:
            print("本轮没有选择任何客户端参与训练，损失不变。")
            # 如果没有客户端被选中，记录空数据或默认值
            C_consumed = 0
            T_actual = 0
            finalize_overhead(C_consumed)
            # 返回 0 表示没有消耗，损失不变
            return (0, 0.0, 0.0, 0.0, current_performance, 0, 0, 0, 0.0, 0, current_performance)
        else:
            C_consumed = sum(c.cost for c in final_allocation)

            latencies = []
            comps = []
            comms = []
            for config in final_allocation:
                client = self.clients[config.client_id]
                if allocation_strategy == 'FAH_QLoRA':
                    t_comp, t_comm, t_total = client.calculate_fah_latency_components(config.assignment)
                else:
                    t_comp, t_comm, t_total = client.calculate_latency_components(config.assignment)
                latencies.append(t_total)
                comps.append(t_comp)
                comms.append(t_comm)

            # 1. 本轮系统实际耗时 (取决于最慢的客户端)
            T_actual = max(latencies)
            # 2. 统计用的平均指标 (用于观察系统瓶颈)
            avg_comp_latency = np.mean(comps)
            avg_comm_latency = np.mean(comms)
            # --- 模拟其他指标 ---
            # 在真实研究中，这些值应该从客户端的训练报告中获取
            # 1. 反向传播计算成本 (TFLOPs)
            # 假设每个 LoRA 层的反向传播成本约为 0.1 TFLOPs
            avg_flops = np.mean([c.cost * 0.1 for c in final_allocation])
            # 2. 内存使用量 (GB)
            avg_memory = np.mean([c.memory for c in final_allocation])
            # 3. 通信成本 (MB/R)
            # LoRA 的通信成本 = 上传的 LoRA 参数大小
            # LoRA r=8, d=768, float16, 参数量=12288, 大小 = 12288 * 2 / 1024^2 ≈ 0.023 MB
            avg_comm_cost = np.mean([c.cost * 0.023 for c in final_allocation])
            # --- 模拟结束 ---


        print(f"集成调度完成。实际最大延迟 T_max* = {T_actual:.2f}s. " # 建议这里直接打印 T_actual
              f"选择了 {len(final_allocation)} 个客户端参与训练。")

        # --- 客户端训练 ---
        # --- 核心修改：动态组装当前轮次的模型 ---
        # 1. 从干净的架构创建新模型实例
        current_round_model = copy.deepcopy(self.pristine_model)
        # 2. 加载最新的全局权重
        current_round_model.load_state_dict(self.global_model_weights)

        # --- 客户端训练 (并行版) ---
        local_updates = []
        final_client_assignments = {c.client_id: c for c in final_allocation}

        # 1. 定义单客户端训练任务
        # 注意：这里我们传入 self.global_lora_weights (只读)，避免多线程竞争写入
        def run_single_client(client_id, config):
            client = self.clients[client_id]
            # [核心修复] 计算实际激活的物理矩阵个数（不论 Rank 大小）
            actual_matrix_count = sum(1 for a in config.assignment if a > 0)
            print(f"  -> 客户端 {client.client_id} 开始训练... (激活 {actual_matrix_count} 个矩阵, 消耗标准预算: {config.cost:.3f}, 内存: {config.memory:.2f}GB)")

            try:
                # 打印设备信息，确认并行
                # print(f"  [启动] Client {client_id} on {client.device}")
                lora_w = client.train(config, self.global_lora_weights)
                return lora_w, config
            except Exception as e:
                print(f"  !!! 客户端 {client_id} 训练失败: {e}")
                return None, None
            finally:
                # [新增] 再次确保清理
                torch.cuda.empty_cache()

        # 2. 设置并行度。OPT-1.3B 的名称通常不包含 "large"，需要用结构和任务一起判断。
        model_config = self.pristine_model.config
        model_name = str(getattr(model_config, "_name_or_path", "")).lower()
        model_type = str(getattr(model_config, "model_type", "")).lower()
        hidden_layers = int(getattr(model_config, "num_hidden_layers", 0) or 0)
        hidden_size = int(getattr(model_config, "hidden_size", 0) or 0)

        is_large_model = (
            self.args.dataset in ["samsum", "dolly15k"]
            or "1.3b" in model_name
            or "opt-1.3" in model_name
            or model_type in ["opt", "llama", "gpt_neox"]
            or hidden_layers >= 24
            or hidden_size >= 2048
        )

        if is_large_model:
            optimal_workers = min(len(final_allocation), 1)  # 单卡 OPT-1.3B 每个客户端都会复制模型，必须串行
        else:
            max_client_workers = max(1, int(getattr(self.args, "max_client_train_workers", 10)))
            optimal_workers = min(len(final_allocation), max_client_workers)  # Base 模型默认并行 10 个

        # 每次启动线程池前，强制清空一次服务器评估遗留的显存垃圾
        torch.cuda.empty_cache()

        # 3. 启动线程池
        print(
            f"--- 启动并行训练 (Workers: {optimal_workers} on A800 | Large Model: {is_large_model} "
            f"| model_type={model_type} | name={model_name} | layers={hidden_layers} | hidden={hidden_size}) ---"
        )


        training_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(run_single_client, cid, cfg): cid
                for cid, cfg in final_client_assignments.items()
            }

            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                lora_w, config = future.result()
                if lora_w:
                    local_updates.append((lora_w, config))
                else:
                    cid = futures[future]
                    print(f"  -- 警告: 客户端 {cid} 未返回权重 --")
        overhead_metrics["client_training_time_sec"] = time.perf_counter() - training_start

        if not local_updates:
            print("没有客户端成功返回权重，跳过聚合。")
            finalize_overhead(0)
            return (0, 0.0, 0.0, 0.0, current_performance, 0, 0, 0, 0.0, 0, current_performance)

        # --- [修复] 聚合与更新 ---
        # 1. 计算新的全局权重
        aggregation_start = time.perf_counter()
        self._write_update_cosine_trace(allocation_strategy, local_updates)
        new_global_lora_weights = self._fedavg_aggregation(local_updates)

        # 2. [至关重要] 更新服务器的全局状态！！！
        # 注意：average_lora_weights 可能只返回了部分键（因为是异构的），
        # 我们需要将其更新到 self.global_lora_weights 中
        self.global_lora_weights.update(new_global_lora_weights)
        overhead_metrics["aggregation_time_sec"] = time.perf_counter() - aggregation_start

        # [核心修改] 评估调度器
        new_loss = current_performance # 默认继承上一轮性能
        raw_lm_loss = current_performance

        # if do_evaluate:
        evaluation_start = time.perf_counter()
        if True:
            # --- [修复] 评估 ---
            print("\n[服务器] 正在评估新一轮全局模型...")

            # 1. 组装用于评估的完整模型
            from models.lora_utils import apply_lora_to_model

            # 使用全量 assignment (所有层都激活) 或者基于上一轮的覆盖情况
            eval_model_base = copy.deepcopy(self.pristine_model)

            # 必须应用一个能容纳当前所有权重的 LoRA Config,简单起见，我们对所有层应用 LoRA
            full_assignment = [1] * self.total_layers
            eval_model = apply_lora_to_model(eval_model_base, full_assignment)

            # 加载刚刚更新的全局权重 (strict=False 是因为可能某些层还没被训练过)
            # eval_model.load_state_dict(self.global_lora_weights, strict=False)

            eval_state = eval_model.state_dict()
            for k, global_tensor in self.global_lora_weights.items():
                if k in eval_state:
                    eval_tensor = eval_state[k]
                    if global_tensor.shape == eval_tensor.shape:
                        eval_tensor.copy_(global_tensor)
                    else:
                        # 找到最小公共尺寸，进行安全裁剪
                        common_shape = tuple(min(sg, sl) for sg, sl in zip(global_tensor.shape, eval_tensor.shape))
                        slices_g = tuple(slice(0, dim) for dim in common_shape)
                        slices_e = tuple(slice(0, dim) for dim in common_shape)
                        eval_tensor[slices_e].copy_(global_tensor[slices_g])

            rouge_score = 0.0
            accuracy = 0.0

            if self.args.dataset in ['natural_instructions', 'dolly15k', 'samsum']:
                raw_lm_loss = evaluate_model_loss(eval_model, self.val_loader, self.device)
                print(f"  - [评估结果] Raw LM Loss = {raw_lm_loss:.4f}")

                rouge_score = evaluate_rouge_l(eval_model, self.tokenizer, self.val_loader, self.device)
                print(f"  - [评估结果] Rouge-L = {rouge_score:.4f}")

                if self.args.dataset == 'samsum':
                    new_loss = max(0.0, 1.0 - rouge_score)
                    print(f"  - [评估结果] SAMSum Performance Proxy (1 - Rouge-L) = {new_loss:.4f}")
                else:
                    new_loss = raw_lm_loss

            elif self.args.dataset in['cifar100', 'domainnet', 'sst2', 'qnli', 'qqp', 'mnli', 'imdb', 'ledgar']:
                accuracy = evaluate_accuracy(eval_model, self.val_loader, self.device)
                print(f"  - [评估结果] Accuracy = {accuracy:.4f}")
                new_loss = evaluate_model_loss(eval_model, self.val_loader, self.device)
                raw_lm_loss = new_loss
                print(f"  - [评估结果] Validation Loss = {new_loss:.4f}")

                if self.args.dataset in ['cifar100', 'domainnet']:
                    rouge_score = accuracy

            else:
                new_loss = evaluate_model_loss(eval_model, self.val_loader, self.device)
                raw_lm_loss = new_loss
                print(f"  - [评估结果] Validation Loss = {new_loss:.4f}")

            actual_T_avg = np.mean([c.delay for c in final_allocation]) if final_allocation else 0
            print(f"该轮次: T_Actual(Max)={T_actual:.2f}s | Avg_Comp={avg_comp_latency:.2f}s | Avg_Comm={avg_comm_latency:.2f}s | Loss={new_loss:.4f}")
        overhead_metrics["evaluation_time_sec"] = time.perf_counter() - evaluation_start

        finalize_overhead(C_consumed)
        return (C_consumed, T_actual, avg_comp_latency, avg_comm_latency, new_loss,
                avg_flops, avg_memory, avg_comm_cost, rouge_score, accuracy, raw_lm_loss)
