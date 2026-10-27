#!/usr/bin/env python3
"""SST-2 scoring/granularity ablation without altering the main methods.

All four variants use the existing FedEnv, training, aggregation and MCKP.
Only the isolated proxy score source or physical-layer menu generator is
substituted in this experiment-specific entry point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import FedEnv as fed_env_module
from FedEnv import FedEnv
from baselines.budget_schedule_strategy import BudgetScheduleStrategy
from main_drl import get_model_and_tokenizer
from models.client import Client
from models.scheduling import Configuration
from models.server import Server
from utils.data_utils import custom_collate_fn_for_generation, load_and_preprocess_data
from utils.evaluation_runner import EvaluationRunner
from utils.options import args_parser


VARIANTS = ("Full", "Server-Proxy", "Layer-Level", "No-Depth")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", required=True, type=int, choices=[2024, 2025, 2026])
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def physical_layer_menus(client: Client, deadline: float):
    """Generate exact additive-utility options over whole physical layers."""
    groups = defaultdict(list)
    for index, meta in enumerate(client.candidate_metadata[:client.total_layers]):
        groups[int(meta["block_id"])].append(index)
    if len(groups) != 12 or any(len(indices) != 3 for indices in groups.values()):
        raise RuntimeError("Layer-Level requires 12 RoBERTa blocks with exactly three QKV matrices each")
    blocks = sorted(groups)
    block_score = {block: float(sum(client.scores[index] for index in groups[block])) for block in blocks}
    options = []
    checked = rejected = 0
    for first_position, first_block in enumerate(blocks):
        suffix = sorted(blocks[first_position + 1:], key=lambda block: (-block_score[block], block))
        for extra in range(len(suffix) + 1):
            chosen = [first_block] + suffix[:extra]
            assignment = [0] * client.total_layers
            for block in chosen:
                for index in groups[block]:
                    assignment[index] = 1
            cost = sum(assignment)
            if not (client.L_min <= cost <= client.L_max):
                continue
            checked += 1
            memory = client.calculate_memory(assignment)
            delay = client.calculate_latency_components(assignment)[2]
            if memory > client.M_max or delay > deadline:
                rejected += 1
                continue
            options.append(Configuration(
                client_id=client.client_id,
                value=float(sum(client.scores[index] for index, selected in enumerate(assignment) if selected)),
                cost=float(cost), delay=delay, memory=memory, assignment=tuple(assignment),
            ))
    result = client._prune_configurations(options)
    client.last_config_generation_summary = {
        "generated_configs": len(options), "pruned_configs": len(result),
        "memory_rejects": rejected, "transition_checks": checked,
        "c_time_max": int(deadline // client.T_layer_avg),
        "c_max_global": min(client.L_max, client.total_layers),
        "physical_layer_only": True,
    }
    return result


class AblationServer(Server):
    variant = "Full"
    proxy_dataset = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proxy_scores = None
        self.proxy_profile_sec = 0.0
        if self.variant == "Layer-Level":
            for client in self.clients.values():
                client.generate_configurations = MethodType(physical_layer_menus, client)
        if self.variant == "Server-Proxy":
            if self.proxy_dataset is None or len(self.proxy_dataset) < 16:
                raise RuntimeError("Independent server proxy split is missing")
            for client in self.clients.values():
                client.calculate_local_gradients = MethodType(
                    lambda _client, *_args, **_kwargs: self.proxy_scores.copy(), client)

    def _write_score_trace(self, method, clients, source, scoring_method="", global_max=""):
        if self.variant == "Server-Proxy":
            source = "independent_train_server_proxy"
        return super()._write_score_trace(method, clients, source, scoring_method, global_max)

    def run_federated_round(self, C_k, T_max, current_performance, do_evaluate=True, allocation_strategy="Random_FIM"):
        next_round = self.round_k + 1
        self.proxy_profile_sec = 0.0
        if self.variant == "Server-Proxy" and (self.proxy_scores is None or next_round % 2 == 1):
            reference = self.clients[0]
            old_dataset, old_indices = reference.dataset, reference.data_indices
            try:
                start = time.perf_counter()
                proxy_parts = []
                for part in range(len(self.clients)):
                    reference.dataset = self.proxy_dataset
                    reference.data_indices = list(range(part * 16, (part + 1) * 16))
                    # Match one local mini-batch per client with equally many
                    # disjoint proxy mini-batches, averaged into one shared score.
                    proxy_parts.append(Client.calculate_local_gradients(
                        reference, self.global_lora_weights,
                        scoring_method="fim", apply_depth_scaling=True))
                self.proxy_scores = np.mean(proxy_parts, axis=0)
                self.proxy_profile_sec = time.perf_counter() - start
            finally:
                reference.dataset, reference.data_indices = old_dataset, old_indices
            if not np.isfinite(self.proxy_scores).all() or max(self.proxy_scores) <= 0:
                raise RuntimeError("Server proxy produced invalid FIM scores")
        result = super().run_federated_round(
            C_k, T_max, current_performance, do_evaluate=do_evaluate,
            allocation_strategy=allocation_strategy)
        if self.variant == "Server-Proxy":
            os.makedirs(self.trace_dir, exist_ok=True)
            path = Path(self.trace_dir) / "proxy_profile_overhead.csv"
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["run_id", "round", "proxy_profile_sec", "profile_refresh"])
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow({"run_id": self.args.run_id, "round": self.round_k,
                                 "proxy_profile_sec": self.proxy_profile_sec,
                                 "profile_refresh": int(self.proxy_profile_sec > 0)})
        return result


def main():
    cli = parse()
    import sys
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        args = args_parser()
    finally:
        sys.argv = original_argv
    seed_all(cli.seed)
    args.dataset = "sst2"
    args.distribution = "dirichlet:1.0"
    args.total_budget = 7200
    args.max_rounds = 2 if cli.pilot else 80
    args.local_bs = 16
    args.local_ep = 1
    args.lr = 1e-5
    args.optimizer = "adamw"
    args.lambda_prime = 0.05
    args.memory_model = "position"
    args.seed = cli.seed
    args.gpu = 0 if torch.cuda.is_available() else -1
    if args.gpu < 0 and not cli.pilot:
        raise RuntimeError("Full ablation requires a GPU")
    args.dataloader_num_workers = 0
    args.max_client_train_workers = 2
    args.enable_trace = True
    args.log_selection_analysis = True
    args.position_calibration_mode = "legacy"
    args.score_utility_mode = "raw"
    args.experiment_id = f"SCORE_ABL_20261025_SST2_D1_{cli.variant.replace('-', '')}_seed{cli.seed}"
    suffix = f"_{os.environ.get('SLURM_JOB_ID', datetime.now().strftime('%Y%m%d_%H%M%S'))}"
    if cli.pilot:
        args.experiment_id += "_PILOT"
    args.run_id = args.experiment_id + suffix
    output = ROOT / "evaluation_results/sst2" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    args.trace_dir = str(output / "traces")

    model, tokenizer = get_model_and_tokenizer(args, share_memory=False)
    from models.lora_utils import find_lora_candidates
    args.total_lora_layers = len(find_lora_candidates(model))
    args.client_l_max = args.total_lora_layers
    train, test, labels = load_and_preprocess_data(args, tokenizer)
    permutation = np.random.default_rng(cli.seed + 1025).permutation(len(train))
    proxy_indices = permutation[:256].tolist()
    private_indices = permutation[256:].tolist()
    if set(proxy_indices) & set(private_indices):
        raise RuntimeError("Proxy overlaps private training samples")
    proxy = train.select(proxy_indices)
    private = train.select(private_indices)
    proxy_loader = DataLoader(proxy.select(range(16)), batch_size=16, shuffle=False,
                              collate_fn=custom_collate_fn_for_generation, num_workers=0)
    val = test.select(range(min(500, len(test))))
    val_loader = DataLoader(val, batch_size=16, shuffle=False,
                            collate_fn=custom_collate_fn_for_generation, num_workers=0)
    AblationServer.variant = cli.variant
    AblationServer.proxy_dataset = proxy
    fed_env_module.Server = AblationServer
    env = FedEnv(
        args=args, tokenizer=tokenizer, train_dataset=private, test_dataset=test,
        val_loader=val_loader, small_val_loader=proxy_loader, val_text_labels=None,
        global_model=model, allocation_strategy="Random_FIM")
    strategy = "Random_FIM_NoDepth" if cli.variant == "No-Depth" else "Random_FIM"
    schedule = BudgetScheduleStrategy(total_budget=args.total_budget,
                                      max_rounds=args.max_rounds,
                                      per_round_cap=args.num_clients * args.client_l_max,
                                      min_ratio=0.1, max_ratio=1.0)
    metadata = {
        "run_id": args.run_id, "experiment_id": args.experiment_id, "dataset": "sst2",
        "distribution": args.distribution, "method": cli.variant, "method_raw": strategy,
        "seed": cli.seed, "metric": "validation_accuracy",
        "config_path": "scripts/run_sst2_scoring_ablation_20261025.py",
        "raw_result_path": str((output / f"eval_{strategy}.csv").relative_to(ROOT)),
        "trace_dir": str((output / "traces").relative_to(ROOT)),
        "proxy_indices_sha256": hashlib.sha256(np.asarray(proxy_indices, dtype=np.int64).tobytes()).hexdigest(),
        "private_indices_sha256": hashlib.sha256(np.asarray(private_indices, dtype=np.int64).tobytes()).hexdigest(),
        "proxy_pool_size": len(proxy), "proxy_profile_size_per_batch": 16,
        "proxy_batches_per_refresh": args.num_clients,
        "proxy_source": "SST-2 training split only; disjoint from all client private samples and validation",
        "rounds": args.max_rounds, "budget": args.total_budget, "rank": 8,
        "local_bs": args.local_bs, "local_ep": args.local_ep, "lr": args.lr,
        "memory_model": args.memory_model, "lambda_prime": args.lambda_prime,
        "max_client_train_workers": args.max_client_train_workers,
        "pilot": cli.pilot,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    runner = EvaluationRunner(env, args, run_id=args.run_id)
    runner.run(schedule, strategy, seed=cli.seed)
    env.close()
    print("FINISHED", output)


if __name__ == "__main__":
    main()
