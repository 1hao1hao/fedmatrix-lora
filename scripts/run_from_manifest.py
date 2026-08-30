#!/usr/bin/env python3
"""Run one experiment row from a CSV manifest.

This is intended for Slurm job arrays. Slurm still allocates CPU/GPU resources;
the manifest only provides the experiment parameters for each array row.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one main_drl.py experiment from a manifest row.")
    parser.add_argument("--manifest", default="experiments/critical_gap_manifest.csv")
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="1-based data row index. Defaults to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    rows = load_manifest(manifest_path)
    row_index = args.row
    if row_index is None:
        array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if not array_id:
            raise SystemExit("--row is required when SLURM_ARRAY_TASK_ID is not set")
        row_index = int(array_id)

    if row_index < 1 or row_index > len(rows):
        raise SystemExit(f"row index {row_index} is outside manifest range 1..{len(rows)}")

    row = rows[row_index - 1]
    command = [
        sys.executable,
        "-u",
        "main_drl.py",
        "--experiment_id",
        row["experiment_id"],
        "--dataset",
        row["dataset"],
        "--distribution",
        row["distribution_arg"],
        "--local_bs",
        row["local_bs"],
        "--lr",
        row["lr"],
        "--local_ep",
        row["local_ep"],
        "--total_budget",
        row["total_budget"],
        "--max_rounds",
        row["max_rounds"],
        "--seed",
        row["seed"],
        "--test_method",
        row["test_method"],
        "--gpu",
        str(args.gpu),
        "--num_parallel_envs",
        row.get("num_parallel_envs", "1") or "1",
    ]

    optional_args = {
        "lambda_prime": "--lambda_prime",
        "fixed_t_max": "--fixed_t_max",
        "dataloader_num_workers": "--dataloader_num_workers",
        "max_client_train_workers": "--max_client_train_workers",
        "memory_model": "--memory_model",
        "mem_levels_gb": "--mem_levels_gb",
        "mem_dist_ratio": "--mem_dist_ratio",
        "data_subsample_ratio": "--data_subsample_ratio",
        "taylor_fim_blend": "--taylor_fim_blend",
        "depth_non_iid_extra": "--depth_non_iid_extra",
        "depth_severe_extra": "--depth_severe_extra",
        "score_ema_beta": "--score_ema_beta",
        "score_stability_eps": "--score_stability_eps",
        "score_utility_mode": "--score_utility_mode",
        "cifar100_prior_mode": "--cifar100_prior_mode",
        "baseline_granularity": "--baseline_granularity",
        "fedavg_physical_policy": "--fedavg_physical_policy",
        "fedhello_anchor_layers": "--fedhello_anchor_layers",
        "depth_prior_weight": "--depth_prior_weight",
        "depth_mix_min": "--depth_mix_min",
        "depth_mix_max": "--depth_mix_max",
        "position_prior_max_weight": "--position_prior_max_weight",
        "position_signal_temperature": "--position_signal_temperature",
        "position_band_fraction": "--position_band_fraction",
        "position_calibration_mode": "--position_calibration_mode",
        "agreement_weight": "--agreement_weight",
        "agreement_eps": "--agreement_eps",
        "marginal_memory_penalty": "--marginal_memory_penalty",
        "marginal_latency_penalty": "--marginal_latency_penalty",
        "block_coherence_weight": "--block_coherence_weight",
        "qkv_completeness_weight": "--qkv_completeness_weight",
    }
    for column, cli_arg in optional_args.items():
        value = row.get(column, "")
        if value not in ("", None):
            command.extend([cli_arg, value])

    if str(row.get("enable_trace", "")).strip().lower() in {"1", "true", "yes", "y"}:
        command.append("--enable_trace")
    for column, cli_arg in {
        "use_score_ema": "--use_score_ema",
        "adaptive_depth_mix": "--adaptive_depth_mix",
        "use_block_coherence": "--use_block_coherence",
        "log_selection_analysis": "--log_selection_analysis",
        "log_update_cosine": "--log_update_cosine",
    }.items():
        if str(row.get(column, "")).strip().lower() in {"1", "true", "yes", "y"}:
            command.append(cli_arg)
    trace_dir = row.get("trace_dir", "")
    if trace_dir:
        command.extend(["--trace_dir", trace_dir])

    print("Manifest:", manifest_path)
    print("Row:", row_index, row["experiment_id"])
    print("Command:", " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
