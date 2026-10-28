#!/usr/bin/env python3
"""Fixed-protocol P2: paired matrix selection versus whole-layer binding.

Prepare writes all four masks before any P2 training. Train consumes those
masks, reuses protocol-matched non-first P rows, and retrains P's first mask
with the same RNG reset used for every newly trained configuration.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_matrix_score_validation_20261025 import (  # noqa: E402
    Client, IndexedCifar, dataloader, eval_loss, feasible, load_base,
    load_train, load_trainable_state, seed_all,
)
from models.lora_utils import infer_lora_candidate_metadata  # noqa: E402

P_ROOT = ROOT / "paper_assets/02_experiments/matrix_score_20261025/full"
OUT_ROOT = ROOT / "paper_assets/02_experiments/matrix_granularity_p2_20261027"
METHODS = ("Local-Matrix", "Local-Layer", "Proxy-Matrix", "Proxy-Layer")
MEMORY_LEVELS = (40.0, 25.0, 15.0, 40.0, 25.0)
TRAIN_STEPS = 20


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train"))
    parser.add_argument("--distribution", required=True, choices=("iid", "dirichlet:1.0", "dirichlet:0.1"))
    parser.add_argument("--seed", required=True, type=int, choices=(2024, 2025, 2026))
    parser.add_argument("--client", type=int, choices=range(5), default=0)
    parser.add_argument("--checkpoint", choices=("early", "middle"), default="early")
    return parser.parse_args()


def folder_for(a):
    return OUT_ROOT / f"p2_cifar100_{a.distribution.replace(':', '_')}_seed{a.seed}"


def p_folder_for(a):
    return P_ROOT / f"cifar100_{a.distribution.replace(':', '_')}_seed{a.seed}"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_client(model, split, partition, a, client_id):
    group = partition["clients"][client_id]
    client_args = SimpleNamespace(local_bs=16, local_ep=1, distribution=a.distribution,
                                  dataset="cifar100", mem_levels_gb="40:25:15")
    # The original P evaluates each client in a fresh process seeded identically.
    seed_all(a.seed)
    client = Client(client_id, MEMORY_LEVELS[client_id], 1.0, 1, 36, 36,
                    model, IndexedCifar(split, group["train"]),
                    range(len(group["train"])), client_args, -1)
    return client


def candidate_groups(targets):
    groups = {}
    for index, target in enumerate(targets):
        meta = infer_lora_candidate_metadata(target)
        groups.setdefault(int(meta["block_id"]), []).append((index, meta["module_type"]))
    if (set(groups) != set(range(12)) or
            any(sorted(kind for _, kind in entries) != ["k", "q", "v"]
                for entries in groups.values())):
        raise RuntimeError("P2 requires 12 ViT blocks containing exactly Q, K, and V")
    return {block: [index for index, _ in entries] for block, entries in groups.items()}


def best_matrix(scores, client):
    # For fixed count, this model's memory depends on the earliest selected
    # position; the suffix with the largest remaining scores is optimal there.
    candidates = []
    for earliest in range(36 - 12 + 1):
        suffix = sorted(range(earliest + 1, 36), key=lambda j: (-scores[j], j))[:11]
        if len(suffix) != 11:
            continue
        mask = tuple(sorted((earliest, *suffix)))
        allowed, memory, latency = feasible(mask, client)
        if allowed:
            candidates.append((float(scores[list(mask)].sum()), mask, memory, latency))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (-item[0], item[1]))


def best_layer(scores, client, groups):
    candidates = []
    for blocks in itertools.combinations(sorted(groups), 4):
        mask = tuple(sorted(j for block in blocks for j in groups[block]))
        allowed, memory, latency = feasible(mask, client)
        if allowed:
            candidates.append((float(scores[list(mask)].sum()), mask, memory, latency))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (-item[0], item[1]))


def parameter_count(model, targets, mask):
    selected = [targets[j] for j in mask]
    return sum(parameter.numel() for name, parameter in model.named_parameters()
               if ("lora_" in name and any(f".{module}." in name for module in selected))
               or ("lora_" not in name and parameter.requires_grad))


def prepare(a):
    source = p_folder_for(a)
    dest = folder_for(a)
    if not all((source / name).exists() for name in ("scores.json", "partition.json",
                                                     "checkpoint_early.pt", "checkpoint_middle.pt")):
        raise RuntimeError(f"Original P preparation is incomplete: {source}")
    data = json.loads((source / "scores.json").read_text())
    partition = json.loads((source / "partition.json").read_text())
    if data["distribution"] != a.distribution or int(data["seed"]) != a.seed or data["rank"] != 8:
        raise RuntimeError("Original P metadata do not match the fixed P2 protocol")
    split = load_train()
    model, targets = load_base(a.seed)
    if targets != data["candidate_modules"] or len(partition["clients"]) != 5:
        raise RuntimeError("Original P candidates or client count changed")
    rows = []
    for checkpoint in ("early", "middle"):
        stage = data["stages"][checkpoint]
        factor = np.asarray(stage["depth_factor"], dtype=np.float64)
        for client_id in range(5):
            client = build_client(model, split, partition, a, client_id)
            groups = candidate_groups(targets)
            sources = {"Local": np.asarray(stage["clients"][client_id]["raw"], dtype=np.float64),
                       "Proxy": np.asarray(stage["proxy_raw"], dtype=np.float64)}
            for source_name, raw in sources.items():
                calibrated = raw * factor
                divisor = max(float(calibrated.max()), 1e-12)
                normalized = calibrated / divisor
                for granularity, selector in (("Matrix", lambda: best_matrix(normalized, client)),
                                              ("Layer", lambda: best_layer(normalized, client, groups))):
                    chosen = selector()
                    method = f"{source_name}-{granularity}"
                    mask = chosen[1] if chosen else ()
                    rows.append({"run_id": dest.name, "dataset": "cifar100", "method": method,
                                 "seed": a.seed, "metric": "validation_loss_reduction",
                                 "config_path": "scripts/run_matrix_granularity_p2.py",
                                 "raw_result_path": str((dest / "masks.csv").relative_to(ROOT)),
                                 "distribution": a.distribution, "checkpoint": checkpoint,
                                 "client_id": client_id, "status": "feasible" if chosen else "infeasible",
                                 "mask": json.dumps(mask), "matrix_count": len(mask), "rank": 8,
                                 "score_source": source_name, "depth_rho": stage["rho"],
                                 "normalization_divisor": divisor,
                                 "raw_score_sum": float(raw[list(mask)].sum()) if chosen else "",
                                 "calibrated_score_sum": float(calibrated[list(mask)].sum()) if chosen else "",
                                 "normalized_score_sum": chosen[0] if chosen else "",
                                 "memory_gb": chosen[2] if chosen else "",
                                 "modeled_latency_sec": chosen[3] if chosen else "",
                                 "trainable_parameters": parameter_count(model, targets, mask) if chosen else "",
                                 "feasibility_rule": "memory<=client_limit; latency<=200s"})
    dest.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(dest / "masks.csv", rows)
    (dest / "metadata.json").write_text(json.dumps({
        "run_id": dest.name, "dataset": "cifar100", "method": list(METHODS), "seed": a.seed,
        "distribution": a.distribution, "metric": "validation_loss_reduction",
        "config_path": "scripts/run_matrix_granularity_p2.py",
        "raw_result_path": str((dest / "masks.csv").relative_to(ROOT)),
        "original_p_root": str(source.relative_to(ROOT)), "rank": 8, "matrix_count": 12,
        "train_steps": TRAIN_STEPS, "optimizer": "AdamW", "learning_rate": 1e-3,
        "model_construction": "full_adapters_forward_unselected_frozen_classifier_trainable",
        "same_source_score_rule": "raw * original_P_depth_factor / max(raw * factor)",
    }, indent=2), encoding="utf-8")
    print("PREPARED", dest, "masks", len(rows), "infeasible", sum(r["status"] != "feasible" for r in rows))


def old_p_row(a, checkpoint, client_id, method, mask, p2_mask_row):
    path = p_folder_for(a) / "short_train" / f"{checkpoint}_client{client_id}.csv"
    old_method = "Local" if method == "Local-Matrix" else "Server-Proxy"
    matches = [row for row in read_csv(path) if old_method in row["method"].split("+")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one original P {old_method} result in {path}")
    row = matches[0]
    if (tuple(json.loads(row["matrix_indices"])) != mask or
            row["run_id"] != p_folder_for(a).name or row["distribution"] != a.distribution or
            int(row["seed"]) != a.seed or row["checkpoint"] != checkpoint or
            int(row["client_id"]) != client_id or int(row["matrix_count"]) != 12 or
            int(row["rank"]) != 8 or int(row["train_steps"]) != TRAIN_STEPS or
            row["metric"] != "validation_loss_reduction" or
            row["config_path"] != "scripts/run_matrix_score_validation_20261025.py" or
            abs(float(row["memory_gb"]) - float(p2_mask_row["memory_gb"])) > 1e-9 or
            abs(float(row["modeled_latency_sec"]) - float(p2_mask_row["modeled_latency_sec"])) > 1e-9 or
            row["model_construction"] != "full_adapters_forward_unselected_frozen_classifier_trainable"):
        raise RuntimeError(f"Original P row is not reusable under P2: {path}")
    return row, path


def train_one(model, targets, mask, split, group, a, client_id, device):
    candidate = copy.deepcopy(model).to(device)
    chosen = [targets[j] for j in mask]
    for name, parameter in candidate.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(any(f".{module}." in name for module in chosen))
    trainable_lora = [name for name, parameter in candidate.named_parameters()
                      if "lora_" in name and parameter.requires_grad]
    if len(trainable_lora) != 24:
        raise RuntimeError(f"Expected 24 LoRA tensors, got {len(trainable_lora)}")
    # Match the non-first P configurations: their seeded validation iterator
    # advances the RNG before training. Restore that post-validation state for
    # every P2 mask, including the repaired first configuration.
    seed_all(a.seed + client_id * 1000 + 400)
    before = eval_loss(candidate, dataloader(split, group["validation"], 32), device)
    paired_cpu_rng = torch.get_rng_state()
    paired_cuda_rng = torch.cuda.get_rng_state_all()
    optimizer = torch.optim.AdamW((p for p in candidate.parameters() if p.requires_grad), lr=1e-3)
    factory = lambda: dataloader(split, group["train"], 16, True, a.seed + client_id * 1000 + 29)
    torch.set_rng_state(paired_cpu_rng)
    torch.cuda.set_rng_state_all(paired_cuda_rng)
    candidate.train()
    iterator = iter(factory())
    start = time.perf_counter()
    for _ in range(TRAIN_STEPS):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(factory())
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        candidate(**batch).loss.backward()
        optimizer.step()
    duration = time.perf_counter() - start
    after = eval_loss(candidate, dataloader(split, group["validation"], 32), device)
    del candidate, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"val_loss_before": before, "val_loss_after": after,
            "val_loss_reduction": before - after, "train_seconds": duration,
            "result_source": "p2_repaired_rng_or_new_layer", "source_raw_result_path": ""}


def train(a):
    dest = folder_for(a)
    mask_file = dest / "masks.csv"
    if not mask_file.exists():
        raise RuntimeError("All P2 masks must be prepared before training")
    masks = [r for r in read_csv(mask_file) if r["checkpoint"] == a.checkpoint and int(r["client_id"]) == a.client]
    if [r["method"] for r in masks] != list(METHODS):
        raise RuntimeError("Missing or reordered P2 masks")
    result_file = dest / "results" / f"{a.checkpoint}_client{a.client}.csv"
    if result_file.exists():
        print("ALREADY_COMPLETE", result_file)
        return
    seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P2 full training requires a GPU")
    split = load_train()
    partition = json.loads((p_folder_for(a) / "partition.json").read_text())
    model, targets = load_base(a.seed)
    state = torch.load(p_folder_for(a) / f"checkpoint_{a.checkpoint}.pt",
                       map_location="cpu", weights_only=True)
    load_trainable_state(model, state)
    group = partition["clients"][a.client]
    client = build_client(model, split, partition, a, a.client)
    outcomes = {}
    for row in masks:
        if row["status"] != "feasible":
            continue
        mask = tuple(json.loads(row["mask"]))
        if mask in outcomes:
            continue
        matrix_row = next((r for r in masks if r["method"].endswith("Matrix")
                           and tuple(json.loads(r["mask"])) == mask), None)
        old = old_p_row(a, a.checkpoint, a.client, matrix_row["method"], mask,
                        matrix_row) if matrix_row else None
        if old and int(old[0]["config_id"]) > 0:
            old_row, old_path = old
            outcomes[mask] = {"val_loss_before": float(old_row["val_loss_before"]),
                              "val_loss_after": float(old_row["val_loss_after"]),
                              "val_loss_reduction": float(old_row["val_loss_reduction"]),
                              "train_seconds": float(old_row["train_seconds"]),
                              "result_source": "reused_original_p_nonfirst",
                              "source_raw_result_path": str(old_path.relative_to(ROOT))}
        else:
            outcomes[mask] = train_one(model, targets, mask, split, group, a, a.client, device)
        expected_params = int(row["trainable_parameters"])
        if expected_params != parameter_count(model, targets, mask):
            raise RuntimeError("Trainable-parameter accounting mismatch")
    baselines = [float(item["val_loss_before"]) for item in outcomes.values()]
    if baselines and max(baselines) - min(baselines) > 1e-6:
        raise RuntimeError("Paired configurations do not share the same initial validation loss")
    result_rows = []
    for row in masks:
        mask = tuple(json.loads(row["mask"]))
        result = outcomes.get(mask, {})
        result_rows.append({**row, "raw_result_path": str(result_file.relative_to(ROOT)),
                            **result, "train_steps": TRAIN_STEPS if result else "",
                            "paired_training_seed": a.seed + a.client * 1000 + 400 if result else ""})
    write_csv_atomic(result_file, result_rows)
    print("SAVED", result_file, "unique_trained_or_reused", len(outcomes),
          "reused", sum(r["result_source"] == "reused_original_p_nonfirst" for r in outcomes.values()))


def main():
    a = args()
    if a.stage == "prepare":
        prepare(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
