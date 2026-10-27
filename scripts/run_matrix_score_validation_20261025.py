#!/usr/bin/env python3
"""Controlled CIFAR100 matrix-score and short-horizon utility experiment.

Prepare creates disjoint train/proxy/warmup/validation splits and two shared
LoRA checkpoints. Evaluate tests equal-cost configurations from a single
checkpoint, freezing unselected LoRA modules while retaining them in forward.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.client import Client
from models.lora_utils import find_lora_candidates


MODEL = "google/vit-base-patch16-224-in21k"
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class IndexedCifar(Dataset):
    def __init__(self, split, indices):
        self.split = split
        self.indices = list(map(int, indices))
        names = set(split.column_names)
        self.image_key = "image" if "image" in names else "img"
        self.label_key = "labels" if "labels" in names else "fine_label"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.split[self.indices[index]]
        return {"pixel_values": TRANSFORM(item[self.image_key].convert("RGB")),
                "labels": int(item[self.label_key])}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "evaluate"])
    p.add_argument("--distribution", required=True, choices=["iid", "dirichlet:1.0", "dirichlet:0.1"])
    p.add_argument("--seed", required=True, type=int, choices=[2024, 2025, 2026])
    p.add_argument("--client", type=int, default=0)
    p.add_argument("--checkpoint", choices=["early", "middle"], default="early")
    p.add_argument("--output-root", type=Path, default=ROOT / "paper_assets/02_experiments/matrix_score_20261025")
    p.add_argument("--pilot", action="store_true", help="Two clients, four configurations, short warmup for smoke test only.")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def path_for(a):
    token = a.distribution.replace(":", "_")
    return a.output_root / ("pilot" if a.pilot else "full") / f"cifar100_{token}_seed{a.seed}"


def load_base(seed):
    seed_all(seed)
    base = ViTForImageClassification.from_pretrained(
        MODEL, num_labels=100, ignore_mismatched_sizes=True, local_files_only=True)
    targets = find_lora_candidates(base)
    if len(targets) != 36:
        raise RuntimeError(f"Expected 36 ViT QKV candidates, got {len(targets)}")
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, target_modules=targets, lora_dropout=0.1,
        bias="none", modules_to_save=["classifier"]))
    return model, targets


def load_train():
    return load_from_disk(str(ROOT / "cifar100_local"))["train"]


def split_data(split, seed, distribution, n_clients):
    rng = np.random.default_rng(seed)
    all_idx = rng.permutation(len(split))
    proxy = all_idx[:512].tolist()
    warmup = all_idx[512:2512].tolist()
    remainder = all_idx[2512:]
    labels_key = "labels" if "labels" in split.column_names else "fine_label"
    if distribution == "iid":
        chunks = [x.tolist() for x in np.array_split(remainder, n_clients)]
    else:
        alpha = float(distribution.split(":", 1)[1])
        labels = np.asarray(split[labels_key], dtype=np.int64)
        chunks = [[] for _ in range(n_clients)]
        for class_id in np.unique(labels):
            members = rng.permutation(remainder[labels[remainder] == class_id])
            counts = rng.multinomial(len(members), rng.dirichlet([alpha] * n_clients))
            offset = 0
            for client, count in enumerate(counts):
                chunks[client].extend(members[offset:offset + count].tolist())
                offset += count
        for chunk in chunks:
            rng.shuffle(chunk)
    if min(map(len, chunks)) < 116:
        raise RuntimeError("A client has fewer than 116 samples; cannot create disjoint profile/train/validation data")
    clients = []
    for chunk in chunks:
        clients.append({"validation": chunk[:100], "train": chunk[100:]})
    used = proxy + warmup + [i for c in clients for key in ("train", "validation") for i in c[key]]
    if len(used) != len(split) or len(set(used)) != len(split):
        raise RuntimeError("Data split is not disjoint and exhaustive")
    return {"proxy": proxy, "warmup": warmup, "clients": clients}


def dataloader(split, indices, batch_size, shuffle=False, seed=0):
    return DataLoader(IndexedCifar(split, indices), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0, pin_memory=True)


def trainable_state(model):
    state = model.state_dict()
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {name: state[name].detach().cpu().clone() for name in names}


def load_trainable_state(model, state):
    own = model.state_dict()
    missing = [key for key in state if key not in own or own[key].shape != state[key].shape]
    if missing:
        raise RuntimeError(f"Checkpoint incompatible: {missing[:4]}")
    with torch.no_grad():
        for key, value in state.items():
            own[key].copy_(value)


def profile(model, loader, targets, device):
    model.eval()
    model.zero_grad(set_to_none=True)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    start = time.perf_counter()
    model(**batch).loss.backward()
    result = np.zeros(len(targets), dtype=np.float64)
    for name, param in model.named_parameters():
        if "lora_" not in name or param.grad is None:
            continue
        for index, target in enumerate(targets):
            if f".{target}." in name:
                result[index] += float(param.grad.detach().square().sum())
                break
    elapsed = time.perf_counter() - start
    model.zero_grad(set_to_none=True)
    if not np.isfinite(result).all() or max(result) <= 0:
        raise RuntimeError("Invalid or zero gradient-squared scores")
    return result, elapsed


def depth_factor(n, distribution):
    extra = 0 if distribution == "iid" else (2 if distribution.endswith("0.1") else 1)
    rho = 1 + extra
    depth = np.arange(n, dtype=np.float64) / max(1, n - 1)
    return np.power(1 + depth, rho), rho


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare(a):
    out = path_for(a)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed)
    split = load_train()
    n_clients = 2 if a.pilot else 5
    partition = split_data(split, a.seed, a.distribution, n_clients)
    write_json(out / "partition.json", partition)
    model, targets = load_base(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not a.pilot:
        raise RuntimeError("Full study requires a GPU")
    model.to(device)
    warmup_steps = 4 if a.pilot else 100
    profile_size = 16
    results = {"dataset": "cifar100", "seed": a.seed, "distribution": a.distribution,
               "rank": 8, "lora_alpha": 16, "profile_size": profile_size,
               "proxy_pool_size": len(partition["proxy"]), "warmup_steps": warmup_steps,
               "checkpoint_policy": "early=pretrained LoRA initialization; middle=warmup training on separate train-only reserve",
               "candidate_modules": targets, "stages": {}}
    for checkpoint in ("early", "middle"):
        if checkpoint == "middle":
            model.train()
            opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
            loader = dataloader(split, partition["warmup"], 16, True, a.seed + 7)
            iterator = iter(loader)
            for _ in range(warmup_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                model(**batch).loss.backward()
                opt.step()
            del opt
        state = trainable_state(model)
        torch.save(state, out / f"checkpoint_{checkpoint}.pt")
        proxy_parts = []
        proxy_sec = 0.0
        for part in range(n_clients):
            segment = partition["proxy"][part * profile_size:(part + 1) * profile_size]
            part_scores, part_sec = profile(model, dataloader(split, segment, profile_size), targets, device)
            proxy_parts.append(part_scores)
            proxy_sec += part_sec
        proxy_scores = np.mean(proxy_parts, axis=0)
        clients = []
        for client_id, client in enumerate(partition["clients"]):
            local_scores, elapsed = profile(model, dataloader(split, client["train"][:profile_size], profile_size), targets, device)
            clients.append({"client_id": client_id, "raw": local_scores.tolist(), "profile_sec": elapsed})
        mean_raw = np.mean([c["raw"] for c in clients], axis=0)
        factor, rho = depth_factor(len(targets), a.distribution)
        results["stages"][checkpoint] = {
            "rho": rho, "depth_factor": factor.tolist(), "proxy_raw": proxy_scores.tolist(),
            "proxy_profile_sec": proxy_sec, "local_mean_raw": mean_raw.tolist(), "clients": clients,
        }
    write_json(out / "scores.json", results)
    write_json(out / "metadata.json", {
        "run_id": out.name, "dataset": "cifar100", "method": "score_validation_prepare", "seed": a.seed,
        "metric": "validation_loss", "config_path": "scripts/run_matrix_score_validation_20261025.py",
        "raw_result_path": str((out / "scores.json").relative_to(ROOT)),
        "partition_path": str((out / "partition.json").relative_to(ROOT)),
        "checkpoint_paths": [str((out / f"checkpoint_{x}.pt").relative_to(ROOT)) for x in ("early", "middle")],
        "model": MODEL, "model_construction": "full 36-QKV LoRA; classifier trainable",
        "train_test_leakage": "none; only CIFAR100 train split is used",
        "pilot": a.pilot, "profile_samples_per_client": profile_size,
        "proxy_profile_samples": n_clients * profile_size,
    })
    print("PREPARED", out)


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    losses = []
    counts = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = float(model(**batch).loss)
        losses.append(loss)
        counts.append(len(batch["labels"]))
    return float(np.average(losses, weights=counts))


def feasible(config, client):
    assignment = [int(j in config) for j in range(client.total_layers)]
    memory = client.calculate_memory(assignment)
    latency = client.calculate_latency_components(assignment)[2]
    return memory <= client.M_max and latency <= 200.0, memory, latency


def build_configs(scores, client, seed, count):
    raw = np.asarray(scores["clients"][client.client_id]["raw"])
    mean = np.asarray(scores["local_mean_raw"])
    proxy = np.asarray(scores["proxy_raw"])
    factor = np.asarray(scores["depth_factor"])
    target_count = 12
    sources = {"Local": raw * factor, "Local-Mean": mean * factor,
               "Server-Proxy": proxy * factor}
    configs = {}
    method_ids = {}
    def add(name, indices):
        key = tuple(sorted(map(int, indices)))
        ok, mem, latency = feasible(key, client)
        if ok:
            configs.setdefault(key, {"indices": key, "memory_gb": mem, "latency_sec": latency})
            method_ids[name] = key
    for name, values in sources.items():
        add(name, np.argsort(-values, kind="stable")[:target_count])
    add("Local-Raw", np.argsort(-raw, kind="stable")[:target_count])
    rng = np.random.default_rng(seed + 100 * client.client_id)
    attempts = 0
    while len(configs) < count and attempts < 20000:
        attempts += 1
        add(f"Random-{attempts}", rng.choice(len(raw), target_count, replace=False))
    if len(configs) < count:
        raise RuntimeError(f"Only {len(configs)} unique feasible configurations found")
    selected = list(configs.items())[:count]
    retained = {key for key, _ in selected}
    method_ids = {method: key for method, key in method_ids.items() if key in retained and not method.startswith("Random-")}
    return selected, method_ids, sources, raw


def evaluate(a):
    out = path_for(a)
    if not (out / "scores.json").exists():
        raise RuntimeError(f"Missing prepare output: {out / 'scores.json'}")
    partition = json.loads((out / "partition.json").read_text())
    score_file = json.loads((out / "scores.json").read_text())
    if a.client >= len(partition["clients"]):
        raise ValueError("client index exceeds partition")
    score = score_file["stages"][a.checkpoint]
    split = load_train()
    seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not a.pilot:
        raise RuntimeError("Full study requires a GPU")
    model, targets = load_base(a.seed)
    state = torch.load(out / f"checkpoint_{a.checkpoint}.pt", map_location="cpu", weights_only=True)
    load_trainable_state(model, state)
    group = partition["clients"][a.client]
    client_args = SimpleNamespace(local_bs=16, local_ep=1, distribution=a.distribution,
                                  dataset="cifar100", mem_levels_gb="40:25:15")
    memory_levels = [40.0, 25.0, 15.0, 40.0, 25.0]
    client = Client(a.client, memory_levels[a.client], 1.0, 1, len(targets), len(targets),
                    model, IndexedCifar(split, group["train"]), range(len(group["train"])), client_args, -1)
    count = 4 if a.pilot else 12
    configs, method_ids, sources, raw = build_configs(score, client, a.seed, count)
    val_loader = dataloader(split, group["validation"], 32)
    train_loader_factory = lambda: dataloader(split, group["train"], 16, True, a.seed + a.client * 1000 + 29)
    steps = 3 if a.pilot else 20
    result_dir = out / "short_train"
    result_dir.mkdir(exist_ok=True)
    result_path = result_dir / f"{a.checkpoint}_client{a.client}.csv"
    rows = []
    reference_before = None
    for config_id, (indices, resource) in enumerate(configs):
        seed_all(a.seed + a.client * 1000 + 400)
        candidate_model = copy.deepcopy(model).to(device)
        active_names = [targets[j] for j in indices]
        trainable_lora_names = []
        for name, parameter in candidate_model.named_parameters():
            if "lora_" in name:
                active = any(f".{module}." in name for module in active_names)
                parameter.requires_grad_(active)
                if active:
                    trainable_lora_names.append(name)
        if len(trainable_lora_names) != 2 * len(indices):
            raise RuntimeError(f"Expected {2 * len(indices)} trainable LoRA tensors, got {len(trainable_lora_names)}")
        before = eval_loss(candidate_model, val_loader, device)
        if reference_before is None:
            reference_before = before
        elif abs(before - reference_before) > 1e-6:
            raise RuntimeError("Frozen-subset configurations do not share an identical initial validation loss")
        if config_id == 0:
            seed_all(a.seed)
            removed_base = ViTForImageClassification.from_pretrained(
                MODEL, num_labels=100, ignore_mismatched_sizes=True, local_files_only=True)
            removed_model = get_peft_model(removed_base, LoraConfig(
                r=8, lora_alpha=16, target_modules=active_names, lora_dropout=0.1,
                bias="none", modules_to_save=["classifier"]))
            subset_state = {key: value for key, value in state.items()
                            if key in removed_model.state_dict() and removed_model.state_dict()[key].shape == value.shape}
            load_trainable_state(removed_model, subset_state)
            removed_model.to(device)
            removed_loss = eval_loss(removed_model, val_loader, device)
            write_json(result_dir / f"{a.checkpoint}_client{a.client}_construction_audit.json", {
                "frozen_unselected_initial_loss": before,
                "removed_unselected_initial_loss": removed_loss,
                "difference": removed_loss - before,
                "construction_used_for_all_training": "full_adapters_forward_unselected_frozen_classifier_trainable",
                "removed_version_used_for_training": False,
                "audited_config_indices": indices,
            })
            del removed_model, removed_base
            if device.type == "cuda":
                torch.cuda.empty_cache()
        opt = torch.optim.AdamW((p for p in candidate_model.parameters() if p.requires_grad), lr=1e-3)
        start = time.perf_counter()
        candidate_model.train()
        iterator = iter(train_loader_factory())
        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader_factory())
                batch = next(iterator)
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            candidate_model(**batch).loss.backward()
            opt.step()
        train_sec = time.perf_counter() - start
        after = eval_loss(candidate_model, val_loader, device)
        row = {
            "run_id": out.name, "dataset": "cifar100", "method": "+".join(k for k, v in method_ids.items() if v == indices) or "Random",
            "seed": a.seed, "metric": "validation_loss_reduction", "config_path": "scripts/run_matrix_score_validation_20261025.py",
            "raw_result_path": str(result_path.relative_to(ROOT)), "distribution": a.distribution,
            "checkpoint": a.checkpoint, "client_id": a.client, "config_id": config_id,
            "matrix_indices": json.dumps(indices), "matrix_count": len(indices), "rank": 8,
            "memory_gb": resource["memory_gb"], "modeled_latency_sec": resource["latency_sec"],
            "train_steps": steps, "train_seconds": train_sec, "val_loss_before": before,
            "val_loss_after": after, "val_loss_reduction": before - after,
            "pred_local_raw": float(raw[list(indices)].sum()),
            "pred_local_depth": float(sources["Local"][list(indices)].sum()),
            "pred_local_mean_depth": float(sources["Local-Mean"][list(indices)].sum()),
            "pred_proxy_depth": float(sources["Server-Proxy"][list(indices)].sum()),
            "model_construction": "full_adapters_forward_unselected_frozen_classifier_trainable",
        }
        rows.append(row)
        del candidate_model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{a.checkpoint} client={a.client} config={config_id} gain={before-after:.5f}", flush=True)
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gains = np.array([r["val_loss_reduction"] for r in rows])
    summary = {name: float(spearmanr([r[column] for r in rows], gains).statistic)
               for name, column in (("raw", "pred_local_raw"), ("depth", "pred_local_depth"))}
    write_json(result_dir / f"{a.checkpoint}_client{a.client}_summary.json", summary)
    print("SAVED", result_path, summary)


def main():
    a = args()
    if a.stage == "prepare":
        prepare(a)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
