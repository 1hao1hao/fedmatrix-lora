#!/usr/bin/env python3
"""Run one reproducible CIFAR-100 matrix-position motivation experiment."""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "google/vit-base-patch16-224-in21k"
STRATEGIES = (
    "shallow_qkv",
    "deep_qkv",
    "q_only",
    "k_only",
    "v_only",
    "all_qkv_reference",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def set_reproducible_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CifarDataset(Dataset):
    def __init__(self, split, transform):
        self.split = split
        self.transform = transform
        columns = set(split.column_names)
        self.image_column = "image" if "image" in columns else "img"
        self.label_column = "labels" if "labels" in columns else "fine_label"

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        item = self.split[index]
        image = item[self.image_column].convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "labels": int(item[self.label_column]),
        }


def load_cifar100():
    local_path = ROOT / "cifar100_local"
    if local_path.exists():
        return load_from_disk(str(local_path)), str(local_path)
    return load_dataset("cifar100"), "huggingface:cifar100"


def target_modules(model, strategy):
    selected = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or "attention" not in name:
            continue
        match = re.search(r"\.layer\.(\d+)\..*\.(query|key|value)$", name)
        if not match:
            continue
        layer = int(match.group(1))
        matrix_type = match.group(2)
        include = (
            (strategy == "shallow_qkv" and layer < 4)
            or (strategy == "deep_qkv" and layer >= 8)
            or (strategy == "q_only" and matrix_type == "query")
            or (strategy == "k_only" and matrix_type == "key")
            or (strategy == "v_only" and matrix_type == "value")
            or strategy == "all_qkv_reference"
        )
        if include:
            selected.append(name)
    expected = 36 if strategy == "all_qkv_reference" else 12
    if len(selected) != expected:
        raise RuntimeError(
            f"Strategy {strategy} selected {len(selected)} matrices; expected {expected}."
        )
    return sorted(selected)


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        predictions = model(pixel_values=pixel_values).logits.argmax(dim=-1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    if was_training:
        model.train()
    return correct / total


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "[TODO-METADATA]"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    set_reproducible_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"CIFAR100_MATRIX_POSITION_{args.strategy}_seed{args.seed}"
    output_dir = args.output_root.resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset, dataset_path = load_cifar100()
    train_dataset = CifarDataset(dataset["train"], image_transform)
    test_dataset = CifarDataset(dataset["test"], image_transform)
    validation_size = min(args.validation_size, len(test_dataset))
    validation_dataset = torch.utils.data.Subset(
        test_dataset, list(range(validation_size))
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=100,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    selected_modules = target_modules(model, args.strategy)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=selected_modules,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        ),
    ).to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate
    )

    script_path = Path(__file__).resolve()
    metadata = {
        "run_id": run_id,
        "dataset": "cifar100",
        "dataset_path": dataset_path,
        "method": args.strategy,
        "seed": args.seed,
        "primary_metric": "full_test_accuracy",
        "config_path": str(script_path.relative_to(ROOT)),
        "raw_result_path": str((output_dir / "metrics.csv").relative_to(ROOT)),
        "model": MODEL_NAME,
        "lora_rank": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "matrix_budget": len(selected_modules),
        "selected_modules": selected_modules,
        "trainable_parameters": trainable_parameters,
        "max_steps": args.max_steps,
        "eval_interval": args.eval_interval,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "validation_size": validation_size,
        "test_size": len(test_dataset),
        "transform": "Resize(224), ToTensor, ImageNet normalization",
        "device": str(device),
        "git_commit": git_commit(),
        "script_sha256": file_sha256(script_path),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)

    records = []
    start_time = time.perf_counter()
    initial_accuracy = evaluate_accuracy(model, validation_loader, device)
    records.append(
        {
            "run_id": run_id,
            "dataset": "cifar100",
            "method": args.strategy,
            "seed": args.seed,
            "step": 0,
            "train_loss": "",
            "validation_accuracy": initial_accuracy,
            "elapsed_sec": time.perf_counter() - start_time,
        }
    )

    model.train()
    train_iterator = iter(train_loader)
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = model(pixel_values=pixel_values, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        validation_accuracy = ""
        if step % args.eval_interval == 0 or step == args.max_steps:
            validation_accuracy = evaluate_accuracy(model, validation_loader, device)
        records.append(
            {
                "run_id": run_id,
                "dataset": "cifar100",
                "method": args.strategy,
                "seed": args.seed,
                "step": step,
                "train_loss": float(loss.item()),
                "validation_accuracy": validation_accuracy,
                "elapsed_sec": time.perf_counter() - start_time,
            }
        )
        if step % 25 == 0 or step == args.max_steps:
            print(
                f"run_id={run_id} step={step}/{args.max_steps} "
                f"loss={loss.item():.6f} val_acc={validation_accuracy}",
                flush=True,
            )

    full_test_accuracy = evaluate_accuracy(model, test_loader, device)
    fieldnames = list(records[0].keys())
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary = {
        **metadata,
        "initial_validation_accuracy": initial_accuracy,
        "final_validation_accuracy": float(records[-1]["validation_accuracy"]),
        "best_validation_accuracy": max(
            float(row["validation_accuracy"])
            for row in records
            if row["validation_accuracy"] != ""
        ),
        "full_test_accuracy": full_test_accuracy,
        "elapsed_sec": time.perf_counter() - start_time,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
