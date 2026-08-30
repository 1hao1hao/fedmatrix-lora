#!/usr/bin/env python3
"""Run one reproducible SST-2/RoBERTa matrix-position experiment."""

from __future__ import annotations

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
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models_zoo/roberta-base"
DATASET_PATH = ROOT / "data_local/sst2"
STRATEGIES = (
    "shallow_qkv",
    "deep_qkv",
    "q_only",
    "k_only",
    "v_only",
    "all_qkv_reference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def target_modules(model, strategy: str) -> list[str]:
    selected = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
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
def evaluate_accuracy(model, loader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        predictions = model(**batch).logits.argmax(dim=-1)
        labels = batch["labels"]
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    if was_training:
        model.train()
    return correct / max(1, total)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "[TODO-METADATA]"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"SST2_MATRIX_POSITION_{args.strategy}_seed{args.seed}"
    output_dir = args.output_root.resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    dataset = load_from_disk(str(DATASET_PATH))

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, load_from_cache_file=True)
    keep_columns = {"input_ids", "attention_mask", "label"}
    remove_columns = [
        column for column in tokenized["train"].column_names if column not in keep_columns
    ]
    tokenized = tokenized.remove_columns(remove_columns)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        tokenized["validation"],
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
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
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps,
    )

    script_path = Path(__file__).resolve()
    metadata = {
        "run_id": run_id,
        "dataset": "sst2",
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "method": args.strategy,
        "seed": args.seed,
        "primary_metric": "validation_accuracy",
        "config_path": str(script_path.relative_to(ROOT)),
        "raw_result_path": portable_path(output_dir / "metrics.csv"),
        "model": str(MODEL_PATH.relative_to(ROOT)),
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
        "scheduler": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "validation_size": len(tokenized["validation"]),
        "device": str(device),
        "git_commit": git_commit(),
        "script_sha256": file_sha256(script_path),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)

    records = []
    start_time = time.perf_counter()
    initial_accuracy = evaluate_accuracy(model, validation_loader, device)
    records.append(
        {
            "run_id": run_id,
            "dataset": "sst2",
            "method": args.strategy,
            "seed": args.seed,
            "step": 0,
            "train_loss": "",
            "validation_accuracy": initial_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
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
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        validation_accuracy = ""
        if step % args.eval_interval == 0 or step == args.max_steps:
            validation_accuracy = evaluate_accuracy(model, validation_loader, device)
        records.append(
            {
                "run_id": run_id,
                "dataset": "sst2",
                "method": args.strategy,
                "seed": args.seed,
                "step": step,
                "train_loss": float(loss.item()),
                "validation_accuracy": validation_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_sec": time.perf_counter() - start_time,
            }
        )
        if step % args.eval_interval == 0 or step == args.max_steps:
            print(
                f"run_id={run_id} step={step}/{args.max_steps} "
                f"loss={loss.item():.6f} val_acc={validation_accuracy}",
                flush=True,
            )

    with (output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    evaluated = [
        float(row["validation_accuracy"])
        for row in records
        if row["validation_accuracy"] != ""
    ]
    summary = {
        **metadata,
        "initial_validation_accuracy": initial_accuracy,
        "final_validation_accuracy": evaluated[-1],
        "best_validation_accuracy": max(evaluated),
        "elapsed_sec": time.perf_counter() - start_time,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
