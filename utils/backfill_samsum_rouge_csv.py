"""Backfill Rouge_l into historical SAMSum evaluation CSV files.

Older runs computed Rouge-L and printed it to Slurm stdout, but did not export
the Rouge_l column because of a CSV column-name typo. This utility repairs those
CSV files so downstream plotting/analysis can read Rouge_l directly.
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


DEFAULT_COLUMNS = [
    "Round",
    "Loss",
    "Raw_LM_Loss",
    "Rouge_l",
    "Accuracy",
    "T_Actual",
    "T_Comp_Avg",
    "T_Comm_Avg",
    "Cumulative_Time",
    "Consumed_Budget",
    "Cumulative_Budget",
    "Remaining_Budget",
    "Avg_Memory_GB",
    "Avg_Comm_MB",
    "Avg_Flops",
    "Action_Value",
]


def find_log_for_run(run_name, logs_dir):
    for log_path in glob.glob(os.path.join(logs_dir, "*.out")):
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
                content = handle.read()
        except OSError:
            continue

        if run_name in content:
            return content

    return ""


def extract_rouge_by_strategy(log_content, strategy_name):
    pattern = re.compile(
        rf"\[{re.escape(strategy_name)}\]\s+Round\s+(\d+):.*?rouge-l=([0-9.]+)"
    )
    return {
        int(round_id): float(rouge_l)
        for round_id, rouge_l in pattern.findall(log_content)
    }


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    rename_map = {}
    for col in df.columns:
        normalized = col.lower().replace("-", "_")
        if normalized in {"rouge_l", "rougel", "rouge_l_score"}:
            rename_map[col] = "Rouge_l"
        elif normalized in {"raw_lm_loss", "lm_loss", "raw_loss"}:
            rename_map[col] = "Raw_LM_Loss"
    return df.rename(columns=rename_map)


def repair_csv(csv_path, log_content, write=True):
    strategy_name = os.path.basename(csv_path).replace("eval_", "").replace(".csv", "")
    df = normalize_columns(pd.read_csv(csv_path))

    rouge_existing = (
        "Rouge_l" in df.columns
        and pd.to_numeric(df["Rouge_l"], errors="coerce").fillna(0).sum() > 0
    )
    if rouge_existing:
        return False, f"{strategy_name}: Rouge_l already present"

    rouge_by_round = extract_rouge_by_strategy(log_content, strategy_name)
    if not rouge_by_round:
        return False, f"{strategy_name}: no Rouge-L values found in logs"

    if "Round" not in df.columns:
        return False, f"{strategy_name}: missing Round column"

    recovered = df["Round"].map(lambda step: rouge_by_round.get(int(step), np.nan))
    if not recovered.notna().any():
        return False, f"{strategy_name}: no matching rounds found"

    if "Raw_LM_Loss" not in df.columns and "Loss" in df.columns:
        df["Raw_LM_Loss"] = df["Loss"]

    df["Rouge_l"] = recovered.ffill()

    loss_values = pd.to_numeric(df["Loss"], errors="coerce") if "Loss" in df.columns else pd.Series(dtype=float)
    if "Loss" not in df.columns or not np.isfinite(loss_values.dropna()).any():
        df["Loss"] = 1.0 - pd.to_numeric(df["Rouge_l"], errors="coerce").clip(lower=0.0, upper=1.0)

    ordered_columns = [col for col in DEFAULT_COLUMNS if col in df.columns]
    extra_columns = [col for col in df.columns if col not in ordered_columns]
    df = df[ordered_columns + extra_columns]

    if write:
        df.to_csv(csv_path, index=False)

    return True, f"{strategy_name}: backfilled {df['Rouge_l'].notna().sum()} Rouge_l values"


def repair_run_dir(run_dir, logs_dir, write=True):
    run_name = os.path.basename(run_dir.rstrip(os.sep))
    log_content = find_log_for_run(run_name, logs_dir)
    if not log_content:
        raise FileNotFoundError(f"No Slurm stdout log found for run: {run_name}")

    csv_files = sorted(glob.glob(os.path.join(run_dir, "eval_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No eval_*.csv files found in: {run_dir}")

    changed = 0
    for csv_path in csv_files:
        did_change, message = repair_csv(csv_path, log_content, write=write)
        changed += int(did_change)
        print(f"{os.path.basename(csv_path)}: {message}")

    return changed


def main():
    parser = argparse.ArgumentParser(description="Backfill Rouge_l into historical SAMSum CSV files.")
    parser.add_argument("run_dirs", nargs="+", help="SAMSum evaluation result directories to repair.")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing Slurm .out logs.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing CSV files.")
    args = parser.parse_args()

    total_changed = 0
    for run_dir in args.run_dirs:
        print(f"==> Repairing {run_dir}")
        total_changed += repair_run_dir(run_dir, args.logs_dir, write=not args.dry_run)

    action = "would update" if args.dry_run else "updated"
    print(f"Done: {action} {total_changed} CSV files.")


if __name__ == "__main__":
    main()
