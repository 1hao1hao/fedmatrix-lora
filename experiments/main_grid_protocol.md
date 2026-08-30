# Main Grid Experiment Protocol

Scope: this protocol locks the naming and execution fields for the remaining main-grid experiments. It does not claim that historical raw CSVs are complete unless a run has dataset, distribution, method, seed, config/slurm path, run_id, and raw result path metadata.

## Naming Rules

| Field | Rule |
|---|---|
| `dataset` | Lowercase code used by `main_drl.py`: `cifar100`, `sst2`, `qnli`, `qqp`, `mnli`, `samsum`. |
| `task_family` | `classification` for CIFAR100/SST2/QNLI/QQP/MNLI; `summarization` for SAMSum. |
| `distribution_arg` | CLI value passed to `--distribution`: `iid`, `dirichlet:1.0`, `dirichlet:0.1`, `length_skew:mild`, or `length_skew:severe`. |
| `distribution_token` | Experiment-ID token: `iid`, `dirichlet_1p0`, `dirichlet_0p1`, `length_skew_mild`, `length_skew_severe`. |
| `distribution_label` | Human/table label: `IID`, `Dirichlet-1.0`, `Dirichlet-0.1`, `length-skew mild`, `length-skew severe`. |
| `method_raw` | Method passed to `--test_method`, e.g. `Random_FIM`, `Random_Taylor`, `FedAvg_LoRA_homogeneous`, `Uniform_allocation`. |
| `method_paper` | Paper-facing method name from `experiments/method_name_mapping.csv`, e.g. `Ours-FIM`, `Ours-Taylor`, `FedAvg-LoRA homogeneous`, `Uniform allocation`. |
| `experiment_id` | `MAIN_{Dataset}_{distribution_token}_{MethodToken}` where `Dataset` keeps paper casing (`SAMSum`) and method tokens use underscores. |
| `seed` | Main-grid reruns use `2024` unless a multi-seed extension is explicitly added later. Historical runs without archived seed remain `[TODO-SEED]`. |
| `run_id` | Runtime directory remains code-generated as `Dist-{safe_distribution}_Bgt-{total_budget}_{timestamp}`. The stable paper key is `experiment_id`; each run writes `metadata.json` binding both keys. |

## Distribution Contract

| Dataset group | Main-grid distributions |
|---|---|
| Classification datasets: CIFAR100, SST2, QNLI, QQP, MNLI | `iid`, `dirichlet:1.0`, `dirichlet:0.1` |
| SAMSum | `iid`, `length_skew:mild`, `length_skew:severe` |

SAMSum must not be described as using Dirichlet heterogeneity in the main grid. Its non-IID settings are length-skew settings.

## Main-Grid Method Set

| Raw method | Paper method | Role |
|---|---|---|
| `FedAvg_LoRA_homogeneous` | FedAvg-LoRA homogeneous | Homogeneous LoRA/FedAvg reference baseline. |
| `Uniform_allocation` | Uniform allocation | Non-importance uniform matrix allocation baseline. |
| `Fed-HeLLo` | Fed-HeLLo | Proxy/global-importance baseline. |
| `LEGEND` | LEGEND | Heterogeneous-rank/layer-wise baseline boundary; not the proposed method. |
| `MS` | MS | Memory-saving physical-layer heuristic baseline. |
| `Random_FIM` | Ours-FIM | Proposed matrix-level DP/MCKP with local FIM profiling. |
| `Random_Taylor` | Ours-Taylor | Proposed matrix-level DP/MCKP with local Taylor profiling. |

## Files

| File | Purpose |
|---|---|
| `experiments/main_grid_manifest.csv` | Atomic experiment manifest: one row per dataset/distribution/method paper result. |
| `experiments/critical_gap_manifest.csv` | Job-oriented manifest for the immediate missing main-grid experiments; some rows intentionally run multiple methods in one Slurm job. |
| `experiments/ablation_representative_manifest.csv` | Representative six-ablation manifest; each row enables trace output and isolates one component claim. |
| `run_manifest_array.slurm` | Slurm array entry that reads a manifest row and launches `main_drl.py`. |

## Immediate Execution Target

Run the missing critical main-grid jobs first:

```bash
sbatch --array=1-16 run_manifest_array.slurm experiments/critical_gap_manifest.csv
```

For one-method-per-job tracking, use the atomic manifest:

```bash
sbatch --array=1-35 run_manifest_array.slurm experiments/main_grid_manifest.csv
```

Do not submit both manifests for the same settings at the same time unless duplicate reruns are intentional.
