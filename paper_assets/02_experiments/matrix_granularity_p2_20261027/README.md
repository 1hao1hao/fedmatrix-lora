# P2: fixed-budget matrix versus whole-layer allocation

Status (2026-10-28): all 90 training cells and the strict paired analysis completed.

## Read the results

- [Short conclusion and all 12 setting means](../../05_figures/experiment_p2_20261027/summary_zh.md)
- [Paired gain figure, PDF](../../05_figures/experiment_p2_20261027/experiment_P2.pdf) ([PNG](../../05_figures/experiment_p2_20261027/experiment_P2.png))
- [All method-level results](../../05_figures/experiment_p2_20261027/all_results.csv), [all pre-generated masks](../../05_figures/experiment_p2_20261027/all_masks.csv), and [per-matrix score inputs](../../05_figures/experiment_p2_20261027/score_inputs.csv)
- [Client-level paired differences](../../05_figures/experiment_p2_20261027/paired_client_differences.csv), [seed-level paired differences](../../05_figures/experiment_p2_20261027/paired_seed_differences.csv), [three-seed summary](../../05_figures/experiment_p2_20261027/paired_summary.csv), and [run ledger](../../05_figures/experiment_p2_20261027/run_ledger.csv)

The Matrix-minus-Layer validation-loss-reduction mean is positive in all 12 partition/checkpoint/source settings. It is positive in 35 of 36 seed-level paired means and 161 of 180 client-level pairs. The negative seed-level case is IID, middle checkpoint, Proxy, seed 2024 (-0.00085); 19 negative client-level cases are retained in the CSV. This is evidence for this fixed 20-step proxy-quality comparison, not a guarantee of superior final accuracy or convergence. The figure uses independent y-axis scales for the three partitions.

- Dataset/model: CIFAR100 / ViT-base, original P's disjoint split and early/middle checkpoints.
- Distributions: IID, Dirichlet 1.0, Dirichlet 0.1; seeds: 2024, 2025, 2026; five clients per seed/distribution.
- Methods: Local-Matrix, Local-Layer, Proxy-Matrix, Proxy-Layer; rank 8, twelve selected Q/K/V matrices, 20 AdamW steps at learning rate 1e-3.
- Scores: original P's raw local/proxy scores, depth factors, and within-source normalization. Matrix maximizes the score over feasible twelve-matrix subsets; Layer maximizes over feasible combinations of four complete Q/K/V blocks. Both use the same original P memory and latency tests.
- All 360 method masks were generated before P2 training; 360/360 pass the fixed resource limits. All 180 Matrix masks match their original P counterparts. There are 90 paired client/checkpoint cells: 38 have four unique masks, 45 have three, and seven have two.
- Original P config_id=0 is not reused because its removed-adapter audit disturbed the training RNG. Non-first P Matrix records are reused only after checking mask, configuration, checkpoint, seed, metric, and construction protocol. New P2 training restores the paired post-validation RNG state before the optimizer step and batch iteration. Distinct masks within a paired cell share one training outcome.
- Completed unique new trainings: 90 repaired first-configuration masks plus 154 new Layer masks, totaling 244. Another 57 unique non-first Matrix masks were reused; this accounting retains every method record even when masks coincide.
- Primary metric: validation-loss reduction. The paired contrast is Matrix reduction minus Layer reduction, aggregated across clients before the three-seed mean and sample standard deviation.
- The original P depth exponent varies by partition; differences between partitions cannot be attributed solely to data heterogeneity.

## Jobs

| Role | Slurm job | Dependency |
|---|---:|---|
| GPU pilot, array index 0 | 1776409 | none |
| Remaining GPU cells, indices 1-89, at most 12 concurrent | 1776411 | pilot afterok |
| Strict paired analysis and plot | 1776413 | both GPU jobs afterok |

All listed jobs finished successfully. The combined files contain 360 complete method records, no resource-infeasible records, and 6480 per-matrix score inputs. Memory and latency columns are model estimates, not measured GPU peaks or wall-clock latency. All 12-matrix configurations have 224356 trainable parameters under the original P classifier rule.

Source protocol: `scripts/run_matrix_score_validation_20261025.py`.
P2 runner: `scripts/run_matrix_granularity_p2.py`.
P2 analysis: `scripts/analyze_matrix_granularity_p2.py`.
Raw masks: `p2_cifar100_<distribution>_seed<seed>/masks.csv`.
Raw results after training: `p2_cifar100_<distribution>_seed<seed>/results/<checkpoint>_client<id>.csv`.
Final ledger, combined CSV, plot, and conclusion: `paper_assets/05_figures/experiment_p2_20261027/`.
