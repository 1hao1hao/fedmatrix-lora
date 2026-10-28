# P2 locked rewrite evidence map

- Active English source: `paper_assets/04_manuscript/main.tex`; inspected pre-patch Git blob: `e86de20b821c9316c7c5448445c95d65cf51fb92`.
- Backup: `paper_assets/04_manuscript/.rewrite_backups/main_before_p2_e86de20b821c.tex`.
- Full source diff, including the float-only bibliography transition: `paper_assets/04_manuscript/generated/p2_rewrite.diff`.
- Compiled output: `paper_assets/04_manuscript/main.pdf` (13 pages after replacing the bibliography `\clearpage` with `\FloatBarrier`). The Chinese source and PDF were not updated.
- P2 source: `paper_assets/05_figures/experiment_p2_20261027/all_results.csv`, `paired_summary.csv`, and `paired_seed_differences.csv`; copied figure: `paper_assets/04_manuscript/figures/fig1_p2_granularity.pdf`; derived complete table: `generated/fm_p2_complete_table.tex`.
- A source: 90 original 12-configuration CSV pools under `paper_assets/02_experiments/matrix_score_20261025/full/`; corrected derived records: `generated/a_corrected_configuration_records.csv`; correlation table: `generated/fm_a_verified_table.tex`; source-control table: `generated/fm_p_corrected_table.tex`. All 90 first-configuration outcomes were replaced by matching P2 paired-RNG outcomes. Original pools were not modified.
- B source: `paper_assets/05_figures/experiment_20261025/P_A_B_review/B_round_curves.csv` and `B_three_seed_summary.csv`; derived full-trajectory figure: `paper_assets/04_manuscript/figures/fig6_ablation_budget.pdf`. All four variants, three seeds, 80 rounds, and final budget 7200 were verified.
- Preparation tool issue: the locked package's B plot required an identical budget grid across *different methods*. Layer-Level actually differs by up to 27 matrix-equivalent units at intermediate rounds. A temporary `/tmp` copy moved `grid=None` inside the method loop, requiring identical grids only across seeds of the same method. No interpolation, raw-data change, or package-file change was made.
- Compilation check: no undefined references, LaTeX errors, or overfull boxes in the final log. Rendered pages 2, 5, 9, 10, 11, and 12 were inspected. The Appendix B table floats to the top of page 11, before its heading; its contents and reference are intact.
- Unresolved: the English draft is 13 pages and is not within the former INFOCOM 9+1 page limit. The Chinese reading version is intentionally not synchronized.
