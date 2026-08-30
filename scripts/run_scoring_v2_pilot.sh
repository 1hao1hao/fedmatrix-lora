#!/usr/bin/env bash
set -euo pipefail

# Small pilot only. This script intentionally does not submit Slurm jobs and
# does not touch the full paper experiment manifests.

PILOT_MAX_ROUNDS="${PILOT_MAX_ROUNDS:-20}"
PILOT_TOTAL_BUDGET="${PILOT_TOTAL_BUDGET:-2400}"
PILOT_SEED="${PILOT_SEED:-2024}"

DATASETS_STR="${DATASETS:-cifar100 qnli}"
DISTRIBUTIONS_STR="${DISTRIBUTIONS:-iid dirichlet:0.1}"
METHODS_STR="${METHODS:-Old_Random_FIM Ours_FIM MS Fed-HeLLo}"
read -r -a DATASETS <<< "${DATASETS_STR}"
read -r -a DISTRIBUTIONS <<< "${DISTRIBUTIONS_STR}"
read -r -a METHODS <<< "${METHODS_STR}"

LOCAL_BS="${LOCAL_BS:-16}"
LOCAL_LR="${LOCAL_LR:-2e-5}"
LOCAL_EP="${LOCAL_EP:-1}"
GPU="${GPU:-0}"

PILOT_PREFIX="${PILOT_PREFIX:-scoring_v2_pilot}"
TRACE_ROOT="${TRACE_ROOT:-paper_assets/02_experiments/scoring_v2_pilot_traces}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${TRACE_ROOT}"

sanitize() {
  printf '%s' "$1" | tr ':/ -' '____'
}

for dataset in "${DATASETS[@]}"; do
  for distribution in "${DISTRIBUTIONS[@]}"; do
    for method in "${METHODS[@]}"; do
      safe_dist="$(sanitize "${distribution}")"
      safe_method="$(sanitize "${method}")"
      experiment_id="${PILOT_PREFIX}_${dataset}_${safe_dist}_${safe_method}"
      trace_dir="${TRACE_ROOT}/${experiment_id}"

      common_args=(
        --experiment_id "${experiment_id}"
        --dataset "${dataset}"
        --distribution "${distribution}"
        --total_budget "${PILOT_TOTAL_BUDGET}"
        --max_rounds "${PILOT_MAX_ROUNDS}"
        --local_bs "${LOCAL_BS}"
        --lr "${LOCAL_LR}"
        --local_ep "${LOCAL_EP}"
        --seed "${PILOT_SEED}"
        --gpu "${GPU}"
        --test_method "${method}"
        --trace_dir "${trace_dir}"
      )

      extra_args=()
      if [[ "${method}" == "Ours_FIM" ]]; then
        extra_args=(
          --use_score_ema
          --score_ema_beta 0.8
          --score_utility_mode full
          --adaptive_depth_mix
          --depth_prior_weight 0.3
          --agreement_weight 0.2
          --log_selection_analysis
          --log_update_cosine
        )
      fi

      echo "=== Running ${experiment_id} ==="
      if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'python -u main_drl.py'
        printf ' %q' "${common_args[@]}" "${extra_args[@]}"
        printf '\n'
      else
        python -u main_drl.py "${common_args[@]}" "${extra_args[@]}"
      fi
    done
  done
done
