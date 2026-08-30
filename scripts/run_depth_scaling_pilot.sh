#!/usr/bin/env bash
set -euo pipefail

# Small depth-scaling ablation pilot. This does not submit Slurm jobs and does
# not modify the full paper experiment manifests.

PILOT_MAX_ROUNDS="${PILOT_MAX_ROUNDS:-20}"
PILOT_TOTAL_BUDGET="${PILOT_TOTAL_BUDGET:-2400}"
PILOT_SEED="${PILOT_SEED:-2024}"

DATASETS_STR="${DATASETS:-cifar100 mnli}"
DISTRIBUTIONS_BY_DATASET="${DISTRIBUTIONS_BY_DATASET:-cifar100=dirichlet:0.1 mnli=dirichlet:1.0}"
METHODS_STR="${METHODS:-Pure_FIM Old_FIM_Depth Lite_FIM_WeakDepth MS}"
read -r -a DATASETS <<< "${DATASETS_STR}"
read -r -a METHODS <<< "${METHODS_STR}"

LOCAL_BS="${LOCAL_BS:-16}"
LOCAL_LR="${LOCAL_LR:-2e-5}"
LOCAL_EP="${LOCAL_EP:-1}"
GPU="${GPU:-0}"

PILOT_PREFIX="${PILOT_PREFIX:-depth_scaling_pilot}"
TRACE_ROOT="${TRACE_ROOT:-paper_assets/02_experiments/depth_scaling_pilot_traces}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${TRACE_ROOT}"

sanitize() {
  printf '%s' "$1" | tr ':/ -' '____'
}

distribution_for_dataset() {
  local dataset="$1"
  local mapping
  for mapping in ${DISTRIBUTIONS_BY_DATASET}; do
    if [[ "${mapping%%=*}" == "${dataset}" ]]; then
      printf '%s' "${mapping#*=}"
      return 0
    fi
  done
  echo "Missing distribution mapping for ${dataset}" >&2
  return 1
}

for dataset in "${DATASETS[@]}"; do
  distribution="$(distribution_for_dataset "${dataset}")"
  for method in "${METHODS[@]}"; do
    safe_dist="$(sanitize "${distribution}")"
    safe_method="$(sanitize "${method}")"
    experiment_id="${PILOT_PREFIX}_${dataset}_${safe_dist}_${safe_method}"
    trace_dir="${TRACE_ROOT}/${experiment_id}"

    args=(
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
      --log_selection_analysis
    )

    echo "=== Running ${experiment_id} ==="
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf 'python -u main_drl.py'
      printf ' %q' "${args[@]}"
      printf '\n'
    else
      python -u main_drl.py "${args[@]}"
    fi
  done
done
