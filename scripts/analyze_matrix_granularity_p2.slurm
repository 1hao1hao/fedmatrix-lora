#!/bin/bash
#SBATCH -J p2_analyze
#SBATCH -p cpuXeon6458
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o logs/p2_analyze_%j.out
#SBATCH -e logs/p2_analyze_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
source activate bjh_lora || conda activate bjh_lora
set -u
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_p2_${SLURM_JOB_ID}"
python -u scripts/analyze_matrix_granularity_p2.py
