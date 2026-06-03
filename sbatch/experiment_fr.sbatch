#!/bin/bash
#SBATCH --job-name=lh2-fr
#SBATCH --partition=hpc-mid
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
export HF_HOME=/mnt/home/brendan/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/home/brendan/loophole-v2
.venv/bin/python scripts/experiment_fr.py "$1" grpo "$2"
