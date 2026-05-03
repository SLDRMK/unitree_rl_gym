#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
RUN_NAME="${RUN_NAME:-fullbody_isaaclab_randomized}"

echo "[g1_fullbody] task=g1_upper"
echo "[g1_fullbody] training_stage=joint_finetune"
echo "[g1_fullbody] num_envs=${NUM_ENVS}"
echo "[g1_fullbody] max_iterations=${MAX_ITERATIONS}"
echo "[g1_fullbody] run_name=${RUN_NAME}"
echo "[g1_fullbody] randomization=on, observation_noise=on"

python legged_gym/scripts/train.py \
  --task=g1_upper \
  --training_stage=joint_finetune \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --run_name="${RUN_NAME}" \
  --headless
