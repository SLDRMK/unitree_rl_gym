#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

LOWER_BODY_CHECKPOINT="${LOWER_BODY_CHECKPOINT:-${REPO_ROOT}/logs/g1/Apr13_07-17-29_/model_10000.pt}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
RUN_NAME="${RUN_NAME:-stage2_upper_stable}"

python legged_gym/scripts/train.py \
  --task=g1_upper \
  --training_stage=upper_body \
  --lower_body_checkpoint="${LOWER_BODY_CHECKPOINT}" \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --run_name="${RUN_NAME}" \
  --headless
