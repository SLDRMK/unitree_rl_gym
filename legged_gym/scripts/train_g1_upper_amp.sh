#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
RUN_NAME="${RUN_NAME:-g1_upper_amp_mink}"
# Override with your AMASS/pipeline folder (mink pickles from convert_fit_motion.py).
export MOTION_REF_DATA_DIR="${MOTION_REF_DATA_DIR:-/home/sldrmk/WorkSpace/AMASS-POST-PROCESS/smpl_retarget/retargeted_motion_data/mink}"

echo "[g1_upper_amp] task=g1_upper_amp (AMP discriminator, motion_ref_dof reward off)"
echo "[g1_upper_amp] MOTION_REF_DATA_DIR=${MOTION_REF_DATA_DIR}"
echo "[g1_upper_amp] training_stage=joint_finetune"
echo "[g1_upper_amp] num_envs=${NUM_ENVS}"
echo "[g1_upper_amp] run_name=${RUN_NAME}"

python legged_gym/scripts/train.py \
  --task=g1_upper_amp \
  --training_stage=joint_finetune \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --run_name="${RUN_NAME}" \
  --headless
