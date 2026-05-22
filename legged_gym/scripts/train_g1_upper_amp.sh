#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-25000}"
RUN_NAME="${RUN_NAME:-g1_upper_amp_mink}"
# Override with reference motion folder: flat/mink pickles or GMR smplx_to_robot_dataset output (nested subdirs OK).
# export MOTION_REF_DATA_DIR="${MOTION_REF_DATA_DIR:-/home/sldrmk/WorkSpace/AMASS-POST-PROCESS/smpl_retarget/retargeted_motion_data/mink}"
export MOTION_REF_DATA_DIR="${MOTION_REF_DATA_DIR:-/home/sldrmk/WorkSpace/GMR/motion_data/robot_cmu_subset}"
# export MOTION_REF_DATA_DIR="${MOTION_REF_DATA_DIR:-/home/sldrmk/WorkSpace/GMR/motion_data_retargeted}"

echo "[g1_upper_amp] task=g1_upper_amp (AMP discriminator, motion_ref_dof reward off)"
echo "[g1_upper_amp] MOTION_REF_DATA_DIR=${MOTION_REF_DATA_DIR}"
echo "[g1_upper_amp] training_stage=joint_finetune"
echo "[g1_upper_amp] num_envs=${NUM_ENVS}"
echo "[g1_upper_amp] run_name=${RUN_NAME}"
# ----- Resume examples (manual flags; checkpoint format unchanged — torch.load dict with "iter", etc.) -----
# Latest model_* in run (iteration from file + dirname for logs):
#   --resume --load_run=May19_08-00-14_g1_upper_amp_mink
# Specific iter file:
#   --resume --load_run=... --checkpoint=8500
# Full path to .pt :
#   --resume --checkpoint=/path/to/logs/g1_upper_amp/May19_.../model_8500.pt
# Global iteration cap this process (remaining = TARGET − restored iter; aligns AMP curriculum milestones):
#   --train_to_iteration=15000
# Write TensorBoard/checkpoints under a NEW timestamp dir instead of the checkpoint run folder :
#   --resume_fork

python legged_gym/scripts/train.py \
  --task=g1_upper_amp \
  --training_stage=joint_finetune \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --run_name="${RUN_NAME}" \
  --headless
  # --resume --load_run="May20_06-31-49_g1_upper_amp_mink" --checkpoint=15000
