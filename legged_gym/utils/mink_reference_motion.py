"""Load mink retarget pickles (convert_fit_motion.py) and map DOF rows to Isaac G1 order.

Storage layout follows AMASS pipeline after ``count_pose_aa`` concat:
first 19 rows of mink dof_pos [:19] plus [22:26] (see ``convert_fit_motion.py``):

- 0-11  legs (left then right), same naming as Isaac
- 12    waist_yaw
- 13-14 waist_roll, waist_pitch (no matching actuators in g1_23dof URDF)
- 15-18 left shoulder pitch/roll/yaw, elbow (left wrist_roll absent in slice)
- 19-22 right shoulder pitch/roll/yaw, elbow (right wrist_roll absent)

Isaac ``g1_23dof`` order (URDF actuator tree): legs, waist_yaw, left arm ×5,
right arm ×5. Wrist_roll columns have no pickle source; callers fill from default pose.
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


# Isaac dof index -> storage column (-1 use default posture for that dof)
_G1_STORAGE_MAP: Tuple[int, ...] = (
    0,
    1,
    2,
    3,
    4,
    5,  # left leg
    6,
    7,
    8,
    9,
    10,
    11,  # right leg
    12,  # waist_yaw
    15,
    16,
    17,
    18,  # left shoulder .. elbow (no stor for wrist_roll)
    -1,  # left_wrist_roll
    19,
    20,
    21,
    22,  # right arm to elbow (no stor for wrist_roll)
    -1,  # right_wrist_roll
)


def mink_dof_row_to_isaac(
    dof_storage: np.ndarray, default_pose_isaac: np.ndarray
) -> np.ndarray:
    """Vectorized: dof_storage shape (..., 23), last dim mink-storage order."""
    out = np.broadcast_to(default_pose_isaac, dof_storage.shape).copy()
    for isaac_i, stor_i in enumerate(_G1_STORAGE_MAP):
        if stor_i >= 0:
            out[..., isaac_i] = dof_storage[..., stor_i]
    return out


def _interp_clip(dof: np.ndarray, fps: float, dt_policy: float) -> np.ndarray:
    """Uniformly resample along time axis (motion_interpolation_pkl-style linear interp per channel)."""
    if dof.ndim != 2 or dof.shape[1] != 23:
        raise ValueError(f"expected dof shaped [T, 23], got {dof.shape}")

    fps = float(fps)
    if fps <= 0:
        raise ValueError(f"invalid fps={fps}")

    t_end = dof.shape[0] / fps
    if t_end <= dt_policy:
        return dof.astype(np.float32)

    src_t = np.arange(dof.shape[0], dtype=np.float64) / fps
    dst_t = np.arange(0.0, t_end - 1e-9, dt_policy, dtype=np.float64)
    out = np.stack([np.interp(dst_t, src_t, dof[:, j].astype(np.float64)) for j in range(dof.shape[1])], axis=1)
    return out.astype(np.float32)


def load_mink_motion_dict(blob: dict) -> dict:
    payload = blob[next(iter(blob.keys()))]
    dof = np.asarray(payload["dof"], dtype=np.float32)
    fps = payload.get("fps", 30)
    if dof.ndim != 2 or dof.shape[-1] != 23:
        raise ValueError(f" dof must be [T,23], got {dof.shape}")
    return {"dof": dof, "fps": float(np.asarray(fps).reshape(-1)[0])}


def gather_mink_pkl_paths(data_dir: str, pattern: str = "*.pkl") -> List[str]:
    paths = sorted(glob.glob(os.path.join(os.path.expanduser(data_dir), pattern)))
    if not paths:
        raise FileNotFoundError(f"no pickle files matched {pattern!r} under {data_dir!r}")
    return paths


class MinkReferenceMotionBank:
    """Pre-resampled looping clips [T, num_dof] at policy dt, on CPU or device."""

    def __init__(
        self,
        clip_tensors: Sequence[torch.Tensor],
    ):
        self._clips = [c.contiguous() for c in clip_tensors]
        lengths = [c.shape[0] for c in self._clips]
        if min(lengths) < 2:
            raise ValueError("each motion clip needs at least 2 resampled frames for interpolation")

        self.num_clips = len(self._clips)
        self.policy_dt_ref = float("nan")  # optional debug

    def sample_clip_indices(self, n: int, device) -> torch.Tensor:
        """Random clip index per env (uniform)."""
        return torch.randint(0, self.num_clips, (n,), device=device)

    def sample_phase_times(self, clip_ids: torch.Tensor, device) -> torch.Tensor:
        """Uniform start phase in [0, duration) for selected clips."""
        dur = torch.tensor(
            [self._clips[i].shape[0] for i in clip_ids.cpu().tolist()],
            device=device,
            dtype=torch.float32,
        )
        dur_seconds = dur * self.policy_dt_ref
        frac = torch.rand_like(dur_seconds)
        return frac * dur_seconds.clamp(min=self.policy_dt_ref)

    def advance(self, times: torch.Tensor, dt: float) -> None:
        times += dt

    def gather_dof_pos(self, clip_ids: torch.Tensor, times: torch.Tensor, default_pose: torch.Tensor) -> torch.Tensor:
        """clip_ids [N], times [N] seconds since clip start — linear wrap indices."""
        N = clip_ids.shape[0]
        out = default_pose.unsqueeze(0).expand(N, -1).clone()

        uniq = clip_ids.unique().tolist()
        for cid in uniq:
            mask = clip_ids == cid
            clip = self._clips[cid].to(device=times.device, dtype=out.dtype)
            T = clip.shape[0]
            phase = times[mask] / self.policy_dt_ref
            i0 = torch.floor(phase).long() % T
            i1 = (i0 + 1) % T
            w = (phase - torch.floor(phase)).unsqueeze(1)
            ref_local = (1.0 - w) * clip[i0] + w * clip[i1]
            out[mask] = ref_local

        return out

    def gather_dof_vel(self, clip_ids: torch.Tensor, times: torch.Tensor, default_pose: torch.Tensor) -> torch.Tensor:
        """Time-derivative of linearly interpolated joint angles along looping clips."""
        del default_pose  # symmetry with gather_dof_pos; velocities do not mix in default joints
        N = clip_ids.shape[0]
        out = torch.zeros(N, self._clips[0].shape[1], dtype=times.dtype, device=times.device)
        uniq = clip_ids.unique().tolist()
        for cid in uniq:
            mask = clip_ids == cid
            clip = self._clips[cid].to(device=times.device, dtype=out.dtype)
            T = clip.shape[0]
            dt = self.policy_dt_ref
            phase = times[mask] / dt
            i0 = torch.floor(phase).long() % T
            i1 = (i0 + 1) % T
            dq = clip[i1] - clip[i0]
            out[mask] = dq / dt
        return out

    def gather_amp_dof_features(
        self,
        clip_ids: torch.Tensor,
        center_times: torch.Tensor,
        history_len: int,
        default_dof_row: torch.Tensor,
        scale_dof_pos: float,
        scale_dof_vel: float,
    ) -> torch.Tensor:
        """Stack discriminator-style joint features along `history_len` past steps (oldest → newest).

        Each frame is [ (q − q_default)·s_pos ; q_dot·s_vel ] flattened; output [N, history_len · 46] for G1 (23 dof).
        """
        if history_len < 1:
            raise ValueError("history_len must be >= 1")
        dt = float(self.policy_dt_ref)
        device = center_times.device
        dtype = default_dof_row.dtype
        default_dof_row = default_dof_row.to(device=device, dtype=dtype)
        N = clip_ids.shape[0]
        frame_counts = torch.tensor([float(c.shape[0]) for c in self._clips], device=device, dtype=dtype)
        durations = frame_counts[clip_ids] * dt
        durations = durations.clamp(min=dt * 2.0)

        offs = dt * torch.arange(history_len - 1, -1, -1, device=device, dtype=dtype).view(1, -1)
        t_mat = torch.remainder(center_times.unsqueeze(1) - offs, durations.unsqueeze(1))

        dof_dim = default_dof_row.shape[-1]
        frames = torch.zeros(N, history_len, dof_dim * 2, device=device, dtype=dtype)
        for k in range(history_len):
            tk = t_mat[:, k]
            qp = self.gather_dof_pos(clip_ids, tk, default_dof_row).to(dtype=dtype)
            qv = self.gather_dof_vel(clip_ids, tk, default_dof_row).to(dtype=dtype)
            q_rel = (qp - default_dof_row.unsqueeze(0)) * scale_dof_pos
            frames[:, k] = torch.cat([q_rel, qv * scale_dof_vel], dim=-1)

        return frames.reshape(N, -1)


def build_mink_motion_bank(
    data_dir: str,
    default_pose_isaac: np.ndarray,
    policy_dt: float,
    *,
    glob_pattern: str = "*.pkl",
    clip_limit: Optional[int] = None,
    device=None,
):
    paths = gather_mink_pkl_paths(data_dir, glob_pattern)
    if clip_limit is not None:
        paths = paths[:clip_limit]

    clips: List[torch.Tensor] = []
    for path in paths:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        item = load_mink_motion_dict(blob)
        dof_s = item["dof"]
        fps = item["fps"]
        isaac_dof = mink_dof_row_to_isaac(dof_s, default_pose_isaac)
        dof_rs = _interp_clip(isaac_dof, fps, policy_dt)
        clips.append(torch.tensor(dof_rs, dtype=torch.float32, device=device))

    bank = MinkReferenceMotionBank(clips)
    bank.policy_dt_ref = policy_dt
    print(
        f"[MinkReferenceMotionBank] Loaded {len(clips)} clips from {data_dir!r} "
        f"resampled @ policy_dt={policy_dt:.5f}s"
    )
    return bank
