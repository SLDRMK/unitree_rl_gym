"""Load reference-motion pickles for G1 AMP / motion_ref.

Supports:

1. **Mink** pickles from ``convert_fit_motion.py``: top-level dict mapping clip id → payload
   with ``dof`` shaped ``[T, 23]`` in mink *storage* layout (see below).

2. **GMR** pickles from ``GMR/scripts/smplx_to_robot_dataset.py`` (``unitree_g1``): flat dict with
   ``dof_pos`` shaped ``[T, 29]`` (MuJoCo ``qpos[7:]`` for ``g1_mocap_29dof.xml``). Rows are
   reindexed to Isaac ``g1_23dof`` order (drops waist_roll / waist_pitch / wrist pitch–yaw).

Mink storage layout follows AMASS pipeline after ``count_pose_aa`` concat:
first 19 rows of mink dof_pos [:19] plus [22:26] (see ``convert_fit_motion.py``):

- 0-11  legs (left then right), same naming as Isaac
- 12    waist_yaw
- 13-14 waist_roll, waist_pitch (no matching actuators in g1_23dof URDF)
- 15-18 left shoulder pitch/roll/yaw, elbow (left wrist_roll absent in slice)
- 19-22 right shoulder pitch/roll/yaw, elbow (right wrist_roll absent)

Isaac ``g1_23dof`` order (URDF actuator tree): legs, waist_yaw, left arm ×5,
right arm ×5. Wrist_roll columns have no pickle source; callers fill from default pose.

GMR MuJoCo slice → Isaac 23 uses actuator/joint order from ``g1_mocap_29dof.xml``: indices
``0:12`` legs, ``12`` waist_yaw, ``15:20`` left arm through wrist_roll, ``22:27`` right arm
through wrist_roll (skip ``13:14`` waist roll/pitch and ``21,28`` wrist pitch/yaw columns).
"""

from __future__ import annotations

import glob
import os
import pickle
import sys
from types import ModuleType
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


def _ensure_numpy2_pickle_compat() -> None:
    """Unpickle arrays saved with NumPy 2.x when this env runs NumPy 1.x (e.g. Python 3.8).

    Such pickles import ``numpy._core.multiarray`` / ``numpy._core.umath``, which only exist on NumPy 2+.
    Map them to ``numpy.core`` equivalents before ``pickle.load``.
    """
    if hasattr(np, "_core"):
        return
    import numpy.core.multiarray as multiarray
    import numpy.core.umath as umath

    core = ModuleType("numpy._core")
    core.multiarray = multiarray
    core.umath = umath
    sys.modules.setdefault("numpy._core", core)
    sys.modules.setdefault("numpy._core.multiarray", multiarray)
    sys.modules.setdefault("numpy._core.umath", umath)


# Mink: Isaac dof index -> storage column (-1 use default posture for that dof).
# GMR: dof_pos columns (MuJoCo qpos after freejoint, g1 29 DoF) picked into Isaac g1_23dof order.
_GMR_MUJOCO_TO_ISAAC23_COLS: Tuple[int, ...] = (
    *range(12),
    12,
    15,
    16,
    17,
    18,
    19,
    22,
    23,
    24,
    25,
    26,
)

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
    """Return ``dof`` ``[T,23]`` in Isaac order, ``fps`` scalar float, ``dof_layout`` for downstream."""

    if isinstance(blob, dict) and "dof_pos" in blob:
        dof_raw = np.asarray(blob["dof_pos"], dtype=np.float32)
        if dof_raw.ndim != 2:
            raise ValueError(f"GMR dof_pos must be 2D, got {dof_raw.shape}")
        fps_raw = blob.get("fps", 30)
        fps_f = float(np.asarray(fps_raw).reshape(-1)[0])
        last = dof_raw.shape[-1]
        if last == 29:
            cols = np.asarray(_GMR_MUJOCO_TO_ISAAC23_COLS, dtype=np.int64)
            dof = dof_raw[:, cols]
        elif last == 23:
            dof = dof_raw
        else:
            raise ValueError(
                f"GMR dof_pos expected [T,29] or [T,23] for G1, got {dof_raw.shape}; "
                "check robot xml / retargeting pipeline."
            )
        if dof.shape[-1] != 23:
            raise ValueError(f"internal: expected [T,23] after GMR slice, got {dof.shape}")
        return {"dof": dof, "fps": fps_f, "dof_layout": "isaac"}

    if not isinstance(blob, dict) or not blob:
        raise ValueError("motion pickle must be a non-empty dict")

    payload = blob[next(iter(blob.keys()))]
    if not isinstance(payload, dict) or "dof" not in payload:
        raise ValueError(
            "expected mink motion dict {clip_id: {dof, fps}} or GMR dict with dof_pos; "
            f"got keys {list(blob.keys())[:5]}..."
        )
    dof = np.asarray(payload["dof"], dtype=np.float32)
    fps = payload.get("fps", 30)
    if dof.ndim != 2 or dof.shape[-1] != 23:
        raise ValueError(f"mink dof must be [T,23], got {dof.shape}")
    return {"dof": dof, "fps": float(np.asarray(fps).reshape(-1)[0]), "dof_layout": "mink_storage"}


def gather_mink_pkl_paths(data_dir: str, pattern: str = "*.pkl") -> List[str]:
    base = os.path.abspath(os.path.expanduser(data_dir))
    if "**" in pattern:
        glob_pat = os.path.join(base, pattern)
        recursive = True
    else:
        glob_pat = os.path.join(base, "**", pattern)
        recursive = True
    paths = sorted(glob.glob(glob_pat, recursive=recursive))
    if not paths:
        raise FileNotFoundError(
            f"no pickle files matched pattern {pattern!r} (recursive) under {data_dir!r}"
        )
    return paths


class MinkReferenceMotionBank:
    """Pre-resampled looping clips [T, num_dof] at policy dt, on CPU or device.

    Internally stores a padded tensor ``[num_clips, T_max, dof]`` so AMP expert sampling can batch-gather
    without a Python loop over distinct clip ids (which dominated ``learn_time`` for large libraries).
    """

    def __init__(
        self,
        clip_tensors: Sequence[torch.Tensor],
    ):
        if not clip_tensors:
            raise ValueError("motion bank requires at least one clip tensor")
        clips = [c.contiguous().float() for c in clip_tensors]
        lengths_li = [int(c.shape[0]) for c in clips]
        if min(lengths_li) < 2:
            raise ValueError("each motion clip needs at least 2 resampled frames for interpolation")

        self.num_clips = len(clips)
        dof = int(clips[0].shape[1])
        if any(int(c.shape[1]) != dof for c in clips):
            raise ValueError("all clips must share the same dof dimension")
        t_max = max(lengths_li)
        ref_dev = clips[0].device
        pad = torch.zeros(self.num_clips, t_max, dof, dtype=torch.float32, device=ref_dev)
        for i, c in enumerate(clips):
            ti = c.shape[0]
            pad[i, :ti] = c
        self._clips_pad = pad
        self._lengths = torch.tensor(lengths_li, dtype=torch.float32, device=ref_dev)
        self.policy_dt_ref = float("nan")  # optional debug

    def sample_clip_indices(self, n: int, device) -> torch.Tensor:
        """Random clip index per env (uniform)."""
        return torch.randint(0, self.num_clips, (n,), device=device)

    def sample_phase_times(self, clip_ids: torch.Tensor, device) -> torch.Tensor:
        """Uniform start phase in [0, duration) for selected clips."""
        lens = self._lengths.to(device=device)[clip_ids.long()]
        dur_seconds = lens * self.policy_dt_ref
        frac = torch.rand_like(dur_seconds)
        return frac * dur_seconds.clamp(min=self.policy_dt_ref)

    def advance(self, times: torch.Tensor, dt: float) -> None:
        times += dt

    def gather_dof_pos(self, clip_ids: torch.Tensor, times: torch.Tensor, default_pose: torch.Tensor) -> torch.Tensor:
        """clip_ids [N], times [N] seconds since clip start — linear wrap indices (vectorized)."""
        device = times.device
        dtype = default_pose.dtype
        ids = clip_ids.long()
        lens = self._lengths.to(device=device, dtype=torch.long)[ids].clamp(min=2)
        pad = self._clips_pad.to(device=device, dtype=dtype)
        dt = self.policy_dt_ref
        phase = times / dt
        flo = torch.floor(phase)
        i0 = flo.long().remainder(lens)
        i1 = (i0 + 1).remainder(lens)
        w = (phase - flo).unsqueeze(-1)
        c0 = pad[ids, i0]
        c1 = pad[ids, i1]
        return (1.0 - w) * c0 + w * c1

    def gather_dof_vel(self, clip_ids: torch.Tensor, times: torch.Tensor, default_pose: torch.Tensor) -> torch.Tensor:
        """Time-derivative of linearly interpolated joint angles along looping clips (vectorized)."""
        device = times.device
        dtype = default_pose.dtype
        ids = clip_ids.long()
        lens = self._lengths.to(device=device, dtype=torch.long)[ids].clamp(min=2)
        pad = self._clips_pad.to(device=device, dtype=dtype)
        dt = self.policy_dt_ref
        phase = times / dt
        flo = torch.floor(phase)
        i0 = flo.long().remainder(lens)
        i1 = (i0 + 1).remainder(lens)
        dq = pad[ids, i1] - pad[ids, i0]
        return dq / dt

    def gather_amp_dof_features(
        self,
        clip_ids: torch.Tensor,
        center_times: torch.Tensor,
        history_len: int,
        default_dof_row: torch.Tensor,
        scale_dof_pos: float,
        scale_dof_vel: float,
        *,
        history_window_s: Optional[float] = None,
    ) -> torch.Tensor:
        """Stack discriminator-style joint features along `history_len` samples (oldest → newest).

        Each sample is [ (q − q_default)·s_pos ; q_dot·s_vel ]; output [N, history_len · 2·dof].

        Temporal layout:
        - If ``history_window_s`` is None: consecutive policy steps spaced by ``policy_dt_ref``.
        - Else: uniformly spaced times covering ``[center - history_window_s, center]`` (inclusive endpoints).
        """
        if history_len < 1:
            raise ValueError("history_len must be >= 1")
        dt = float(self.policy_dt_ref)
        device = center_times.device
        dtype = default_dof_row.dtype
        default_dof_row = default_dof_row.to(device=device, dtype=dtype)
        N = clip_ids.shape[0]
        frame_counts = self._lengths.to(device=device, dtype=dtype)[clip_ids.long()]
        durations = frame_counts * dt
        durations = durations.clamp(min=dt * 2.0)

        if history_len == 1:
            offs_sec = torch.zeros(1, device=device, dtype=dtype)
        elif history_window_s is not None and float(history_window_s) > 0:
            ws = float(history_window_s)
            denom = float(max(history_len - 1, 1))
            offs_sec = ws * torch.arange(history_len - 1, -1, -1, device=device, dtype=dtype) / denom
        else:
            offs_sec = dt * torch.arange(history_len - 1, -1, -1, device=device, dtype=dtype)

        t_mat = torch.remainder(center_times.unsqueeze(1) - offs_sec.view(1, -1), durations.unsqueeze(1))

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
    """Load clips as tensors on ``device`` when set (e.g. env GPU).

    Keeping clips on GPU avoids per-forward CPU→GPU copies in ``gather_dof_pos`` / ``gather_dof_vel``,
    which becomes costly when many distinct clips appear in one AMP minibatch (likely when the motion
    library is large).
    """
    _ensure_numpy2_pickle_compat()
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
        if item.get("dof_layout", "mink_storage") == "isaac":
            isaac_dof = dof_s
        else:
            isaac_dof = mink_dof_row_to_isaac(dof_s, default_pose_isaac)
        dof_rs = _interp_clip(isaac_dof, fps, policy_dt)
        clips.append(torch.tensor(dof_rs, dtype=torch.float32, device=device))

    bank = MinkReferenceMotionBank(clips)
    bank.policy_dt_ref = policy_dt
    _, t_pad, d_pad = bank._clips_pad.shape
    print(
        f"[MinkReferenceMotionBank] Loaded {len(clips)} clips from {data_dir!r} "
        f"resampled @ policy_dt={policy_dt:.5f}s (padded T_max={t_pad}, dof={d_pad})"
    )
    return bank
