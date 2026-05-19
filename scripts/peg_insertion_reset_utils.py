"""Reset-time initialization helpers for the custom peg_insertion task."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


PEG = "green_peg_1"
BLOCK = "wooden_hole_block_1"
PEG_QUAT_WXYZ = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
BLOCK_QUAT_WXYZ = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
PEG_FIXED_Z = 0.955
BLOCK_FIXED_Z = 0.900

_LEVEL_CONFIGS = {
    "debug_small": {
        "peg_x_range": (-0.035, 0.025),
        "peg_y_range": (-0.075, -0.030),
        "block_x_range": (0.040, 0.095),
        "block_y_range": (0.115, 0.170),
        "min_sep": 0.145,
    },
    "medium": {
        "peg_x_range": (-0.090, 0.075),
        "peg_y_range": (-0.120, 0.035),
        "block_x_range": (-0.055, 0.105),
        "block_y_range": (0.060, 0.205),
        "min_sep": 0.150,
    },
    "final": {
        "peg_x_range": (-0.120, 0.110),
        "peg_y_range": (-0.160, 0.070),
        "block_x_range": (-0.095, 0.130),
        "block_y_range": (0.020, 0.230),
        "min_sep": 0.155,
    },
    "diverse": {
        "peg_x_range": (-0.140, 0.140),
        "peg_y_range": (-0.185, 0.210),
        "block_x_range": (-0.140, 0.140),
        "block_y_range": (-0.185, 0.210),
        "min_sep": 0.165,
    },
    "diverse_v2": {
        "mode": "continuous",
        # Shared visible/reachable table area for every object. Far-right x
        # samples beyond ~0.13m are physically unreliable for grasping the peg.
        "table_x_range": (-0.120, 0.125),
        "table_y_range": (-0.125, 0.250),
        "min_sep": 0.130,
        "max_sep": 0.180,
    },
}


def _sample_xy(rng: np.random.Generator, x_range, y_range) -> np.ndarray:
    return np.asarray(
        [float(rng.uniform(*x_range)), float(rng.uniform(*y_range))],
        dtype=np.float64,
    )


def _bin_id(xy: np.ndarray, x_range, y_range, nx: int = 3, ny: int = 3) -> int:
    xi = int(np.clip((xy[0] - x_range[0]) / (x_range[1] - x_range[0]) * nx, 0, nx - 1))
    yi = int(np.clip((xy[1] - y_range[0]) / (y_range[1] - y_range[0]) * ny, 0, ny - 1))
    return int(xi + yi * nx)


def _shared_reachable_xy(xy: np.ndarray) -> bool:
    """Conservative shared workspace filter used for every object.

    The far back-left corner causes reliable grasping but unreliable carried
    insertion because the arm cannot maintain a stable grasp while descending
    there. This is a reset-time infeasible-layout rejection, not a per-object
    placement zone.
    """
    x, y = float(xy[0]), float(xy[1])
    return not (x < -0.080 and y < -0.080)


def compute_reset_positions(seed: int, randomization_level: str = "debug_small") -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return (peg_xyz, block_xyz, peg_bin_id, block_bin_id)."""
    cfg = _LEVEL_CONFIGS.get(randomization_level, _LEVEL_CONFIGS["debug_small"])
    rng = np.random.default_rng(seed + 104729)

    if cfg.get("mode") == "continuous":
        x_range = cfg["table_x_range"]
        y_range = cfg["table_y_range"]
        block_xy = _sample_xy(rng, x_range, y_range)
        for _ in range(256):
            if _shared_reachable_xy(block_xy):
                break
            block_xy = _sample_xy(rng, x_range, y_range)
        peg_xy = np.asarray([0.0, -0.06], dtype=np.float64)
        for _ in range(256):
            candidate = _sample_xy(rng, x_range, y_range)
            distance = float(np.linalg.norm(candidate - block_xy))
            if (
                _shared_reachable_xy(candidate)
                and distance >= float(cfg["min_sep"])
                and distance <= float(cfg.get("max_sep", 999.0))
            ):
                peg_xy = candidate
                break
        peg_bin_id = _bin_id(peg_xy, x_range, y_range, 3, 3)
        block_bin_id = _bin_id(block_xy, x_range, y_range, 3, 3)
    else:
        peg_xy = _sample_xy(rng, cfg["peg_x_range"], cfg["peg_y_range"])
        block_xy = _sample_xy(rng, cfg["block_x_range"], cfg["block_y_range"])
        for _ in range(128):
            if np.linalg.norm(peg_xy - block_xy) >= float(cfg["min_sep"]):
                break
            peg_xy = _sample_xy(rng, cfg["peg_x_range"], cfg["peg_y_range"])
            block_xy = _sample_xy(rng, cfg["block_x_range"], cfg["block_y_range"])
        peg_bin_id = _bin_id(peg_xy, cfg["peg_x_range"], cfg["peg_y_range"], 3, 2)
        block_bin_id = _bin_id(block_xy, cfg["block_x_range"], cfg["block_y_range"], 3, 2)

    peg_xyz = np.asarray([peg_xy[0], peg_xy[1], PEG_FIXED_Z], dtype=np.float64)
    block_xyz = np.asarray([block_xy[0], block_xy[1], BLOCK_FIXED_Z], dtype=np.float64)
    return peg_xyz, block_xyz, peg_bin_id, block_bin_id


def apply_peg_insertion_reset_randomization(
    env: Any,
    obs: Dict[str, Any],
    seed: int,
    settle_steps: int = 20,
    randomization_level: str = "debug_small",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Place peg and socket block before rollout begins."""
    peg_xyz, block_xyz, peg_bin_id, block_bin_id = compute_reset_positions(seed, randomization_level)

    peg = env.env.get_object(PEG)
    block = env.env.get_object(BLOCK)
    env.env.sim.data.set_joint_qpos(peg.joints[0], np.concatenate([peg_xyz, PEG_QUAT_WXYZ]))
    env.env.sim.data.set_joint_qpos(block.joints[0], np.concatenate([block_xyz, BLOCK_QUAT_WXYZ]))
    env.env.sim.forward()
    for _ in range(settle_steps):
        env.env.sim.step()
        try:
            env.env._post_process()
        except Exception:
            pass

    reset_info: Dict[str, Any] = {
        "peg_bin_id": int(peg_bin_id),
        "block_bin_id": int(block_bin_id),
        "peg_xyz": peg_xyz.astype(float).tolist(),
        "block_xyz": block_xyz.astype(float).tolist(),
        "hole_xyz": (block_xyz + np.asarray([0.0, 0.0, 0.021], dtype=np.float64)).astype(float).tolist(),
    }
    try:
        env.env._update_observables(force=True)
        return env.env._get_observations(force_update=True), reset_info
    except Exception:
        env.env.sim.forward()
        return obs, reset_info
