"""Reset-time initialization helpers for the custom tool_sweep task.

Lane-based placement:
  dustpan at far end of lane, block 11.5 cm before entry, pusher 13 cm before block.
  Dustpan is rotated so its -y opening faces toward the block (= -lane_dir).

diverse_v2 uses orientation-family structured sampling:
  6 lane families (N, S, NE, NW, SE, SW) assigned deterministically per seed,
  with full center-position coverage across the reachable table area.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


PUSHER = "pusher_tool_1"
BLOCK = "red_block_1"
DUSTPAN = "dustpan_1"

# World Z for each object origin when resting on table (z_bottom = 0.900).
PUSHER_Z = 0.960
BLOCK_Z = 0.920
DUSTPAN_Z = 0.900

IDENTITY_QUAT = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

# Lane layout distances (along sweep direction pusher → dustpan):
BLOCK_TO_DUSTPAN_ENTRY = 0.115   # block center → dustpan entry point
PUSHER_TO_BLOCK = 0.130          # pusher handle center → block center
LANE_TOTAL = PUSHER_TO_BLOCK + BLOCK_TO_DUSTPAN_ENTRY  # 0.245 m
LANE_HALF = LANE_TOTAL / 2.0                           # 0.1225 m (midpoint offset from pusher)

# Reachable table area (world frame, relative to robot base).
# X capped at ±0.115 — robot can't grasp pusher reliably beyond this at low z.
TABLE_X = (-0.115, 0.115)
TABLE_Y = (-0.155, 0.240)

# Lane orientation families for diverse_v2.
# Each entry: (name, base_angle_deg, half_jitter_deg, cx_lo, cx_hi, cy_lo, cy_hi)
# cx/cy define the valid range for the lane midpoint (center of pusher→dustpan span).
# Ranges derived from TABLE bounds at base angle; rejection sampling handles jitter edge cases.
#
# N/S families: lane_dir has no x-component at base angle → full x range available.
# Diagonal families: lane_dir has equal x/y components → x range narrows to ±0.028m.
#   0.1225 * cos(45°) ≈ 0.087m, TABLE_X half = 0.115m → slack = 0.028m each side.
_LANE_FAMILY_SPECS: List[Tuple] = [
    # idx  name  base_angle  jitter  cx_lo   cx_hi   cy_lo   cy_hi
    # 0: sweep toward +y (front-to-back, dustpan at top)
    ("N",   90.0, 22.0, -0.090,  0.090, -0.030,  0.115),
    # 1: sweep toward -y (back-to-front, dustpan at bottom)
    ("S",  270.0, 22.0, -0.090,  0.090, -0.030,  0.115),
    # 2: sweep toward +x+y (diagonal NE)
    ("NE",  45.0, 18.0, -0.020,  0.020, -0.058,  0.148),
    # 3: sweep toward -x+y (diagonal NW)
    ("NW", 135.0, 18.0, -0.020,  0.020, -0.058,  0.148),
    # 4: sweep toward -x-y (diagonal SW)
    ("SW", 225.0, 18.0, -0.020,  0.020, -0.058,  0.148),
    # 5: sweep toward +x-y (diagonal SE)
    ("SE", 315.0, 18.0, -0.020,  0.020, -0.058,  0.148),
]

_N_FAMILIES = len(_LANE_FAMILY_SPECS)


_LEVEL_CONFIGS: Dict[str, Any] = {
    "debug_small": {
        "mode": "fixed_lane",
        "lane_base_angle_deg": 90.0,
        "angle_jitter_deg": 8.0,
        "dustpan_xy": (0.020, 0.165),
        "dustpan_xy_jitter": 0.010,
        "pusher_perp_jitter": 0.005,
        "block_perp_jitter": 0.005,
        "block_along_jitter": 0.005,
    },
    "medium": {
        "mode": "semi_random",
        "lane_angle_range_deg": (45.0, 135.0),
        "dustpan_x_range": (-0.050, 0.060),
        "dustpan_y_range": (0.110, 0.200),
        "pusher_perp_jitter": 0.008,
        "block_perp_jitter": 0.008,
        "block_along_jitter": 0.008,
    },
    "diverse_v2": {
        "mode": "lane_family",
        "pusher_perp_jitter": 0.012,
        "block_perp_jitter": 0.012,
        "block_along_jitter": 0.010,
        "max_sample_attempts": 256,
    },
}


def _reachable(xy: np.ndarray) -> bool:
    x, y = float(xy[0]), float(xy[1])
    return (TABLE_X[0] <= x <= TABLE_X[1]) and (TABLE_Y[0] <= y <= TABLE_Y[1])


def _lane_dir(angle_deg: float) -> np.ndarray:
    rad = float(np.radians(angle_deg))
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)


def _dustpan_quat(lane_dir_2d: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) rotating dustpan so its opening faces -lane_dir (toward block)."""
    dx, dy = float(lane_dir_2d[0]), float(lane_dir_2d[1])
    theta = float(np.arctan2(-dx, dy))
    return np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)], dtype=np.float64)


def _bin_id(xy: np.ndarray, nx: int = 3, ny: int = 3) -> int:
    """3×3 grid bin ID for an XY position within TABLE bounds."""
    x_range, y_range = TABLE_X, TABLE_Y
    xi = int(np.clip((xy[0] - x_range[0]) / (x_range[1] - x_range[0]) * nx, 0, nx - 1))
    yi = int(np.clip((xy[1] - y_range[0]) / (y_range[1] - y_range[0]) * ny, 0, ny - 1))
    return int(xi + yi * nx)


def compute_reset_positions(
    seed: int,
    randomization_level: str = "debug_small",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (pusher_xyz, block_xyz, dustpan_xyz, dustpan_quat_wxyz, lane_dir_2d, extra_info)."""
    cfg = _LEVEL_CONFIGS.get(randomization_level, _LEVEL_CONFIGS["debug_small"])
    rng = np.random.default_rng(seed + 208109)
    mode = cfg.get("mode", "fixed_lane")

    if mode == "fixed_lane":
        base_angle = float(cfg["lane_base_angle_deg"])
        jitter = float(cfg.get("angle_jitter_deg", 0.0))
        angle = base_angle + float(rng.uniform(-jitter, jitter))
        d = _lane_dir(angle)
        dx, dy = float(cfg["dustpan_xy"][0]), float(cfg["dustpan_xy"][1])
        j = float(cfg.get("dustpan_xy_jitter", 0.0))
        dustpan_xy = np.array([dx + float(rng.uniform(-j, j)), dy + float(rng.uniform(-j, j))], dtype=np.float64)
        perp = np.array([-d[1], d[0]], dtype=np.float64)
        block_xy = dustpan_xy - BLOCK_TO_DUSTPAN_ENTRY * d
        block_xy = block_xy + float(rng.uniform(-float(cfg.get("block_along_jitter", 0.0)), float(cfg.get("block_along_jitter", 0.0)))) * d
        block_xy = block_xy + float(rng.uniform(-float(cfg.get("block_perp_jitter", 0.0)), float(cfg.get("block_perp_jitter", 0.0)))) * perp
        pusher_xy = block_xy - PUSHER_TO_BLOCK * d
        pusher_xy = pusher_xy + float(rng.uniform(-float(cfg.get("pusher_perp_jitter", 0.0)), float(cfg.get("pusher_perp_jitter", 0.0)))) * perp
        family_idx = 0
        center_xy = (pusher_xy + dustpan_xy) / 2.0

    elif mode in ("semi_random", "continuous"):
        x_range = cfg["dustpan_x_range"]
        y_range = cfg["dustpan_y_range"]
        angle_range = cfg["lane_angle_range_deg"]
        max_attempts = int(cfg.get("max_sample_attempts", 128))
        angle = float(np.mean(angle_range))
        d = _lane_dir(angle)
        dustpan_xy = np.array([float(np.mean(x_range)), float(np.mean(y_range))], dtype=np.float64)
        block_xy = dustpan_xy - BLOCK_TO_DUSTPAN_ENTRY * d
        pusher_xy = block_xy - PUSHER_TO_BLOCK * d
        family_idx = 0
        center_xy = (pusher_xy + dustpan_xy) / 2.0
        for _ in range(max_attempts):
            angle_c = float(rng.uniform(*angle_range))
            d_c = _lane_dir(angle_c)
            perp_c = np.array([-d_c[1], d_c[0]], dtype=np.float64)
            pan_c = np.array([float(rng.uniform(*x_range)), float(rng.uniform(*y_range))], dtype=np.float64)
            blk_c = pan_c - BLOCK_TO_DUSTPAN_ENTRY * d_c
            blk_c += float(rng.uniform(-float(cfg.get("block_along_jitter", 0.0)), float(cfg.get("block_along_jitter", 0.0)))) * d_c
            blk_c += float(rng.uniform(-float(cfg.get("block_perp_jitter", 0.0)), float(cfg.get("block_perp_jitter", 0.0)))) * perp_c
            psh_c = blk_c - PUSHER_TO_BLOCK * d_c
            psh_c += float(rng.uniform(-float(cfg.get("pusher_perp_jitter", 0.0)), float(cfg.get("pusher_perp_jitter", 0.0)))) * perp_c
            min_sep = 0.060
            if not (_reachable(pan_c) and _reachable(blk_c) and _reachable(psh_c)):
                continue
            if np.linalg.norm(pan_c - blk_c) < min_sep or np.linalg.norm(pan_c - psh_c) < min_sep or np.linalg.norm(blk_c - psh_c) < min_sep:
                continue
            angle, d, dustpan_xy, block_xy, pusher_xy = angle_c, d_c, pan_c, blk_c, psh_c
            center_xy = (pusher_xy + dustpan_xy) / 2.0
            break

    elif mode == "lane_family":
        max_attempts = int(cfg.get("max_sample_attempts", 256))
        pjitter = float(cfg.get("pusher_perp_jitter", 0.012))
        bjitter_p = float(cfg.get("block_perp_jitter", 0.012))
        bjitter_a = float(cfg.get("block_along_jitter", 0.010))

        # Deterministic family assignment: seed 0→N, 1→S, 2→NE, 3→NW, 4→SW, 5→SE, 6→N, ...
        preferred_family = int(seed % _N_FAMILIES)

        # Fallback defaults (overwritten on success)
        spec0 = _LANE_FAMILY_SPECS[preferred_family]
        angle = spec0[1]
        d = _lane_dir(angle)
        center_xy = np.array([(spec0[3] + spec0[4]) / 2.0, (spec0[5] + spec0[6]) / 2.0], dtype=np.float64)
        dustpan_xy = center_xy + LANE_HALF * d
        block_xy = center_xy + (LANE_HALF - BLOCK_TO_DUSTPAN_ENTRY) * d
        pusher_xy = center_xy - LANE_HALF * d
        family_idx = preferred_family

        for attempt in range(max_attempts):
            # Try preferred family for first half of budget, then cycle to other families
            if attempt < max_attempts // 2:
                fi = preferred_family
            else:
                fi = (preferred_family + (attempt - max_attempts // 2) + 1) % _N_FAMILIES

            spec = _LANE_FAMILY_SPECS[fi]
            _, base_angle_f, half_jitter_f, cx_lo, cx_hi, cy_lo, cy_hi = spec

            angle_c = base_angle_f + float(rng.uniform(-half_jitter_f, half_jitter_f))
            d_c = _lane_dir(angle_c)
            perp_c = np.array([-d_c[1], d_c[0]], dtype=np.float64)

            cx = float(rng.uniform(cx_lo, cx_hi))
            cy = float(rng.uniform(cy_lo, cy_hi))
            center_c = np.array([cx, cy], dtype=np.float64)

            # Derive positions from center
            dustpan_c = center_c + LANE_HALF * d_c
            block_c = (center_c + (LANE_HALF - BLOCK_TO_DUSTPAN_ENTRY) * d_c
                       + float(rng.uniform(-bjitter_a, bjitter_a)) * d_c
                       + float(rng.uniform(-bjitter_p, bjitter_p)) * perp_c)
            pusher_c = block_c - PUSHER_TO_BLOCK * d_c + float(rng.uniform(-pjitter, pjitter)) * perp_c

            if not (_reachable(dustpan_c) and _reachable(block_c) and _reachable(pusher_c)):
                continue
            # Reject if sweep endpoint (dustpan + 40mm overshoot) is outside workspace.
            # Diagonal families can place the sweep target past TABLE_X bounds, causing
            # the arm to stop early and exit the tolerance check before pushing the block.
            if not _reachable(dustpan_c + 0.040 * d_c):
                continue
            min_sep = 0.060
            if (np.linalg.norm(dustpan_c - block_c) < min_sep
                    or np.linalg.norm(dustpan_c - pusher_c) < min_sep
                    or np.linalg.norm(block_c - pusher_c) < min_sep):
                continue

            family_idx = fi
            angle, d, center_xy = angle_c, d_c, center_c
            dustpan_xy, block_xy, pusher_xy = dustpan_c, block_c, pusher_c
            break

    else:
        raise ValueError(f"Unknown randomization mode: {mode!r}")

    pusher_xyz = np.array([pusher_xy[0], pusher_xy[1], PUSHER_Z], dtype=np.float64)
    block_xyz = np.array([block_xy[0], block_xy[1], BLOCK_Z], dtype=np.float64)
    dustpan_xyz = np.array([dustpan_xy[0], dustpan_xy[1], DUSTPAN_Z], dtype=np.float64)
    pan_quat = _dustpan_quat(d)

    family_name = _LANE_FAMILY_SPECS[family_idx][0] if mode == "lane_family" else "N"
    extra_info = {
        "lane_dir_2d": d.tolist(),
        "lane_angle_deg": float(angle),
        "lane_orientation_id": int(family_idx),
        "sweep_direction": family_name,
        "lane_center_xy": center_xy.tolist(),
        "lane_center_bin_id": _bin_id(center_xy),
        "pusher_bin_id": _bin_id(pusher_xy),
        "block_bin_id": _bin_id(block_xy),
        "dustpan_bin_id": _bin_id(dustpan_xy),
        "pusher_to_block_dist": float(np.linalg.norm(pusher_xy - block_xy)),
        "block_to_dustpan_dist": float(np.linalg.norm(block_xy - dustpan_xy)),
    }
    return pusher_xyz, block_xyz, dustpan_xyz, pan_quat, d, extra_info


def apply_tool_sweep_reset_randomization(
    env: Any,
    obs: Dict[str, Any],
    seed: int,
    settle_steps: int = 20,
    randomization_level: str = "debug_small",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Place pusher, block, and dustpan for one tool_sweep episode."""
    pusher_xyz, block_xyz, dustpan_xyz, pan_quat, lane_dir, extra_info = compute_reset_positions(
        seed, randomization_level
    )

    pusher_obj = env.env.get_object(PUSHER)
    block_obj = env.env.get_object(BLOCK)
    dustpan_obj = env.env.get_object(DUSTPAN)

    env.env.sim.data.set_joint_qpos(
        pusher_obj.joints[0], np.concatenate([pusher_xyz, IDENTITY_QUAT])
    )
    env.env.sim.data.set_joint_qpos(
        block_obj.joints[0], np.concatenate([block_xyz, IDENTITY_QUAT])
    )
    env.env.sim.data.set_joint_qpos(
        dustpan_obj.joints[0], np.concatenate([dustpan_xyz, pan_quat])
    )
    env.env.sim.forward()
    for _ in range(settle_steps):
        env.env.sim.step()
        try:
            env.env._post_process()
        except Exception:
            pass

    reset_info: Dict[str, Any] = {
        **extra_info,
        "pusher_xyz": pusher_xyz.tolist(),
        "block_xyz": block_xyz.tolist(),
        "dustpan_xyz": dustpan_xyz.tolist(),
        "dustpan_quat": pan_quat.tolist(),
        "randomization_level": randomization_level,
        "seed": seed,
    }
    try:
        env.env._update_observables(force=True)
        return env.env._get_observations(force_update=True), reset_info
    except Exception:
        env.env.sim.forward()
        return obs, reset_info
