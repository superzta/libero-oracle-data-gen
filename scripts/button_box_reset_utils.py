"""Reset-time initialization helpers for the custom button_box task."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


CUBE = "blue_cube_1"
BOX = "open_box_1"
BUTTON = "red_button_1"
UPRIGHT_CUBE_QUAT_WXYZ = np.asarray([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float64)
IDENTITY_QUAT_WXYZ = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

# Canonical fixed positions (used when not randomizing a particular object)
BUTTON_FIXED_POS = np.asarray([-0.155, -0.155, 0.900], dtype=np.float64)
BOX_FIXED_POS = np.asarray([0.055, 0.175, 0.900], dtype=np.float64)
CUBE_FIXED_Z = 0.930

# Randomization level configs
# Each level: {"cube_bins_x": [...], "cube_bins_y": [...], "box_jitter": float, "button_jitter": float}
_LEVEL_CONFIGS = {
    "debug_small": {
        # 2×2 grid, x ∈ [-0.030, 0.030], y ∈ [-0.035, -0.005]
        "cube_bins_x": [(-0.030, 0.000), (0.000, 0.030)],
        "cube_bins_y": [(-0.035, -0.022), (-0.022, -0.005)],
        "box_jitter": 0.0,
        "button_jitter": 0.0,
    },
    "medium": {
        # 3×2 grid: x ∈ [-0.045, 0.045] (9 cm range, 3 bins)
        # y ∈ [-0.033, 0.005] (3.8 cm, 2 bins) — empirically safe for grasping
        "cube_bins_x": [(-0.045, -0.012), (-0.012, 0.015), (0.015, 0.045)],
        "cube_bins_y": [(-0.033, -0.015), (-0.015, 0.005)],
        "box_jitter": 0.010,
        "button_jitter": 0.0,
    },
    "final": {
        # 3×3 grid: x ∈ [-0.055, 0.055] (11 cm), y ∈ [-0.035, 0.010] (4.5 cm)
        "cube_bins_x": [(-0.055, -0.020), (-0.020, 0.015), (0.015, 0.055)],
        "cube_bins_y": [(-0.035, -0.015), (-0.015, 0.000), (0.000, 0.010)],
        "box_jitter": 0.020,
        "button_jitter": 0.0,
    },
    "diverse": {
        # 3×3 grid: x ∈ [-0.080, 0.080] (16 cm), y ∈ [-0.060, 0.015] (7.5 cm)
        # Box jitter ±3 cm gives 6 cm coverage in each axis.
        # y range is empirically bounded: negative y is reliable; y > 0.015 risks IK near-singularity.
        "cube_bins_x": [(-0.080, -0.028), (-0.028, 0.028), (0.028, 0.080)],
        "cube_bins_y": [(-0.060, -0.030), (-0.030, -0.005), (-0.005, 0.015)],
        "box_jitter": 0.010,
        "button_jitter": 0.0,
    },
}


def numeric_wait_action(gripper: float = -1.0) -> np.ndarray:
    return np.asarray([0, 0, 0, 0, 0, 0, gripper], dtype=np.float32)


def _sample_bin(rng: np.random.Generator, bins: list, bin_idx: int) -> float:
    lo, hi = bins[bin_idx % len(bins)]
    return float(rng.uniform(lo, hi))


def _cube_box_overlap(cube_xy: np.ndarray, box_xy: np.ndarray,
                      cube_half: float = 0.0275) -> bool:
    """Return True if cube footprint overlaps box interior or any wall geom.

    Box interior clearance prevents BDDL contain_region from firing at reset.
    Wall geom check prevents the cube from landing on top of a wall during settle
    (open_box walls have local centers at ±0.128 from box center, half-thickness
    0.007, half-length 0.121/0.135; they top out at box_z+0.058, well above
    CUBE_FIXED_Z minus cube_half, so any xy overlap means the cube would rest there).
    """
    cx, cy = float(cube_xy[0]), float(cube_xy[1])
    bx, by = float(box_xy[0]), float(box_xy[1])

    # Reject cube centers within the visible tray footprint plus cube margin.
    if abs(cx - bx) < 0.1225 and abs(cy - by) < 0.1225:
        return True

    # Box wall geoms (local pos from box center, half-sizes in x/y):
    #   +x/-x walls: pos=(±0.088, 0), half=(0.007, 0.095)
    #   +y/-y walls: pos=(0, ±0.088), half=(0.081, 0.007)
    walls = [
        ( 0.088,  0.0,  0.007, 0.095),
        (-0.088,  0.0,  0.007, 0.095),
        ( 0.0,    0.088, 0.081, 0.007),
        ( 0.0,   -0.088, 0.081, 0.007),
    ]
    for wx_loc, wy_loc, whx, why in walls:
        if (abs(cx - (bx + wx_loc)) < cube_half + whx and
                abs(cy - (by + wy_loc)) < cube_half + why):
            return True

    return False


def _cube_button_overlap(cube_xy: np.ndarray, button_xy: np.ndarray, min_sep: float = 0.07) -> bool:
    return bool(np.linalg.norm(cube_xy - button_xy) < min_sep)


def compute_reset_positions(
    seed: int,
    randomization_level: str = "debug_small",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cube_xyz, box_xyz, button_xyz) for the given seed and level.

    Cube sampling uses stratified bins so different seeds land in different
    spatial regions. Box and button jitter is uniform around fixed positions.
    Cube positions that collide with box or button are rejected (up to 8 attempts).
    """
    cfg = _LEVEL_CONFIGS.get(randomization_level, _LEVEL_CONFIGS["debug_small"])
    rng = np.random.default_rng(seed + 7919)

    bins_x = cfg["cube_bins_x"]
    bins_y = cfg["cube_bins_y"]
    num_bins = len(bins_x) * len(bins_y)
    bin_idx = seed % num_bins
    xi = bin_idx % len(bins_x)
    yi = bin_idx // len(bins_x)

    box_jitter = cfg["box_jitter"]
    box_xy = BOX_FIXED_POS[:2].copy()
    if box_jitter > 0.0:
        box_xy = box_xy + rng.uniform(-box_jitter, box_jitter, size=2)
    box_xyz = np.asarray([box_xy[0], box_xy[1], BOX_FIXED_POS[2]], dtype=np.float64)

    button_jitter = cfg["button_jitter"]
    button_xy = BUTTON_FIXED_POS[:2].copy()
    if button_jitter > 0.0:
        button_xy = button_xy + rng.uniform(-button_jitter, button_jitter, size=2)
    button_xyz = np.asarray([button_xy[0], button_xy[1], BUTTON_FIXED_POS[2]], dtype=np.float64)

    # Sample cube with rejection for feasibility
    cube_x = 0.0
    cube_y = -0.03
    for attempt in range(64):
        if attempt == 0:
            cube_x = _sample_bin(rng, bins_x, xi)
            cube_y = _sample_bin(rng, bins_y, yi)
        else:
            # fallback: uniform from the full range
            x_lo = bins_x[0][0]
            x_hi = bins_x[-1][1]
            y_lo = bins_y[0][0]
            y_hi = bins_y[-1][1]
            cube_x = float(rng.uniform(x_lo, x_hi))
            cube_y = float(rng.uniform(y_lo, y_hi))
        cube_xy = np.asarray([cube_x, cube_y], dtype=np.float64)
        if not _cube_box_overlap(cube_xy, box_xy) and not _cube_button_overlap(cube_xy, button_xy):
            break
    else:
        cube_x = float(rng.uniform(-0.055, 0.055))
        cube_y = float(rng.uniform(-0.060, -0.010))

    cube_xyz = np.asarray([cube_x, cube_y, CUBE_FIXED_Z], dtype=np.float64)
    return cube_xyz, box_xyz, button_xyz


def apply_button_box_reset_randomization(
    env: Any,
    obs: Dict[str, Any],
    seed: int,
    settle_steps: int = 20,
    randomization_level: str = "debug_small",
) -> Dict[str, Any]:
    """Apply feasible cube/box/button randomization before rollout begins.

    Direct simulator writes here are reset-time initialization, not rollout
    oracle actions. Rollout write guards are installed after this helper.
    """
    cube_xyz, box_xyz, button_xyz = compute_reset_positions(seed, randomization_level)

    cube = env.env.get_object(CUBE)
    box = env.env.get_object(BOX)

    try:
        button = env.env.get_object(BUTTON)
        body_id = env.env.sim.model.body_name2id(button.root_body)
        env.env.sim.model.body_pos[body_id] = button_xyz
        env.env.sim.model.body_quat[body_id] = IDENTITY_QUAT_WXYZ
    except Exception:
        pass

    env.env.sim.data.set_joint_qpos(
        box.joints[0],
        np.concatenate([box_xyz, IDENTITY_QUAT_WXYZ]),
    )
    env.env.sim.data.set_joint_qpos(
        cube.joints[0],
        np.concatenate([cube_xyz, UPRIGHT_CUBE_QUAT_WXYZ]),
    )
    env.env.sim.forward()
    for _ in range(settle_steps):
        env.env.sim.step()
        try:
            env.env._post_process()
        except Exception:
            pass
    try:
        env.env._update_observables(force=True)
        return env.env._get_observations(force_update=True)
    except Exception:
        env.env.sim.forward()
        return obs
