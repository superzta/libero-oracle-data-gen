"""Diversity QA for custom-task initial states in a collected dataset.

Checks that cube (and optionally box/button) starting positions vary
meaningfully across successful episodes. Reports bin occupancy, range,
and std, and can fail-fast with --strict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Diversity thresholds per level.
# diverse_v2 checks named bin IDs (cube_named_bins, box_named_bins) in addition to
# positional range/std metrics.
_THRESHOLDS = {
    "debug_small": {"min_x_range": 0.010, "min_y_range": 0.005, "min_std": 0.005, "min_bins": 1},
    "medium":      {"min_x_range": 0.040, "min_y_range": 0.030, "min_std": 0.012, "min_bins": 3},
    "final":       {"min_x_range": 0.060, "min_y_range": 0.040, "min_std": 0.018, "min_bins": 4},
    "diverse":     {
        "min_x_range": 0.120, "min_y_range": 0.060, "min_std": 0.035, "min_bins": 6,
        "min_box_x_range": 0.030, "min_box_y_range": 0.030,
    },
    "diverse_v2":  {
        "min_x_range": 0.120,
        "min_y_range": 0.080,
        "min_std": 0.035,
        "min_bins": 6,
        "min_bins_5demo": 5,
        "min_cube_named_bins": 6,
        "min_box_named_bins": 3,
        "min_cube_named_bins_5demo": 3,
        "min_box_named_bins_5demo": 2,
        "min_unique_seeds": 5,
        "min_box_x_range": 0.030,
        "min_box_y_range": 0.020,
        # tool_sweep-specific
        "tool_sweep_min_lane_orientations": 4,       # ≥4 distinct families (100-demo)
        "tool_sweep_min_lane_orientations_5demo": 3, # ≥3 for 5-demo gate
        "tool_sweep_min_lane_center_bins": 6,        # ≥6 of 9 lane-center bins (100-demo)
        "tool_sweep_min_lane_center_bins_5demo": 4,  # ≥4 for 5-demo gate
        "tool_sweep_min_block_x_range": 0.080,
        "tool_sweep_min_block_y_range": 0.080,
        "tool_sweep_min_pusher_x_range": 0.060,
        "tool_sweep_min_dustpan_x_range": 0.060,
    },
}

_BIN_NAMES = {
    "cube": ["left-front", "center-front", "right-front", "left-back", "center-back", "right-back"],
    "box":  ["box-left", "box-center", "box-right", "box-back-center"],
}


def load_episode_data(dataset_dir: Path) -> Tuple[List[dict], List[np.ndarray], List[np.ndarray], str, str, str]:
    """Return metadata and primary/target object initial positions."""
    import h5py

    metas, primaries, targets = [], [], []
    primary_key = "blue_cube_1_pos"
    target_key = "open_box_1_pos"
    primary_name = "cube"
    target_name = "box"
    for path in sorted(dataset_dir.glob("success_*.hdf5")):
        with h5py.File(path, "r") as h5:
            meta = json.loads(h5.attrs.get("metadata", "{}"))
            metas.append(meta)
            obs = h5.get("observations", {})
            if "red_block_1_pos" in obs:
                primary_key = "red_block_1_pos"
                target_key = "dustpan_1_pos"
                primary_name = "block"
                target_name = "dustpan"
            elif "green_peg_1_pos" in obs:
                primary_key = "green_peg_1_pos"
                target_key = "wooden_hole_block_1_pos"
                primary_name = "peg"
                target_name = "block"
            if primary_key in obs:
                primaries.append(np.asarray(obs[primary_key][0], dtype=np.float32).reshape(-1))
            if target_key in obs:
                targets.append(np.asarray(obs[target_key][0], dtype=np.float32).reshape(-1))
    return metas, primaries, targets, primary_name, target_name, primary_key


def count_grid_bins(positions_xy: np.ndarray, n_bins_x: int = 3, n_bins_y: int = 2) -> int:
    if len(positions_xy) < 2:
        return len(positions_xy)
    x_lo, x_hi = positions_xy[:, 0].min(), positions_xy[:, 0].max()
    y_lo, y_hi = positions_xy[:, 1].min(), positions_xy[:, 1].max()
    eps = 1e-6
    x_edges = np.linspace(x_lo - eps, x_hi + eps, n_bins_x + 1)
    y_edges = np.linspace(y_lo - eps, y_hi + eps, n_bins_y + 1)
    occupied = set()
    for xy in positions_xy:
        xi = int(np.searchsorted(x_edges[1:], xy[0]))
        yi = int(np.searchsorted(y_edges[1:], xy[1]))
        occupied.add((xi, yi))
    return len(occupied)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diversity QA: check initial state variation in a collected dataset."
    )
    parser.add_argument("dataset_dir")
    parser.add_argument(
        "--level",
        default=None,
        choices=list(_THRESHOLDS.keys()),
        help="Randomization level. If omitted, reads from run_manifest or summary.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit 1 if thresholds not met.")
    parser.add_argument("--grid-x", type=int, default=3, help="X bins for positional grid (default 3).")
    parser.add_argument("--grid-y", type=int, default=2, help="Y bins for positional grid (default 2).")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    metas, cubes, boxes, primary_name, target_name, primary_key = load_episode_data(dataset_dir)

    if not cubes:
        print(json.dumps({"error": f"No {primary_name} initial positions found", "valid": False}, indent=2))
        raise SystemExit(1)

    # Resolve level
    level = args.level
    if level is None:
        for fname in ("run_manifest.json", "summary.json"):
            fp = dataset_dir / fname
            if fp.exists():
                d = json.loads(fp.read_text())
                level = d.get("randomization_level")
                if level:
                    break
    if level is None:
        level = "debug_small"

    cube_xy = np.asarray([c[:2] for c in cubes], dtype=np.float32)
    x_range = float(cube_xy[:, 0].max() - cube_xy[:, 0].min())
    y_range = float(cube_xy[:, 1].max() - cube_xy[:, 1].min())
    xy_std = cube_xy.std(axis=0).astype(float).tolist()
    bins_occupied = count_grid_bins(cube_xy, args.grid_x, args.grid_y)
    max_possible_bins = args.grid_x * args.grid_y
    seeds = [m.get("seed") for m in metas]
    n_episodes = len(cubes)

    box_variation = None
    if boxes:
        box_xy = np.asarray([b[:2] for b in boxes], dtype=np.float32)
        box_variation = {
            "x_range": float(box_xy[:, 0].max() - box_xy[:, 0].min()),
            "y_range": float(box_xy[:, 1].max() - box_xy[:, 1].min()),
            "xy_std": box_xy.std(axis=0).astype(float).tolist(),
        }

    # Named bin occupancy (diverse_v2 only; non-tool_sweep tasks)
    cube_bin_ids = [m.get("peg_bin_id", m.get("cube_bin_id", -1)) for m in metas]
    box_bin_ids  = [m.get("block_bin_id", m.get("box_bin_id", -1)) for m in metas]
    cube_bins_occupied = sorted(set(b for b in cube_bin_ids if b >= 0))
    box_bins_occupied  = sorted(set(b for b in box_bin_ids  if b >= 0))
    cube_bin_counts = {_BIN_NAMES["cube"][b]: cube_bin_ids.count(b)
                       for b in cube_bins_occupied if b < len(_BIN_NAMES["cube"])}
    box_bin_counts  = {_BIN_NAMES["box"][b]:  box_bin_ids.count(b)
                       for b in box_bins_occupied  if b < len(_BIN_NAMES["box"])}

    # tool_sweep lane diversity (from reset_info stored in episode metadata)
    is_tool_sweep = primary_name == "block" and target_name == "dustpan"
    lane_orientation_ids = [m.get("lane_orientation_id", -1) for m in metas]
    lane_center_bin_ids  = [m.get("lane_center_bin_id",  -1) for m in metas]
    sweep_directions     = [m.get("sweep_direction", "") for m in metas]
    lane_orientations_occupied = sorted(set(v for v in lane_orientation_ids if v >= 0))
    lane_center_bins_occupied  = sorted(set(v for v in lane_center_bin_ids  if v >= 0))
    lane_direction_counts = {}
    for d in sweep_directions:
        if d:
            lane_direction_counts[d] = lane_direction_counts.get(d, 0) + 1

    pusher_variation = None
    if is_tool_sweep:
        pusher_key = "pusher_tool_1_pos"
        pushers = []
        import h5py
        for path in sorted(dataset_dir.glob("success_*.hdf5")):
            with h5py.File(path, "r") as h5:
                obs = h5.get("observations", {})
                if pusher_key in obs:
                    pushers.append(np.asarray(obs[pusher_key][0], dtype=np.float32).reshape(-1))
        if pushers:
            p_xy = np.asarray([p[:2] for p in pushers], dtype=np.float32)
            pusher_variation = {
                "x_range": float(p_xy[:, 0].max() - p_xy[:, 0].min()),
                "y_range": float(p_xy[:, 1].max() - p_xy[:, 1].min()),
                "xy_std": p_xy.std(axis=0).astype(float).tolist(),
            }

    thresholds = _THRESHOLDS.get(level, _THRESHOLDS["debug_small"])
    failures: List[str] = []
    is_5demo = n_episodes <= 10

    # Primary object position range / std
    if is_tool_sweep:
        # For tool_sweep, the block ranges widely due to lane diversity; use task-specific thresholds
        min_bx = thresholds.get("tool_sweep_min_block_x_range", thresholds["min_x_range"])
        min_by = thresholds.get("tool_sweep_min_block_y_range", thresholds["min_y_range"])
        if x_range < min_bx and not is_5demo:
            failures.append(f"block x_range {x_range:.4f}m < {min_bx}m")
        if y_range < min_by and not is_5demo:
            failures.append(f"block y_range {y_range:.4f}m < {min_by}m")
    else:
        if x_range < thresholds["min_x_range"]:
            failures.append(f"cube x_range {x_range:.4f}m < {thresholds['min_x_range']}m")
        if y_range < thresholds["min_y_range"]:
            failures.append(f"cube y_range {y_range:.4f}m < {thresholds['min_y_range']}m")
        max_std = max(xy_std)
        if max_std < thresholds["min_std"]:
            failures.append(f"cube max xy_std {max_std:.4f}m < {thresholds['min_std']}m")

    is_5demo_bins = is_5demo and "min_bins_5demo" in thresholds
    min_bins_req = thresholds["min_bins_5demo"] if is_5demo_bins else thresholds["min_bins"]
    if bins_occupied < min_bins_req and not is_tool_sweep:
        failures.append(f"positional bins {bins_occupied}/{max_possible_bins} < {min_bins_req}")
    if box_variation is not None and not is_tool_sweep:
        if "min_box_x_range" in thresholds and box_variation["x_range"] < thresholds["min_box_x_range"]:
            failures.append(f"box x_range {box_variation['x_range']:.4f}m < {thresholds['min_box_x_range']}m")
        if "min_box_y_range" in thresholds and box_variation["y_range"] < thresholds["min_box_y_range"]:
            failures.append(f"box y_range {box_variation['y_range']:.4f}m < {thresholds['min_box_y_range']}m")

    # diverse_v2 named-bin checks (non-tool_sweep)
    if level == "diverse_v2" and any(b >= 0 for b in cube_bin_ids) and not is_tool_sweep:
        req_cube = thresholds["min_cube_named_bins_5demo"] if is_5demo else thresholds["min_cube_named_bins"]
        req_box  = thresholds["min_box_named_bins_5demo"]  if is_5demo else thresholds["min_box_named_bins"]
        if len(cube_bins_occupied) < req_cube:
            failures.append(f"cube named bins {len(cube_bins_occupied)} < {req_cube} required ({'5-demo gate' if is_5demo else '100-demo'})")
        if len(box_bins_occupied) < req_box:
            failures.append(f"box named bins {len(box_bins_occupied)} < {req_box} required")

    # diverse_v2 tool_sweep lane-diversity checks
    if level == "diverse_v2" and is_tool_sweep and lane_orientations_occupied:
        req_orient = thresholds.get("tool_sweep_min_lane_orientations_5demo" if is_5demo else "tool_sweep_min_lane_orientations", 4)
        req_bins   = thresholds.get("tool_sweep_min_lane_center_bins_5demo"  if is_5demo else "tool_sweep_min_lane_center_bins",   6)
        if len(lane_orientations_occupied) < req_orient:
            failures.append(
                f"lane orientations {len(lane_orientations_occupied)} < {req_orient} required "
                f"({'5-demo gate' if is_5demo else '100-demo'}); got: {lane_direction_counts}"
            )
        if len(lane_center_bins_occupied) < req_bins:
            failures.append(
                f"lane center bins {len(lane_center_bins_occupied)}/9 < {req_bins} required "
                f"({'5-demo gate' if is_5demo else '100-demo'})"
            )
        if pusher_variation:
            min_px = thresholds.get("tool_sweep_min_pusher_x_range", 0.060)
            if pusher_variation["x_range"] < min_px and not is_5demo:
                failures.append(f"pusher x_range {pusher_variation['x_range']:.4f}m < {min_px}m")

    report = {
        "dataset_dir": str(dataset_dir),
        "randomization_level": level,
        "n_episodes": n_episodes,
        "unique_seeds": len(set(seeds)),
        "primary_object": primary_name,
        "target_object": target_name,
        "primary_observation_key": primary_key,
        f"{primary_name}_x_range_m": round(x_range, 5),
        f"{primary_name}_y_range_m": round(y_range, 5),
        f"{primary_name}_xy_std_m": [round(v, 5) for v in xy_std],
        "positional_bins_occupied": bins_occupied,
        "max_possible_bins": max_possible_bins,
        "grid": [args.grid_x, args.grid_y],
        "cube_named_bins_occupied": cube_bins_occupied,
        "cube_bin_counts": cube_bin_counts,
        f"{target_name}_named_bins_occupied": box_bins_occupied,
        f"{target_name}_bin_counts": box_bin_counts,
        "lane_orientation_ids_occupied": lane_orientations_occupied,
        "lane_direction_counts": lane_direction_counts,
        "lane_center_bins_occupied": lane_center_bins_occupied,
        "pusher_variation": pusher_variation,
        "thresholds": thresholds,
        f"{target_name}_variation": box_variation,
        "failures": failures,
        "passed": not failures,
        f"{primary_name}_initial_positions_xy": cube_xy.astype(float).round(5).tolist(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
