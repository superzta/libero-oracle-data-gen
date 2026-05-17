"""Diversity QA for button_box initial states in a collected dataset.

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

# Diversity thresholds per level
# "diverse" adds box_x/y_range keys — checked only when box positions are available.
_THRESHOLDS = {
    "debug_small": {"min_x_range": 0.010, "min_y_range": 0.005, "min_std": 0.005, "min_bins": 1},
    "medium":      {"min_x_range": 0.040, "min_y_range": 0.030, "min_std": 0.012, "min_bins": 3},
    "final":       {"min_x_range": 0.060, "min_y_range": 0.040, "min_std": 0.018, "min_bins": 4},
    "diverse":     {
        "min_x_range": 0.120, "min_y_range": 0.060, "min_std": 0.035, "min_bins": 6,
        "min_box_x_range": 0.030, "min_box_y_range": 0.030,
    },
}


def load_cube_positions(dataset_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Return (cube_positions, box_positions, seed_list) from all success HDF5 files."""
    import h5py

    cubes, boxes, seeds = [], [], []
    for path in sorted(dataset_dir.glob("success_*.hdf5")):
        with h5py.File(path, "r") as h5:
            meta = json.loads(h5.attrs.get("metadata", "{}"))
            seeds.append(meta.get("seed"))
            obs = h5.get("observations", {})
            if "blue_cube_1_pos" in obs:
                cubes.append(np.asarray(obs["blue_cube_1_pos"][0], dtype=np.float32).reshape(-1))
            if "open_box_1_pos" in obs:
                boxes.append(np.asarray(obs["open_box_1_pos"][0], dtype=np.float32).reshape(-1))
    return cubes, boxes, seeds


def count_bins(positions_xy: np.ndarray, n_bins_x: int = 2, n_bins_y: int = 2) -> int:
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
        choices=["debug_small", "medium", "final", "diverse"],
        help="Randomization level to check thresholds against. If omitted, reads from run_manifest or summary.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit 1 if diversity thresholds are not met.")
    parser.add_argument("--grid-x", type=int, default=2, help="X bins for occupancy grid (default 2).")
    parser.add_argument("--grid-y", type=int, default=2, help="Y bins for occupancy grid (default 2).")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    cubes, boxes, seeds = load_cube_positions(dataset_dir)

    if not cubes:
        print(json.dumps({"error": "No cube initial positions found", "valid": False}, indent=2))
        raise SystemExit(1)

    # Resolve level from manifest/summary if not provided
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
    bins_occupied = count_bins(cube_xy, args.grid_x, args.grid_y)
    max_possible_bins = args.grid_x * args.grid_y

    box_variation = None
    if boxes:
        box_xy = np.asarray([b[:2] for b in boxes], dtype=np.float32)
        box_variation = {
            "x_range": float(box_xy[:, 0].max() - box_xy[:, 0].min()),
            "y_range": float(box_xy[:, 1].max() - box_xy[:, 1].min()),
            "xy_std": box_xy.std(axis=0).astype(float).tolist(),
        }

    thresholds = _THRESHOLDS.get(level, _THRESHOLDS["debug_small"])
    failures: List[str] = []
    if x_range < thresholds["min_x_range"]:
        failures.append(
            f"cube x_range {x_range:.4f}m < {thresholds['min_x_range']}m required for {level}"
        )
    if y_range < thresholds["min_y_range"]:
        failures.append(
            f"cube y_range {y_range:.4f}m < {thresholds['min_y_range']}m required for {level}"
        )
    max_std = max(xy_std)
    if max_std < thresholds["min_std"]:
        failures.append(
            f"cube max xy_std {max_std:.4f}m < {thresholds['min_std']}m required for {level}"
        )
    if bins_occupied < thresholds["min_bins"]:
        failures.append(
            f"bins_occupied {bins_occupied}/{max_possible_bins} < {thresholds['min_bins']} required for {level}"
        )
    if box_variation is not None:
        if "min_box_x_range" in thresholds and box_variation["x_range"] < thresholds["min_box_x_range"]:
            failures.append(
                f"box x_range {box_variation['x_range']:.4f}m < {thresholds['min_box_x_range']}m required for {level}"
            )
        if "min_box_y_range" in thresholds and box_variation["y_range"] < thresholds["min_box_y_range"]:
            failures.append(
                f"box y_range {box_variation['y_range']:.4f}m < {thresholds['min_box_y_range']}m required for {level}"
            )

    report = {
        "dataset_dir": str(dataset_dir),
        "randomization_level": level,
        "n_episodes": len(cubes),
        "unique_seeds": len(set(seeds)),
        "cube_x_range_m": round(x_range, 5),
        "cube_y_range_m": round(y_range, 5),
        "cube_xy_std_m": [round(v, 5) for v in xy_std],
        "bins_occupied": bins_occupied,
        "max_possible_bins": max_possible_bins,
        "grid": [args.grid_x, args.grid_y],
        "thresholds": thresholds,
        "box_variation": box_variation,
        "failures": failures,
        "passed": not failures,
        "cube_initial_positions_xy": cube_xy.astype(float).round(5).tolist(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
