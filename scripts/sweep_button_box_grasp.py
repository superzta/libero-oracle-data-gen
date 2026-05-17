"""Sweep button_box grasp primitive parameters without video or state helpers."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from debug_button_box_primitives import PrimitiveConfig, run_seed
from libero_env_utils import configure_runtime_env, resolve_bddl_path

configure_runtime_env()


def candidate_sets(max_sets: int) -> List[Dict[str, float]]:
    grasp_z_values = [-0.015, -0.010, -0.005, 0.000, 0.005, 0.010, 0.020, 0.035]
    pregrasp_z_values = [0.08, 0.10, 0.12, 0.16]
    close_steps_values = [20, 30, 40, 60]
    descent_delta_values = [0.01, 0.015, 0.02, 0.035]
    lift_delta_values = [0.01, 0.015, 0.02, 0.025]
    xy_offsets = [(0.0, -0.02), (0.0, -0.01), (0.0, 0.0), (0.0, 0.01), (0.0, 0.02), (-0.01, -0.02), (0.01, -0.02)]

    base = {
        "grasp_z_offset": 0.035,
        "pregrasp_z_offset": 0.16,
        "close_hold_steps": 36,
        "descent_max_delta": 0.035,
        "lift_max_delta": 0.025,
        "grasp_x_offset": 0.0,
        "grasp_y_offset": -0.02,
    }
    out: List[Dict[str, float]] = [dict(base)]

    for value in grasp_z_values:
        item = dict(base)
        item["grasp_z_offset"] = value
        out.append(item)
    for value in pregrasp_z_values:
        item = dict(base)
        item["pregrasp_z_offset"] = value
        out.append(item)
    for value in close_steps_values:
        item = dict(base)
        item["close_hold_steps"] = value
        out.append(item)
    for value in descent_delta_values:
        item = dict(base)
        item["descent_max_delta"] = value
        out.append(item)
    for value in lift_delta_values:
        item = dict(base)
        item["lift_max_delta"] = value
        out.append(item)
    for x_offset, y_offset in xy_offsets:
        item = dict(base)
        item["grasp_x_offset"] = x_offset
        item["grasp_y_offset"] = y_offset
        out.append(item)

    combo_values = itertools.product(
        [-0.005, 0.0, 0.005, 0.02],
        [0.10, 0.12, 0.16],
        [30, 40, 60],
        [0.015, 0.02, 0.035],
        [0.015, 0.02, 0.025],
        [(0.0, -0.02), (0.0, -0.01), (0.0, 0.0)],
    )
    for grasp_z, pre_z, close_steps, desc_delta, lift_delta, (x_offset, y_offset) in combo_values:
        item = {
            "grasp_z_offset": grasp_z,
            "pregrasp_z_offset": pre_z,
            "close_hold_steps": close_steps,
            "descent_max_delta": desc_delta,
            "lift_max_delta": lift_delta,
            "grasp_x_offset": x_offset,
            "grasp_y_offset": y_offset,
        }
        out.append(item)

    unique = []
    seen = set()
    for item in out:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(item)
        if len(unique) >= max_sets:
            break
    return unique


def config_from_params(params: Dict[str, float]) -> PrimitiveConfig:
    cfg = PrimitiveConfig()
    cfg.grasp_z = float(params["grasp_z_offset"])
    cfg.approach_z = float(params["pregrasp_z_offset"])
    cfg.close_steps = int(params["close_hold_steps"])
    cfg.descent_max_delta = float(params["descent_max_delta"])
    cfg.lift_max_delta = float(params["lift_max_delta"])
    cfg.grasp_x_offset = float(params["grasp_x_offset"])
    cfg.grasp_y_offset = float(params["grasp_y_offset"])
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--max-sets", type=int, default=8)
    parser.add_argument("--output-dir", default="reports/button_box_primitive_debug")
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=resolve_bddl_path(custom_task="button_box"),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        camera_names=["agentview"],
        horizon=900,
    )
    results = []
    try:
        for idx, params in enumerate(candidate_sets(args.max_sets)):
            cfg = config_from_params(params)
            per_seed = [run_seed(env, "pick_lift_only", args.seed + seed_idx, cfg) for seed_idx in range(args.num_seeds)]
            pass_count = sum(1 for item in per_seed if item["passed"])
            z_values = [float(item["cube_z_increase_after_lift"] or item["cube_z_increase"]) for item in per_seed]
            xy_errors = [
                item.get("pick_lift_diagnostics", {}).get("xy_error_at_close")
                for item in per_seed
                if item.get("pick_lift_diagnostics", {}).get("xy_error_at_close") is not None
            ]
            failures = Counter(item["failure_reason"] or "passed" for item in per_seed)
            result = {
                "index": idx,
                "params": params,
                "pass_count": pass_count,
                "num_seeds": args.num_seeds,
                "average_cube_z_increase": float(np.mean(z_values)) if z_values else 0.0,
                "average_xy_error_at_close": float(np.mean(xy_errors)) if xy_errors else None,
                "failure_reasons": dict(failures),
                "per_seed_summary": [
                    {
                        "seed": item["seed"],
                        "passed": item["passed"],
                        "failure_reason": item["failure_reason"],
                        "cube_z_increase": item["cube_z_increase_after_lift"],
                        "xy_error_at_close": item.get("pick_lift_diagnostics", {}).get("xy_error_at_close"),
                    }
                    for item in per_seed
                ],
            }
            results.append(result)
            print(
                f"set={idx:03d} pass={pass_count}/{args.num_seeds} "
                f"z={result['average_cube_z_increase']:.4f} xyerr={result['average_xy_error_at_close']} params={params}",
                flush=True,
            )
            if pass_count == args.num_seeds:
                break
    finally:
        env.close()

    best = max(results, key=lambda item: (item["pass_count"], item["average_cube_z_increase"], -(item["average_xy_error_at_close"] or 999.0))) if results else None
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_seeds": args.num_seeds,
        "strict": bool(args.strict),
        "best_parameter_set": None if best is None else best["params"],
        "best_result": best,
        "results": results,
        "passed": bool(best and best["pass_count"] == args.num_seeds),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grasp_sweep.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2, sort_keys=True))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
