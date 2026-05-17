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
    grasp_z_values = [-0.010, -0.005, 0.000, 0.005, 0.010, 0.015, 0.020, 0.025]
    pregrasp_z_values = [0.10, 0.12, 0.16, 0.18]
    close_steps_values = [50, 60, 80]
    descent_delta_values = [0.02, 0.035, 0.05]
    lift_delta_values = [0.02, 0.035, 0.05]
    xy_offsets = [(0.0, 0.0), (0.0, 0.005), (0.0, -0.005), (0.005, 0.0), (-0.005, 0.0)]

    base = {
        "grasp_z_offset": 0.020,
        "pregrasp_z_offset": 0.18,
        "close_hold_steps": 60,
        "descent_max_delta": 0.035,
        "lift_max_delta": 0.05,
        "grasp_x_offset": 0.0,
        "grasp_y_offset": 0.0,
    }
    out: List[Dict[str, float]] = [
        dict(base),
        {**base, "grasp_z_offset": 0.000, "close_hold_steps": 80},
        {**base, "grasp_z_offset": 0.005, "close_hold_steps": 80},
        {**base, "grasp_z_offset": 0.010, "close_hold_steps": 80},
        {**base, "grasp_z_offset": 0.000, "descent_max_delta": 0.02, "close_hold_steps": 80},
    ]

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
        [-0.005, 0.0, 0.005, 0.010, 0.020],
        [0.12, 0.16, 0.18],
        [60, 80],
        [0.02, 0.035, 0.05],
        [0.035, 0.05],
        [(0.0, 0.0), (0.0, 0.005), (0.0, -0.005)],
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
    parser.add_argument(
        "--randomization-level",
        default="debug_small",
        choices=["debug_small", "medium", "final", "diverse"],
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=resolve_bddl_path(custom_task="button_box"),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        camera_names=["agentview"],
        horizon=900,
        ignore_done=True,
    )
    results = []
    param_sets = candidate_sets(args.max_sets)
    try:
        for idx, params in enumerate(param_sets):
            cfg = config_from_params(params)
            if not args.quiet:
                print(
                    "[sweep {idx:02d}/{total:02d}] params={{grasp_z={grasp:.3f}, "
                    "pregrasp={pre:.3f}, close_hold={close}, descent={desc:.3f}, "
                    "lift={lift:.3f}, xoff={xoff:.3f}, yoff={yoff:.3f}}}".format(
                        idx=idx + 1,
                        total=len(param_sets),
                        grasp=params["grasp_z_offset"],
                        pre=params["pregrasp_z_offset"],
                        close=params["close_hold_steps"],
                        desc=params["descent_max_delta"],
                        lift=params["lift_max_delta"],
                        xoff=params["grasp_x_offset"],
                        yoff=params["grasp_y_offset"],
                    ),
                    flush=True,
                )
            per_seed = []
            for seed_idx in range(args.num_seeds):
                seed = args.seed + seed_idx
                item = run_seed(env, "pick_lift_only", seed, cfg, randomization_level=args.randomization_level)
                per_seed.append(item)
                if not args.quiet:
                    status = "PASS" if item["passed"] else "FAIL"
                    z_inc = float(item.get("cube_z_increase_after_lift") or item.get("cube_z_increase") or 0.0)
                    reason = item["failure_reason"] or "ok"
                    print(f"  seed={seed} {status} cube_z_inc={z_inc:.3f} reason={reason}", flush=True)
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
            if not args.quiet:
                rate = pass_count / args.num_seeds if args.num_seeds else 0.0
                fail_count = args.num_seeds - pass_count
                common = Counter(item["failure_reason"] or "passed" for item in per_seed if not item["passed"]).most_common(1)
                common_reason = common[0][0] if common else "none"
                print(
                    f"  running: {pass_count}/{args.num_seeds} pass, fail={fail_count}, "
                    f"success_rate={rate:.3f}, most_common_failure={common_reason}",
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
        "randomization_level": args.randomization_level,
        "best_parameter_set": None if best is None else best["params"],
        "best_result": best,
        "results": results,
        "passed": bool(best and best["pass_count"] == args.num_seeds),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grasp_sweep.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "grasp_sweep_results.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if best is not None:
        print(f"best_parameter_set={best['params']} pass={best['pass_count']}/{best['num_seeds']}", flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2, sort_keys=True))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
