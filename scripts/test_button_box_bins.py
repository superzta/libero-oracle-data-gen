"""Per-bin feasibility testing for button_box diverse_v2 randomization.

For each (cube_bin_id, box_bin_id) combination, runs a small number of seeds
through the pick+lift primitive and reports pass/fail rates.  Infeasible bins
can then be removed from the diverse_v2 config.

Usage:
  # Quick pick+lift test (default, ~2 min for 6×4 bins × 3 seeds):
  python scripts/test_button_box_bins.py --mode pick_lift --seeds-per-bin 3

  # Full rollout test (slower, more reliable):
  python scripts/test_button_box_bins.py --mode full --seeds-per-bin 3 --horizon 1200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libero_env_utils import configure_runtime_env, resolve_bddl_path

configure_runtime_env()

from button_box_reset_utils import (
    _LEVEL_CONFIGS,
    apply_button_box_reset_randomization,
    compute_reset_positions,
)

LEVEL = "diverse_v2"
CUBE_BIN_NAMES = ["left-front", "center-front", "right-front", "left-back", "center-back", "right-back"]
BOX_BIN_NAMES  = ["box-left", "box-center", "box-right", "box-back-center"]


def _seeds_for_combo(cube_bin_id: int, box_bin_id: int, n_cube_bins: int, n_box_bins: int,
                     n_seeds: int, search_range: int = 500) -> List[int]:
    """Return up to n_seeds seeds that map to (cube_bin_id, box_bin_id).

    Matches compute_reset_positions: cube_bin = seed % n_cube_bins,
    box_bin = (seed // n_cube_bins) % n_box_bins.
    """
    found = []
    for s in range(search_range):
        if s % n_cube_bins == cube_bin_id and s % n_box_bins == box_bin_id:
            found.append(s)
            if len(found) >= n_seeds:
                break
    return found


def _test_pick_lift(env, seed: int, horizon: int) -> Dict[str, Any]:
    """Run a minimal pick+lift rollout and return result dict."""
    from controllers.button_box_controller import ButtonBoxController

    env.seed(seed)
    obs = env.reset()
    try:
        env.env.sim.forward()
    except Exception:
        pass
    obs, reset_info = apply_button_box_reset_randomization(env, obs, seed, settle_steps=20, randomization_level=LEVEL)
    cube_init = np.asarray(obs.get("blue_cube_1_pos", [0, 0, 0]), dtype=np.float32).copy()

    controller = ButtonBoxController()
    controller.reset(env, obs, {})

    cube_lifted = False
    steps_to_lift = horizon
    for step in range(horizon):
        action = np.asarray(controller.act(obs), dtype=np.float32)
        obs, _reward, _done, _info = env.step(action)
        cube_pos = np.asarray(obs.get("blue_cube_1_pos", [0, 0, 0]), dtype=np.float32)
        if cube_pos[2] - cube_init[2] >= 0.040:
            cube_lifted = True
            steps_to_lift = step
            break

    return {
        "seed": seed,
        "cube_bin_id": reset_info["cube_bin_id"],
        "box_bin_id": reset_info["box_bin_id"],
        "cube_init_xyz": cube_init.round(4).tolist(),
        "box_xyz": reset_info["box_xyz"],
        "lifted": cube_lifted,
        "steps": steps_to_lift,
    }


def _test_full_rollout(env, seed: int, horizon: int) -> Dict[str, Any]:
    """Run a full FSM rollout and report success."""
    from controllers.button_box_controller import ButtonBoxController

    env.seed(seed)
    obs = env.reset()
    try:
        env.env.sim.forward()
    except Exception:
        pass
    obs, reset_info = apply_button_box_reset_randomization(env, obs, seed, settle_steps=20, randomization_level=LEVEL)

    controller = ButtonBoxController()
    controller.reset(env, obs, {})

    success = False
    for _step in range(horizon):
        action = np.asarray(controller.act(obs), dtype=np.float32)
        obs, _r, _d, _info = env.step(action)
        if controller.is_success(obs, _info, env):
            success = True
            break

    return {
        "seed": seed,
        "cube_bin_id": reset_info["cube_bin_id"],
        "box_bin_id": reset_info["box_bin_id"],
        "cube_init_xyz": reset_info["cube_xyz"],
        "box_xyz": reset_info["box_xyz"],
        "success": success,
        "stage": controller.stage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-bin feasibility test for diverse_v2.")
    parser.add_argument("--mode", choices=["pick_lift", "full"], default="pick_lift",
                        help="Test mode: pick_lift (fast) or full rollout (slow, default: pick_lift).")
    parser.add_argument("--seeds-per-bin", type=int, default=3,
                        help="Seeds per bin combination (default: 3).")
    parser.add_argument("--horizon", type=int, default=400,
                        help="Steps per rollout for pick_lift; 1200 for full (default: 400).")
    parser.add_argument("--output", default=None, help="JSON output file (default: stdout).")
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=resolve_bddl_path(custom_task="button_box"),
        camera_heights=128, camera_widths=128,
        camera_names=["agentview"],
        horizon=args.horizon,
        ignore_done=True,
    )

    cfg = _LEVEL_CONFIGS[LEVEL]
    n_cube_bins = len(cfg["cube_bins"])
    n_box_bins  = len(cfg["box_bins"])

    results_by_combo: Dict[str, List[dict]] = defaultdict(list)
    summary_rows = []

    total_combos = n_cube_bins * n_box_bins
    combo_idx = 0
    for cb in range(n_cube_bins):
        for bb in range(n_box_bins):
            combo_idx += 1
            seeds = _seeds_for_combo(cb, bb, n_cube_bins, n_box_bins, args.seeds_per_bin)
            combo_key = f"cb{cb}_bb{bb}"
            print(f"[{combo_idx}/{total_combos}] cube_bin={cb} ({CUBE_BIN_NAMES[cb]})  "
                  f"box_bin={bb} ({BOX_BIN_NAMES[bb]})  seeds={seeds}")

            for seed in seeds:
                try:
                    if args.mode == "pick_lift":
                        r = _test_pick_lift(env, seed, args.horizon)
                        passed = r["lifted"]
                    else:
                        r = _test_full_rollout(env, seed, args.horizon)
                        passed = r["success"]
                    r["passed"] = passed
                    results_by_combo[combo_key].append(r)
                    status = "PASS" if passed else "FAIL"
                    print(f"    seed={seed}  {status}  cube_init_xyz={r['cube_init_xyz']}")
                except Exception as exc:
                    print(f"    seed={seed}  ERROR: {exc}")
                    results_by_combo[combo_key].append({"seed": seed, "passed": False, "error": str(exc)})

            combo_results = results_by_combo[combo_key]
            n_pass = sum(1 for r in combo_results if r.get("passed"))
            n_total = len(combo_results)
            feasible = n_pass >= max(1, n_total // 2)
            row = {
                "cube_bin_id": cb,
                "cube_bin_name": CUBE_BIN_NAMES[cb],
                "box_bin_id": bb,
                "box_bin_name": BOX_BIN_NAMES[bb],
                "pass_count": n_pass,
                "total_count": n_total,
                "pass_rate": round(n_pass / n_total, 2) if n_total else 0,
                "feasible": feasible,
            }
            summary_rows.append(row)
            verdict = "OK" if feasible else "INFEASIBLE"
            print(f"    → {n_pass}/{n_total} passed  [{verdict}]")

    env.close()

    report = {
        "level": LEVEL,
        "mode": args.mode,
        "seeds_per_bin": args.seeds_per_bin,
        "summary": summary_rows,
        "infeasible_bins": [
            {"cube_bin_id": r["cube_bin_id"], "cube_bin_name": r["cube_bin_name"],
             "box_bin_id": r["box_bin_id"], "box_bin_name": r["box_bin_name"]}
            for r in summary_rows if not r["feasible"]
        ],
        "details": {k: v for k, v in results_by_combo.items()},
    }

    out_str = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(out_str)
        print(f"\nReport saved to {args.output}")
    else:
        print("\n" + out_str)


if __name__ == "__main__":
    main()
