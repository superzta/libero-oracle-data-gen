"""Non-video QA for the tool_sweep oracle rollout.

Checks per seed:
  - FSM reaches DONE
  - Pusher grasped and moved
  - Block moved significantly
  - Block inside dustpan at end
  - No oracle helper
  - No direct rollout pose writes
  - Success gated after WAIT_SETTLE + RETRACT + VERIFY_FINAL_STATE
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controllers.tool_sweep_controller import ToolSweepController
from tool_sweep_reset_utils import apply_tool_sweep_reset_randomization
from libero_env_utils import configure_runtime_env, resolve_bddl_path
from qa_button_box_rollout import JointWriteGuard

configure_runtime_env()

PUSHER = "pusher_tool_1"
BLOCK = "red_block_1"
DUSTPAN = "dustpan_1"


def obj_pos(env, obs, name):
    key = f"{name}_pos"
    if key in obs:
        return np.asarray(obs[key], dtype=np.float32)
    return np.asarray(env.env.sim.data.body_xpos[env.env.obj_body_id[name]], dtype=np.float32)


def run_seed(env, seed: int, horizon: int, randomization_level: str) -> Dict[str, Any]:
    env.seed(seed)
    obs = env.reset()
    try:
        env.env.sim.forward()
    except Exception:
        pass
    obs, reset_info = apply_tool_sweep_reset_randomization(env, obs, seed, settle_steps=20,
                                                           randomization_level=randomization_level)
    controller = ToolSweepController()
    controller.reset(env, obs, {})

    initial_pusher = obj_pos(env, obs, PUSHER).copy()
    initial_block = obj_pos(env, obs, BLOCK).copy()
    initial_dustpan = obj_pos(env, obs, DUSTPAN).copy()

    all_joints: List[str] = []
    for name in [PUSHER, BLOCK, DUSTPAN]:
        try:
            all_joints.extend(env.env.get_object(name).joints)
        except Exception:
            pass

    stage_trace: List[str] = []
    success_step = None
    failure_reason = "max_steps"

    with JointWriteGuard(env, all_joints) as guard:
        for step in range(horizon):
            action = np.asarray(controller.act(obs), dtype=np.float32)
            obs, reward, done, info = env.step(action)
            dbg = controller.get_debug_state()
            stage = dbg.get("stage", "")
            if not stage_trace or stage_trace[-1] != stage:
                stage_trace.append(stage)
            if controller.is_success(obs, info, env) and success_step is None:
                success_step = step
                failure_reason = ""
            if stage == "DONE" and success_step is None:
                failure_reason = dbg.get("transition_reason") or "controller_done_no_success"
                break

    final_pusher = obj_pos(env, obs, PUSHER).copy()
    final_block = obj_pos(env, obs, BLOCK).copy()
    final_dustpan = obj_pos(env, obs, DUSTPAN).copy()

    block_moved = float(np.linalg.norm(final_block[:2] - initial_block[:2]))
    pan_xy = final_dustpan[:2]
    block_dist_from_pan = float(np.linalg.norm(final_block[:2] - pan_xy))
    lane_dir = controller.lane_dir_2d
    y_local = float(np.dot(final_block[:2] - pan_xy, lane_dir))
    block_inside = block_dist_from_pan <= 0.095 and y_local >= -0.015

    if failure_reason == "max_steps" and success_step is None:
        if "DONE" not in stage_trace:
            failure_reason = "did_not_reach_done"
        elif not block_inside:
            failure_reason = "block_not_inside_dustpan"
        elif block_moved < 0.080:
            failure_reason = "block_not_moved"
        elif not controller.pusher_grasped:
            failure_reason = "pusher_not_grasped"
        else:
            failure_reason = "verification_failed"

    passed = success_step is not None and guard.count == 0 and not controller.oracle_helper_used

    return {
        "seed": seed,
        "passed": passed,
        "failure_reason": failure_reason if not passed else "",
        "success_step": success_step,
        "stage_trace": stage_trace,
        "pusher_grasped": controller.pusher_grasped,
        "max_pusher_z_delta": round(controller.max_pusher_z_delta, 5),
        "block_moved_m": round(block_moved, 5),
        "block_dist_from_dustpan_m": round(block_dist_from_pan, 5),
        "block_inside_dustpan": block_inside,
        "block_y_local": round(y_local, 5),
        "direct_pose_writes": int(guard.count),
        "oracle_helper_used": bool(controller.oracle_helper_used),
        "settle_steps_done": controller.settle_steps_done,
        "reset_info": reset_info,
        "pusher_initial_xyz": initial_pusher.round(5).tolist(),
        "block_initial_xyz": initial_block.round(5).tolist(),
        "dustpan_initial_xyz": initial_dustpan.round(5).tolist(),
        "block_final_xyz": final_block.round(5).tolist(),
    }


def print_running_summary(per_seed: List[Dict]) -> None:
    pass_count = sum(1 for r in per_seed if r["passed"])
    fail_count = len(per_seed) - pass_count
    reasons = Counter(r["failure_reason"] for r in per_seed if not r["passed"])
    common = reasons.most_common(1)[0][0] if reasons else "none"
    print(
        f"running: attempts={len(per_seed)} pass={pass_count} fail={fail_count} "
        f"success_rate={pass_count / len(per_seed):.3f} most_common_failure={common}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1400)
    parser.add_argument("--randomization-level", default="debug_small",
                        choices=["debug_small", "medium", "diverse_v2"])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", default="reports/tool_sweep_qa")
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=resolve_bddl_path(custom_task="tool_sweep"),
        camera_heights=64,
        camera_widths=64,
        camera_names=["agentview"],
        horizon=args.horizon + 50,
        ignore_done=True,
    )
    try:
        per_seed = []
        for idx in range(args.num_seeds):
            seed = args.seed + idx
            if not args.quiet:
                print(f"[tool_sweep seed {idx+1}/{args.num_seeds}] seed={seed} start", flush=True)
            result = run_seed(env, seed, args.horizon, args.randomization_level)
            per_seed.append(result)
            if not args.quiet:
                status = "PASS" if result["passed"] else "FAIL"
                print(
                    f"[tool_sweep seed {idx+1}/{args.num_seeds}] seed={seed} {status} "
                    f"moved={result['block_moved_m']:.3f} "
                    f"inside={result['block_inside_dustpan']} "
                    f"reason={result['failure_reason'] or 'ok'}",
                    flush=True,
                )
                print_running_summary(per_seed)
    finally:
        env.close()

    overall_pass = all(r["passed"] for r in per_seed)
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "randomization_level": args.randomization_level,
        "num_seeds": args.num_seeds,
        "horizon": args.horizon,
        "passed": overall_pass,
        "success_rate": sum(1 for r in per_seed if r["passed"]) / len(per_seed),
        "failure_reasons": dict(Counter(r["failure_reason"] for r in per_seed if not r["passed"])),
        "per_seed": per_seed,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"qa_{args.randomization_level}_{args.num_seeds}seeds.json"
    (out_dir / fname).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    console = {k: v for k, v in report.items() if k != "per_seed"}
    console["per_seed"] = [{k: v for k, v in r.items() if k != "reset_info"} for r in per_seed]
    print(json.dumps(console, indent=2, sort_keys=True))
    if args.strict and not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
