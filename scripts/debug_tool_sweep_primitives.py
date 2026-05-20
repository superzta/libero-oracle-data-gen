"""Primitive-level non-video debugging for the tool_sweep task.

Modes:
  grasp_pusher_only  -- approach, grasp, lift pusher; report grasp success
  push_block_only    -- same as grasp_pusher_only, then reposition and execute sweep
  full               -- complete pipeline including settle/retract/verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libero_env_utils import configure_runtime_env, resolve_bddl_path
from tool_sweep_reset_utils import apply_tool_sweep_reset_randomization
from qa_button_box_rollout import JointWriteGuard

configure_runtime_env()

PUSHER = "pusher_tool_1"
BLOCK = "red_block_1"
DUSTPAN = "dustpan_1"


def eef_pos(obs: Dict[str, Any]) -> np.ndarray:
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float32)


def obj_pos(env, obs: Dict[str, Any], name: str) -> np.ndarray:
    key = f"{name}_pos"
    if key in obs:
        return np.asarray(obs[key], dtype=np.float32)
    return np.asarray(env.env.sim.data.body_xpos[env.env.obj_body_id[name]], dtype=np.float32)


def contact_info(env, name: str) -> List[List[str]]:
    try:
        sim = env.env.sim
        pairs = []
        for idx in range(sim.data.ncon):
            c = sim.data.contact[idx]
            n1 = sim.model.geom_id2name(c.geom1) or ""
            n2 = sim.model.geom_id2name(c.geom2) or ""
            joined = f"{n1} {n2}".lower()
            if name.split("_")[0] in joined or "gripper" in joined or "finger" in joined:
                pairs.append([n1, n2])
        return pairs
    except Exception:
        return []


def gripper_on_pusher(pairs: List[List[str]]) -> bool:
    for pair in pairs:
        txt = " ".join(str(p).lower() for p in pair)
        if "pusher" in txt and ("gripper" in txt or "finger" in txt):
            return True
    return False


def make_action(obs, target, gripper, gain=20.0, max_delta=0.10) -> np.ndarray:
    delta = (np.asarray(target, dtype=np.float32) - eef_pos(obs)) * gain
    delta = np.clip(delta, -max_delta, max_delta)
    return np.concatenate([delta, np.zeros(3, dtype=np.float32), [float(gripper)]])


def wait_action(gripper: float) -> np.ndarray:
    return np.array([0, 0, 0, 0, 0, 0, float(gripper)], dtype=np.float32)


def step_env(env, obs, action, trace, stage, step_idx, pusher_pos, block_pos, dustpan_pos):
    obs, reward, done, info = env.step(action)
    pusher = obj_pos(env, obs, PUSHER)
    block = obj_pos(env, obs, BLOCK)
    dustpan = obj_pos(env, obs, DUSTPAN)
    pairs = contact_info(env, PUSHER)
    trace.append({
        "step": step_idx,
        "stage": stage,
        "eef": eef_pos(obs).round(5).tolist(),
        "pusher": pusher.round(5).tolist(),
        "block": block.round(5).tolist(),
        "dustpan": dustpan.round(5).tolist(),
        "gripper": float(action[-1]),
        "contact_pusher_gripper": gripper_on_pusher(pairs),
    })
    return obs, done


def move_to(env, obs, target, gripper, trace, stage, step_idx, max_steps=160, gain=20.0, max_delta=0.10, tol=0.030):
    for _ in range(max_steps):
        obs, done = step_env(env, obs, make_action(obs, target, gripper, gain, max_delta), trace, stage, step_idx,
                             obj_pos(env, obs, PUSHER), obj_pos(env, obs, BLOCK), obj_pos(env, obs, DUSTPAN))
        step_idx += 1
        if np.linalg.norm(eef_pos(obs) - np.asarray(target)) <= tol or done:
            break
    return obs, step_idx


def hold(env, obs, n_steps, gripper, trace, stage, step_idx, target=None, max_delta=0.010):
    for _ in range(n_steps):
        action = wait_action(gripper) if target is None else make_action(obs, target, gripper, gain=20.0, max_delta=max_delta)
        obs, done = step_env(env, obs, action, trace, stage, step_idx,
                             obj_pos(env, obs, PUSHER), obj_pos(env, obs, BLOCK), obj_pos(env, obs, DUSTPAN))
        step_idx += 1
        if done:
            break
    return obs, step_idx


def run_grasp_pusher(env, obs, p0, trace, step_idx):
    above = p0 + np.array([0.0, 0.0, 0.120])
    grasp = p0 + np.array([0.0, 0.0, 0.005])
    lift = p0 + np.array([0.0, 0.0, 0.055])
    obs, step_idx = move_to(env, obs, above, -1.0, trace, "MOVE_ABOVE_PUSHER", step_idx, max_steps=300, max_delta=0.22, tol=0.040)
    obs, step_idx = move_to(env, obs, grasp, -1.0, trace, "DESCEND_TO_PUSHER", step_idx, max_steps=300, max_delta=0.050, tol=0.020)
    obs, step_idx = hold(env, obs, 30, 1.0, trace, "CLOSE_GRIPPER", step_idx, target=grasp)
    obs, step_idx = hold(env, obs, 80, 1.0, trace, "LIFT_PUSHER", step_idx, target=lift, max_delta=0.060)
    obs, step_idx = hold(env, obs, 10, 1.0, trace, "LIFT_HOLD", step_idx, target=lift)
    return obs, step_idx


def run_sweep(env, obs, p0, b0, d0, lane_dir, trace, step_idx):
    """Grasp pusher, move behind block, lower to sweep height, sweep to dustpan."""
    obs, step_idx = run_grasp_pusher(env, obs, p0, trace, step_idx)
    lift_z = 0.055
    sweep_z = 0.031  # blade ~10mm above table; matches ToolSweepController default
    behind_xy = b0[:2] - 0.130 * lane_dir
    behind_high = np.array([behind_xy[0], behind_xy[1], p0[2] + 0.005 + lift_z])
    behind_sweep = np.array([behind_xy[0], behind_xy[1], p0[2] + sweep_z])
    sweep_end = np.array([d0[0] + 0.040 * lane_dir[0], d0[1] + 0.040 * lane_dir[1], p0[2] + sweep_z])
    obs, step_idx = move_to(env, obs, behind_high, 1.0, trace, "MOVE_BEHIND_BLOCK", step_idx, max_steps=240, max_delta=0.080, tol=0.040)
    obs, step_idx = move_to(env, obs, behind_sweep, 1.0, trace, "LOWER_TO_SWEEP", step_idx, max_steps=100, max_delta=0.040, tol=0.012)
    obs, step_idx = move_to(env, obs, sweep_end, 1.0, trace, "SWEEP", step_idx, max_steps=700, gain=15.0, max_delta=0.060, tol=0.040)
    return obs, step_idx


def run_full(env, obs, p0, b0, d0, lane_dir, trace, step_idx):
    obs, step_idx = run_sweep(env, obs, p0, b0, d0, lane_dir, trace, step_idx)
    obs, step_idx = hold(env, obs, 40, 1.0, trace, "WAIT_SETTLE", step_idx)
    retract = np.array([d0[0] - 0.050 * lane_dir[0], d0[1] - 0.050 * lane_dir[1], p0[2] + 0.170])
    obs, step_idx = move_to(env, obs, retract, -1.0, trace, "RETRACT", step_idx, max_steps=100, max_delta=0.120, tol=0.060)
    obs, step_idx = hold(env, obs, 10, -1.0, trace, "VERIFY", step_idx)
    return obs, step_idx


def classify(mode, trace, initial_pusher, initial_block, final_block, dustpan, lane_dir, direct_writes):
    if direct_writes:
        return "direct_pose_write_used"
    stages = [t["stage"] for t in trace]
    # Grasp check: pusher lifted during LIFT_PUSHER
    lift_items = [t for t in trace if t["stage"] == "LIFT_PUSHER"]
    if mode in {"grasp_pusher_only", "push_block_only", "full"}:
        if not lift_items:
            return "grasp_failed_pusher_not_lifted"
        max_lift = max(float(t["pusher"][2]) - float(initial_pusher[2]) for t in lift_items)
        if max_lift < 0.008:
            return "grasp_failed_pusher_not_lifted"
    if mode in {"push_block_only", "full"}:
        block_moved = float(np.linalg.norm(np.asarray(final_block[:2]) - np.asarray(initial_block[:2])))
        if block_moved < 0.060:
            return "block_not_moved"
        block_xy = np.asarray(final_block[:2])
        pan_xy = np.asarray(dustpan[:2])
        dist = float(np.linalg.norm(block_xy - pan_xy))
        y_local = float(np.dot(block_xy - pan_xy, lane_dir))
        if dist > 0.095 or y_local < -0.015:
            return "block_not_inside_dustpan"
    if mode == "full":
        if "WAIT_SETTLE" not in stages or "RETRACT" not in stages or "VERIFY" not in stages:
            return "incomplete_pipeline"
    return ""


def run_seed(env, mode, seed, randomization_level):
    env.seed(seed)
    obs = env.reset()
    env.env.sim.forward()
    obs, reset_info = apply_tool_sweep_reset_randomization(env, obs, seed, settle_steps=20,
                                                           randomization_level=randomization_level)
    p0 = obj_pos(env, obs, PUSHER).copy()
    b0 = obj_pos(env, obs, BLOCK).copy()
    d0 = obj_pos(env, obs, DUSTPAN).copy()
    diff = d0[:2] - p0[:2]
    norm = np.linalg.norm(diff)
    lane_dir = diff / norm if norm > 1e-6 else np.array([0.0, 1.0])
    trace: List[Dict] = []
    all_joints = []
    for name in [PUSHER, BLOCK, DUSTPAN]:
        try:
            all_joints.extend(env.env.get_object(name).joints)
        except Exception:
            pass
    with JointWriteGuard(env, all_joints) as guard:
        step_idx = 0
        if mode == "grasp_pusher_only":
            obs, step_idx = run_grasp_pusher(env, obs, p0, trace, step_idx)
        elif mode == "push_block_only":
            obs, step_idx = run_sweep(env, obs, p0, b0, d0, lane_dir, trace, step_idx)
        elif mode == "full":
            obs, step_idx = run_full(env, obs, p0, b0, d0, lane_dir, trace, step_idx)
        else:
            raise ValueError(mode)
    final_pusher = obj_pos(env, obs, PUSHER).copy()
    final_block = obj_pos(env, obs, BLOCK).copy()
    failure = classify(mode, trace, p0, b0, final_block, d0, lane_dir, guard.count)
    lift_items = [t for t in trace if t["stage"] == "LIFT_PUSHER"]
    max_lift = max((float(t["pusher"][2]) - float(p0[2]) for t in lift_items), default=0.0)
    block_moved = float(np.linalg.norm(final_block[:2] - b0[:2]))
    pan_xy = d0[:2]
    block_dist_from_pan = float(np.linalg.norm(final_block[:2] - pan_xy))
    y_local = float(np.dot(final_block[:2] - pan_xy, lane_dir))
    return {
        "seed": seed,
        "mode": mode,
        "passed": not failure,
        "failure_reason": failure,
        "pusher_initial_xyz": p0.round(5).tolist(),
        "block_initial_xyz": b0.round(5).tolist(),
        "dustpan_initial_xyz": d0.round(5).tolist(),
        "lane_dir": lane_dir.round(5).tolist(),
        "lane_angle_deg": round(float(np.degrees(np.arctan2(float(lane_dir[1]), float(lane_dir[0])))), 2),
        "pusher_max_lift_m": round(max_lift, 5),
        "pusher_grasped": max_lift >= 0.008,
        "block_moved_m": round(block_moved, 5),
        "block_dist_from_dustpan_m": round(block_dist_from_pan, 5),
        "block_y_local": round(y_local, 5),
        "block_inside_dustpan": block_dist_from_pan <= 0.095 and y_local >= -0.015,
        "block_final_xyz": final_block.round(5).tolist(),
        "direct_pose_writes": int(guard.count),
        "reset_info": reset_info,
        "steps_executed": len(trace),
    }


def print_running_summary(per_seed):
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
    parser.add_argument("--mode", choices=["grasp_pusher_only", "push_block_only", "full"], required=True)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomization-level", default="debug_small",
                        choices=["debug_small", "medium", "diverse_v2"])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", default="reports/tool_sweep_primitive_debug")
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=resolve_bddl_path(custom_task="tool_sweep"),
        camera_heights=64,
        camera_widths=64,
        camera_names=["agentview"],
        horizon=1400,
        ignore_done=True,
    )
    try:
        per_seed = []
        for idx in range(args.num_seeds):
            seed = args.seed + idx
            if not args.quiet:
                print(f"[tool_sweep {args.mode} seed {idx+1}/{args.num_seeds}] seed={seed} start", flush=True)
            result = run_seed(env, args.mode, seed, args.randomization_level)
            per_seed.append(result)
            if not args.quiet:
                status = "PASS" if result["passed"] else "FAIL"
                extra = (
                    f"lift={result['pusher_max_lift_m']:.3f} "
                    f"moved={result['block_moved_m']:.3f} "
                    f"inside={result['block_inside_dustpan']} "
                    f"reason={result['failure_reason'] or 'ok'}"
                )
                print(f"[tool_sweep {args.mode} seed {idx+1}/{args.num_seeds}] seed={seed} {status} {extra}", flush=True)
                print_running_summary(per_seed)
    finally:
        env.close()

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "randomization_level": args.randomization_level,
        "num_seeds": args.num_seeds,
        "passed": all(r["passed"] for r in per_seed),
        "per_seed": per_seed,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.mode}_{args.randomization_level}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    console = {**{k: v for k, v in report.items() if k != "per_seed"},
               "per_seed": [{k: v for k, v in r.items() if k not in {"reset_info"}} for r in per_seed]}
    print(json.dumps(console, indent=2, sort_keys=True))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
