"""Primitive-level non-video debugging for peg_insertion manipulation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libero_env_utils import configure_runtime_env, resolve_bddl_path
from peg_insertion_reset_utils import apply_peg_insertion_reset_randomization
from qa_button_box_rollout import JointWriteGuard

configure_runtime_env()


PEG = "green_peg_1"
BLOCK = "wooden_hole_block_1"
PEG_ASSET = "green_peg_4p4cm_square_11cm_tall"
PEG_SIZE_M = [0.044, 0.044, 0.110]
TRANSPORT_STAGES = {"MOVE_ABOVE_HOLE", "ALIGN_WITH_HOLE", "LOWER_INSERT"}


@dataclass
class PrimitiveConfig:
    pos_gain: float = 20.0
    max_delta: float = 0.18
    approach_z: float = 0.135
    grasp_z: float = 0.005
    lift_z: float = 0.145
    hole_above_z: float = 0.195
    align_z: float = 0.120
    insert_z: float = 0.040
    retract_z: float = 0.230
    close_steps: int = 70
    hold_insert_steps: int = 12
    open_steps: int = 18
    settle_steps: int = 35
    reset_settle_steps: int = 20
    move_tol: float = 0.030


def eef_pos(obs: Dict[str, Any]) -> np.ndarray:
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float32)


def object_pos(env, obs: Dict[str, Any], name: str) -> np.ndarray:
    key = f"{name}_pos"
    if key in obs:
        return np.asarray(obs[key], dtype=np.float32)
    return np.asarray(env.env.sim.data.body_xpos[env.env.obj_body_id[name]], dtype=np.float32)


def object_quat(env, obs: Dict[str, Any], name: str) -> Optional[np.ndarray]:
    key = f"{name}_quat"
    if key in obs:
        return np.asarray(obs[key], dtype=np.float32)
    try:
        body_id = env.env.obj_body_id[name]
        return np.asarray(env.env.sim.data.body_xquat[body_id], dtype=np.float32)
    except Exception:
        return None


def contact_pairs(env, object_name: str = PEG) -> List[List[str]]:
    try:
        sim = env.env.sim
        pairs = []
        for idx in range(sim.data.ncon):
            contact = sim.data.contact[idx]
            name1 = sim.model.geom_id2name(contact.geom1) or ""
            name2 = sim.model.geom_id2name(contact.geom2) or ""
            joined = f"{name1} {name2}".lower()
            if object_name in joined or "gripper" in joined or "finger" in joined:
                pairs.append([name1, name2])
        return pairs
    except Exception:
        return []


def gripper_contacted_peg(pairs: List[List[str]]) -> bool:
    for pair in pairs:
        text = " ".join(str(part).lower() for part in pair)
        if PEG in text and ("gripper" in text or "finger" in text):
            return True
    return False


def hole_pos(block: np.ndarray) -> np.ndarray:
    return np.asarray(block, dtype=np.float32) + np.array([0.0, 0.0, 0.021], dtype=np.float32)


def make_action(obs: Dict[str, Any], target: np.ndarray, gripper: float, cfg: PrimitiveConfig, max_delta: Optional[float] = None) -> np.ndarray:
    delta = (np.asarray(target, dtype=np.float32) - eef_pos(obs)) * cfg.pos_gain
    delta = np.clip(delta, -(max_delta or cfg.max_delta), max_delta or cfg.max_delta)
    return np.concatenate([delta, np.zeros(3, dtype=np.float32), np.asarray([gripper], dtype=np.float32)])


def wait_action(gripper: float) -> np.ndarray:
    return np.asarray([0, 0, 0, 0, 0, 0, gripper], dtype=np.float32)


def step_env(env, obs, action, trace: List[Dict[str, Any]], stage: str, target: Optional[np.ndarray], step_idx: int):
    obs, reward, done, info = env.step(action)
    peg = object_pos(env, obs, PEG)
    block = object_pos(env, obs, BLOCK)
    peg_quat = object_quat(env, obs, PEG)
    eef = eef_pos(obs)
    pairs = contact_pairs(env)
    trace.append(
        {
            "step": step_idx,
            "stage": stage,
            "eef_position": eef.astype(float).round(6).tolist(),
            "peg_position": peg.astype(float).round(6).tolist(),
            "block_position": block.astype(float).round(6).tolist(),
            "hole_position": hole_pos(block).astype(float).round(6).tolist(),
            "peg_quat": None if peg_quat is None else peg_quat.astype(float).round(6).tolist(),
            "target_position": None if target is None else np.asarray(target).astype(float).round(6).tolist(),
            "gripper_command": float(action[-1]),
            "peg_eef_distance": float(np.linalg.norm(peg - eef)),
            "peg_eef_xy_distance": float(np.linalg.norm(peg[:2] - eef[:2])),
            "peg_hole_xy_error": float(np.linalg.norm(peg[:2] - block[:2])),
            "contact_pairs": pairs,
            "gripper_contacted_peg": gripper_contacted_peg(pairs),
            "reward": float(reward),
            "done": bool(done),
        }
    )
    return obs, False, info


def move_to(env, obs, target, gripper, cfg, trace, stage, step_idx, max_steps=120, max_delta=None):
    done = False
    for _ in range(max_steps):
        obs, done, info = step_env(env, obs, make_action(obs, target, gripper, cfg, max_delta), trace, stage, target, step_idx)
        step_idx += 1
        if np.linalg.norm(eef_pos(obs) - target) <= cfg.move_tol or done:
            break
    return obs, step_idx, done


def hold(env, obs, n_steps, gripper, trace, stage, step_idx, target=None):
    done = False
    for _ in range(n_steps):
        action = wait_action(gripper) if target is None else make_action(obs, target, gripper, PrimitiveConfig(), max_delta=0.006)
        obs, done, info = step_env(env, obs, action, trace, stage, target, step_idx)
        step_idx += 1
        if done:
            break
    return obs, step_idx, done


def run_pick_lift(env, obs, cfg: PrimitiveConfig, trace: List[Dict[str, Any]], step_idx: int):
    peg0 = object_pos(env, obs, PEG).copy()
    above = peg0 + np.array([0.0, 0.0, cfg.approach_z], dtype=np.float32)
    obs, step_idx, done = move_to(env, obs, above, -1.0, cfg, trace, "MOVE_ABOVE_PEG", step_idx, max_steps=160, max_delta=0.18)
    if done:
        return obs, step_idx, done
    peg_before = object_pos(env, obs, PEG).copy()
    grasp = peg_before + np.array([0.0, 0.0, cfg.grasp_z], dtype=np.float32)
    obs, step_idx, done = move_to(env, obs, grasp, -1.0, cfg, trace, "DESCEND_TO_PEG", step_idx, max_steps=220, max_delta=0.045)
    if done:
        return obs, step_idx, done
    obs, step_idx, done = move_to(env, obs, grasp, 1.0, cfg, trace, "CLOSE_GRIPPER_AND_WAIT", step_idx, max_steps=cfg.close_steps, max_delta=0.010)
    if done:
        return obs, step_idx, done
    lift = object_pos(env, obs, PEG) + np.array([0.0, 0.0, cfg.lift_z], dtype=np.float32)
    obs, step_idx, done = move_to(env, obs, lift, 1.0, cfg, trace, "LIFT_PEG", step_idx, max_steps=140, max_delta=0.050)
    return obs, step_idx, done


def run_insert(env, obs, cfg: PrimitiveConfig, trace: List[Dict[str, Any]], step_idx: int):
    obs, step_idx, done = run_pick_lift(env, obs, cfg, trace, step_idx)
    if done:
        return obs, step_idx, done
    hole = hole_pos(object_pos(env, obs, BLOCK))
    above = hole + np.array([0.0, 0.0, cfg.hole_above_z], dtype=np.float32)
    align = hole + np.array([0.0, 0.0, cfg.align_z], dtype=np.float32)
    insert = hole + np.array([0.0, 0.0, cfg.insert_z], dtype=np.float32)
    obs, step_idx, done = move_to(env, obs, above, 1.0, cfg, trace, "MOVE_ABOVE_HOLE", step_idx, max_steps=460, max_delta=0.026)
    obs, step_idx, done = move_to(env, obs, align, 1.0, cfg, trace, "ALIGN_WITH_HOLE", step_idx, max_steps=220, max_delta=0.018)
    obs, step_idx, done = move_to(env, obs, insert, 1.0, cfg, trace, "LOWER_INSERT", step_idx, max_steps=280, max_delta=0.010)
    obs, step_idx, done = hold(env, obs, cfg.hold_insert_steps, 1.0, trace, "HOLD_INSERT", step_idx, target=insert)
    obs, step_idx, done = hold(env, obs, cfg.open_steps, -1.0, trace, "OPEN_GRIPPER", step_idx, target=insert)
    obs, step_idx, done = hold(env, obs, cfg.settle_steps, -1.0, trace, "WAIT_SETTLE", step_idx)
    retract = hole + np.array([0.0, 0.0, cfg.retract_z], dtype=np.float32)
    obs, step_idx, done = move_to(env, obs, retract, -1.0, cfg, trace, "RETRACT", step_idx, max_steps=80, max_delta=0.060)
    return obs, step_idx, done


def unique_stages(trace: List[Dict[str, Any]]) -> List[str]:
    stages: List[str] = []
    for item in trace:
        if not stages or stages[-1] != item["stage"]:
            stages.append(item["stage"])
    return stages


def last_pose(trace: List[Dict[str, Any]], stage: str, key: str = "peg_position") -> Optional[np.ndarray]:
    for item in reversed(trace):
        if item["stage"] == stage:
            return np.asarray(item[key], dtype=np.float32)
    return None


def classify_failure(mode: str, trace: List[Dict[str, Any]], initial_peg: np.ndarray, final_peg: np.ndarray, final_block: np.ndarray, direct_writes: int) -> str:
    if direct_writes:
        return "direct_pose_write_used"
    stages = unique_stages(trace)
    lift_pose = last_pose(trace, "LIFT_PEG")
    close_pose = last_pose(trace, "CLOSE_GRIPPER_AND_WAIT")
    if close_pose is None:
        close_pose = initial_peg
    peg_lift = float(((lift_pose if lift_pose is not None else final_peg)[2]) - close_pose[2])
    if mode in {"pick_lift_only", "insert_only", "full"}:
        if "LIFT_PEG" not in stages or peg_lift < 0.040:
            return "grasp_failed_peg_not_lifted"
    if mode in {"insert_only", "full"}:
        h = hole_pos(final_block)
        if "OPEN_GRIPPER" not in stages or "WAIT_SETTLE" not in stages or "RETRACT" not in stages:
            return "success_before_release_settle_retract_invalid"
        if np.linalg.norm(final_peg[:2] - h[:2]) > 0.045:
            return "insert_failed_xy_error"
        if abs(float(final_peg[2] - (h[2] + 0.034))) > 0.055:
            return "insert_failed_z_error"
        if np.linalg.norm(final_peg[:2] - initial_peg[:2]) < 0.050:
            return "peg_not_moved"
    return ""


def diagnostics(trace: List[Dict[str, Any]], initial_peg: np.ndarray, block: np.ndarray) -> Dict[str, Any]:
    close = next((item for item in reversed(trace) if item["stage"] == "CLOSE_GRIPPER_AND_WAIT"), None)
    lift = next((item for item in reversed(trace) if item["stage"] == "LIFT_PEG"), None)
    before_insert = next((item for item in reversed(trace) if item["stage"] == "LOWER_INSERT"), None)
    transport = [item for item in trace if item["stage"] in TRANSPORT_STAGES]
    final = trace[-1] if trace else {}
    eef_close = np.asarray(close["eef_position"], dtype=np.float32) if close else None
    peg_close = np.asarray(close["peg_position"], dtype=np.float32) if close else None
    final_peg = np.asarray(final.get("peg_position", initial_peg), dtype=np.float32)
    h = hole_pos(block)
    after_lift_peg = np.asarray(lift["peg_position"], dtype=np.float32) if lift else initial_peg
    transport_peg_z = np.asarray([item["peg_position"][2] for item in transport], dtype=np.float32) if transport else np.asarray([], dtype=np.float32)
    transport_eef_dist = np.asarray([item["peg_eef_distance"] for item in transport], dtype=np.float32) if transport else np.asarray([], dtype=np.float32)
    contact_flags = [bool(item.get("gripper_contacted_peg", False)) for item in transport]
    return {
        "peg_asset": PEG_ASSET,
        "peg_size_m": PEG_SIZE_M,
        "peg_initial_pose": initial_peg.astype(float).round(6).tolist(),
        "block_initial_pose": block.astype(float).round(6).tolist(),
        "hole_initial_pose": h.astype(float).round(6).tolist(),
        "peg_hole_initial_xy_distance": float(np.linalg.norm(initial_peg[:2] - h[:2])),
        "transport_distance": float(np.linalg.norm(initial_peg[:2] - h[:2])),
        "eef_pose_at_grasp": None if close is None else close["eef_position"],
        "xy_error_at_close": None if eef_close is None or peg_close is None else float(np.linalg.norm(eef_close[:2] - peg_close[:2])),
        "z_error_at_close": None if eef_close is None or peg_close is None else float(eef_close[2] - peg_close[2]),
        "peg_pose_after_close": None if close is None else close["peg_position"],
        "peg_pose_after_lift": None if lift is None else lift["peg_position"],
        "peg_z_increase": None if lift is None else float(np.asarray(lift["peg_position"])[2] - initial_peg[2]),
        "peg_pose_before_insertion": None if before_insert is None else before_insert["peg_position"],
        "eef_pose_before_insertion": None if before_insert is None else before_insert["eef_position"],
        "peg_eef_distance_during_transport": transport_eef_dist.astype(float).round(6).tolist(),
        "peg_eef_distance_transport_max": None if transport_eef_dist.size == 0 else float(transport_eef_dist.max()),
        "peg_z_drop_during_transport": None if transport_peg_z.size == 0 else float(after_lift_peg[2] - transport_peg_z.min()),
        "peg_quat_before_insertion": None if before_insert is None else before_insert.get("peg_quat"),
        "gripper_contact_maintained_during_transport": bool(contact_flags and all(contact_flags)),
        "gripper_contact_fraction_during_transport": 0.0 if not contact_flags else float(sum(contact_flags) / len(contact_flags)),
        "peg_eef_distance_during_lift": [item["peg_eef_distance"] for item in trace if item["stage"] == "LIFT_PEG"],
        "peg_final_pose": final_peg.astype(float).round(6).tolist(),
        "peg_hole_xy_error": float(np.linalg.norm(final_peg[:2] - h[:2])),
        "insertion_depth_final_z_error": float(final_peg[2] - (h[2] + 0.034)),
        "gripper_opened_before_success": "OPEN_GRIPPER" in unique_stages(trace),
    }


def run_seed(env, mode: str, seed: int, cfg: PrimitiveConfig, randomization_level: str) -> Dict[str, Any]:
    env.seed(seed)
    obs = env.reset()
    env.env.sim.forward()
    obs, reset_info = apply_peg_insertion_reset_randomization(env, obs, seed, cfg.reset_settle_steps, randomization_level)
    initial_peg = object_pos(env, obs, PEG).copy()
    initial_block = object_pos(env, obs, BLOCK).copy()
    object_joints = []
    for name in [PEG, BLOCK]:
        object_joints.extend(env.env.get_object(name).joints)
    trace: List[Dict[str, Any]] = []
    with JointWriteGuard(env, object_joints) as guard:
        if mode == "pick_lift_only":
            obs, step_idx, done = run_pick_lift(env, obs, cfg, trace, 0)
        elif mode in {"insert_only", "full"}:
            obs, step_idx, done = run_insert(env, obs, cfg, trace, 0)
        else:
            raise ValueError(mode)
    final_peg = object_pos(env, obs, PEG).copy()
    final_block = object_pos(env, obs, BLOCK).copy()
    failure = classify_failure(mode, trace, initial_peg, final_peg, final_block, guard.count)
    diag = diagnostics(trace, initial_peg, initial_block)
    lift_pose = last_pose(trace, "LIFT_PEG")
    close_pose = last_pose(trace, "CLOSE_GRIPPER_AND_WAIT")
    if lift_pose is None:
        lift_pose = final_peg
    if close_pose is None:
        close_pose = initial_peg
    return {
        "seed": seed,
        "mode": mode,
        "passed": not failure,
        "failure_reason": failure,
        "stage_timeline": unique_stages(trace),
        "peg_initial_pose": initial_peg.astype(float).round(6).tolist(),
        "block_initial_pose": initial_block.astype(float).round(6).tolist(),
        "peg_final_pose": final_peg.astype(float).round(6).tolist(),
        "block_final_pose": final_block.astype(float).round(6).tolist(),
        "peg_lift": float(lift_pose[2] - close_pose[2]),
        "transport_distance": diag["transport_distance"],
        "peg_z_drop_during_transport": diag["peg_z_drop_during_transport"],
        "peg_eef_distance_transport_max": diag["peg_eef_distance_transport_max"],
        "gripper_contact_fraction_during_transport": diag["gripper_contact_fraction_during_transport"],
        "peg_hole_xy_error": float(np.linalg.norm(final_peg[:2] - final_block[:2])),
        "oracle_helper_used": False,
        "direct_object_pose_writes_during_rollout": int(guard.count),
        "reset_info": reset_info,
        "diagnostics": diag,
        "trace": trace,
    }


def print_running_summary(per_seed: List[Dict[str, Any]]) -> None:
    pass_count = sum(1 for item in per_seed if item["passed"])
    fail_count = len(per_seed) - pass_count
    failures = Counter(item["failure_reason"] or "passed" for item in per_seed if not item["passed"])
    common = failures.most_common(1)[0][0] if failures else "none"
    print(f"running: attempts={len(per_seed)} pass={pass_count} fail={fail_count} success_rate={pass_count / len(per_seed):.3f} most_common_failure={common}", flush=True)


def _rate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {"n": 0, "success_rate": None}
    passed = sum(1 for item in items if item["passed"])
    return {"n": len(items), "pass": passed, "fail": len(items) - passed, "success_rate": passed / len(items)}


def build_failure_analysis(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    distances = np.asarray([item.get("transport_distance", 0.0) for item in per_seed], dtype=np.float32)
    failures = Counter(item["failure_reason"] or "passed" for item in per_seed if not item["passed"])
    bins = [(0.00, 0.10), (0.10, 0.14), (0.14, 0.18), (0.18, 0.22), (0.22, 999.0)]
    by_distance = {}
    for lo, hi in bins:
        label = f"{lo:.2f}-{hi:.2f}" if hi < 999 else f">={lo:.2f}"
        by_distance[label] = _rate([item for item in per_seed if lo <= item.get("transport_distance", 0.0) < hi])

    peg_xy = np.asarray([item["peg_initial_pose"][:2] for item in per_seed], dtype=np.float32)
    block_xy = np.asarray([item["block_initial_pose"][:2] for item in per_seed], dtype=np.float32)
    failed = [item for item in per_seed if not item["passed"]]
    failed_peg_xy = np.asarray([item["peg_initial_pose"][:2] for item in failed], dtype=np.float32) if failed else np.zeros((0, 2), dtype=np.float32)
    failed_block_xy = np.asarray([item["block_initial_pose"][:2] for item in failed], dtype=np.float32) if failed else np.zeros((0, 2), dtype=np.float32)

    def ranges(arr: np.ndarray) -> Dict[str, Any]:
        if len(arr) == 0:
            return {"n": 0}
        return {
            "n": int(len(arr)),
            "x_min": float(arr[:, 0].min()),
            "x_max": float(arr[:, 0].max()),
            "y_min": float(arr[:, 1].min()),
            "y_max": float(arr[:, 1].max()),
            "xy_mean": arr.mean(axis=0).astype(float).round(5).tolist(),
        }

    passed_distances = distances[[item["passed"] for item in per_seed]] if len(distances) else np.asarray([], dtype=np.float32)
    recommended_max = 0.22
    if failed:
        failed_distances = np.asarray([item["transport_distance"] for item in failed], dtype=np.float32)
        if len(passed_distances):
            recommended_max = float(min(0.22, max(0.12, np.percentile(passed_distances, 80))))
        if len(failed_distances) and float(failed_distances.min()) < recommended_max:
            recommended_max = float(max(0.12, failed_distances.min() - 0.01))
    analysis = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n": len(per_seed),
        "overall": _rate(per_seed),
        "most_common_failure_reason": failures.most_common(1)[0][0] if failures else "none",
        "failure_reasons": dict(failures),
        "success_rate_by_transport_distance": by_distance,
        "success_rate_vs_peg_xy": {
            "all": ranges(peg_xy),
            "failed": ranges(failed_peg_xy),
        },
        "success_rate_vs_block_xy": {
            "all": ranges(block_xy),
            "failed": ranges(failed_block_xy),
        },
        "failures_cluster": bool(len(failed) > 0),
        "failed_seeds": [
            {
                "seed": item["seed"],
                "reason": item["failure_reason"],
                "transport_distance": round(float(item.get("transport_distance", 0.0)), 5),
                "peg_initial_xy": item["peg_initial_pose"][:2],
                "block_initial_xy": item["block_initial_pose"][:2],
                "final_xy_error": round(float(item.get("peg_hole_xy_error", 0.0)), 5),
                "z_drop_transport": item.get("peg_z_drop_during_transport"),
            }
            for item in failed
        ],
        "recommended_feasible_workspace": {
            "shared_x_range": [-0.150, 0.125],
            "shared_y_range": [-0.125, 0.250],
            "min_transport_distance_m": 0.06,
            "max_transport_distance_m": round(recommended_max, 3),
            "reject_region_reason": "reject overlap/too-close starts, far-right reach failures, back-left carried-insertion instability, and overly long transport distances",
        },
    }
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pick_lift_only", "insert_only", "full"], required=True)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--output-dir", default="reports/peg_insertion_primitive_debug")
    parser.add_argument("--randomization-level", default="debug_small", choices=["debug_small", "medium", "final", "diverse", "diverse_v2"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=resolve_bddl_path(custom_task="peg_insertion"),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        camera_names=["agentview"],
        horizon=1400,
        ignore_done=True,
    )
    try:
        per_seed = []
        cfg = PrimitiveConfig()
        for idx in range(args.num_seeds):
            seed = args.seed + idx
            if not args.quiet:
                print(f"[peg {args.mode} seed {idx + 1}/{args.num_seeds}] seed={seed} start", flush=True)
            result = run_seed(env, args.mode, seed, cfg, args.randomization_level)
            per_seed.append(result)
            if not args.quiet:
                status = "PASS" if result["passed"] else "FAIL"
                depth_ok = abs(float(result["diagnostics"]["insertion_depth_final_z_error"])) <= 0.055
                depth_label = "n/a" if args.mode == "pick_lift_only" else ("ok" if depth_ok else "bad")
                print(
                    f"[peg {args.mode} seed {idx + 1}/{args.num_seeds}] seed={seed} {status} "
                    f"dist={result['transport_distance']:.3f} xy_err={result['peg_hole_xy_error']:.3f} "
                    f"depth={depth_label} reason={result['failure_reason'] or 'ok'}",
                    flush=True,
                )
                print_running_summary(per_seed)
    finally:
        env.close()

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "randomization_level": args.randomization_level,
        "num_seeds": args.num_seeds,
        "passed": all(item["passed"] for item in per_seed),
        "per_seed": per_seed,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.mode}.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.mode == "insert_only" and args.randomization_level == "diverse_v2":
        analysis = build_failure_analysis(per_seed)
        Path("reports").mkdir(exist_ok=True)
        Path("reports/peg_insertion_failure_analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    console_report = {
        **{k: v for k, v in report.items() if k != "per_seed"},
        "per_seed": [
            {k: v for k, v in item.items() if k not in {"trace", "diagnostics"}}
            for item in per_seed
        ],
    }
    print(json.dumps(console_report, indent=2, sort_keys=True))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
