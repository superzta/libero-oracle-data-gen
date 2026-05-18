"""Primitive-level non-video debugging for button_box manipulation."""

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
from button_box_reset_utils import apply_button_box_reset_randomization
from qa_button_box_rollout import JointWriteGuard, inside_box, object_pos

configure_runtime_env()


BUTTON = "red_button_1"
CUBE = "blue_cube_1"
BOX = "open_box_1"
CUBE_ASSET = "blue_cube_5p5cm_cube"
CUBE_SIZE_M = [0.055, 0.055, 0.055]


@dataclass
class PrimitiveConfig:
    pos_gain: float = 20.0
    max_delta: float = 0.18
    approach_z: float = 0.18
    grasp_z: float = 0.020
    lift_z: float = 0.14
    lift_x_offset: float = 0.0
    lift_y_offset: float = 0.0
    grasp_y_offset: float = 0.0
    grasp_x_offset: float = 0.0
    box_above_z: float = 0.18
    release_z: float = 0.11
    close_steps: int = 60
    post_close_hold_steps: int = 12
    settle_steps: int = 50
    move_tol: float = 0.025
    pregrasp_max_delta: float = 0.18
    descent_max_delta: float = 0.05
    close_max_delta: float = 0.010
    lift_max_delta: float = 0.05
    lift_success_threshold: float = 0.04
    lift_continue_threshold: float = 0.06
    reset_settle_steps: int = 20


def eef_pos(obs: Dict[str, Any]) -> np.ndarray:
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float32)


def gripper_qpos(obs: Dict[str, Any]) -> np.ndarray:
    return np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float32)


def contact_pairs(env, object_name: str = CUBE) -> List[List[str]]:
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


def selected_geom_positions(env) -> Dict[str, List[float]]:
    names = [
        "blue_cube_1_g1",
        "gripper0_finger1_pad_collision",
        "gripper0_finger2_pad_collision",
        "gripper0_finger1_collision",
        "gripper0_finger2_collision",
    ]
    try:
        sim = env.env.sim
        out = {}
        for name in names:
            geom_id = sim.model.geom_name2id(name)
            out[name] = sim.data.geom_xpos[geom_id].astype(float).round(6).tolist()
        return out
    except Exception:
        return {}


def make_action(obs: Dict[str, Any], target: np.ndarray, gripper: float, cfg: PrimitiveConfig, max_delta: Optional[float] = None) -> np.ndarray:
    delta = (np.asarray(target, dtype=np.float32) - eef_pos(obs)) * cfg.pos_gain
    clip = cfg.max_delta if max_delta is None else max_delta
    delta = np.clip(delta, -clip, clip)
    return np.concatenate([delta, np.zeros(3, dtype=np.float32), np.asarray([gripper], dtype=np.float32)])


def wait_action(gripper: float) -> np.ndarray:
    return np.asarray([0, 0, 0, 0, 0, 0, gripper], dtype=np.float32)


def step_env(env, obs, action, trace: List[Dict[str, Any]], stage: str, target: Optional[np.ndarray], step_idx: int):
    obs, reward, done, info = env.step(action)
    done = False
    cube = object_pos(env, obs, CUBE)
    box = object_pos(env, obs, BOX)
    eef = eef_pos(obs)
    trace.append(
        {
            "step": step_idx,
            "stage": stage,
            "eef_position": eef.astype(float).round(6).tolist(),
            "cube_position": cube.astype(float).round(6).tolist(),
            "box_position": box.astype(float).round(6).tolist(),
            "target_position": None if target is None else np.asarray(target).astype(float).round(6).tolist(),
            "gripper_command": float(action[-1]),
            "gripper_qpos": gripper_qpos(obs).astype(float).round(6).tolist(),
            "contact_pairs": contact_pairs(env),
            "geom_positions": selected_geom_positions(env) if stage in {"CLOSE_GRIPPER_AND_WAIT", "PRE_CLOSE_PAUSE"} else {},
            "cube_eef_distance": float(np.linalg.norm(cube - eef)),
            "cube_eef_xy_distance": float(np.linalg.norm(cube[:2] - eef[:2])),
            "reward": float(reward),
            "done": bool(done),
        }
    )
    return obs, done, info


def move_to(env, obs, target, gripper, cfg, trace, stage, step_idx, max_steps=80, max_delta=None):
    done = False
    for _ in range(max_steps):
        action = make_action(obs, target, gripper, cfg, max_delta=max_delta)
        obs, done, info = step_env(env, obs, action, trace, stage, target, step_idx)
        step_idx += 1
        if np.linalg.norm(eef_pos(obs) - target) <= cfg.move_tol or done:
            break
    return obs, step_idx, done


def hold(env, obs, n_steps, gripper, trace, stage, step_idx):
    done = False
    for _ in range(n_steps):
        obs, done, info = step_env(env, obs, wait_action(gripper), trace, stage, None, step_idx)
        step_idx += 1
        if done:
            break
    return obs, step_idx, done


def run_pick_lift(env, obs, cfg: PrimitiveConfig, trace: List[Dict[str, Any]], step_idx: int):
    cube0 = object_pos(env, obs, CUBE).copy()
    xy_offset = np.array([cfg.grasp_x_offset, cfg.grasp_y_offset, 0.0], dtype=np.float32)
    above = cube0 + xy_offset + np.array([0.0, 0.0, cfg.approach_z], dtype=np.float32)
    obs, step_idx, done = move_to(env, obs, above, -1.0, cfg, trace, "MOVE_ABOVE_CUBE", step_idx, max_steps=180, max_delta=cfg.pregrasp_max_delta)
    if done:
        return obs, step_idx, done
    cube_before_descend = object_pos(env, obs, CUBE).copy()
    grasp = cube_before_descend + xy_offset + np.array([0.0, 0.0, cfg.grasp_z], dtype=np.float32)
    obs, step_idx, done = move_to(env, obs, grasp, -1.0, cfg, trace, "DESCEND_TO_CUBE", step_idx, max_steps=320, max_delta=cfg.descent_max_delta)
    if done:
        return obs, step_idx, done
    for _ in range(80):
        if eef_pos(obs)[2] - object_pos(env, obs, CUBE)[2] <= cfg.grasp_z + 0.005:
            break
        action = make_action(obs, grasp, -1.0, cfg, max_delta=cfg.descent_max_delta)
        obs, done, info = step_env(env, obs, action, trace, "DESCEND_TO_CUBE_CONTACT_SEEK", grasp, step_idx)
        step_idx += 1
        if done:
            return obs, step_idx, done
    obs, step_idx, done = hold(env, obs, 6, -1.0, trace, "PRE_CLOSE_PAUSE", step_idx)
    if done:
        return obs, step_idx, done
    obs, step_idx, done = move_to(env, obs, grasp, 1.0, cfg, trace, "CLOSE_GRIPPER_AND_WAIT", step_idx, max_steps=cfg.close_steps, max_delta=cfg.close_max_delta)
    if done:
        return obs, step_idx, done
    obs, step_idx, done = hold(env, obs, cfg.post_close_hold_steps, 1.0, trace, "POST_CLOSE_HOLD", step_idx)
    if done:
        return obs, step_idx, done
    cube_before_lift = object_pos(env, obs, CUBE).copy()
    lift = cube_before_lift + xy_offset + np.array([cfg.lift_x_offset, cfg.lift_y_offset, cfg.lift_z], dtype=np.float32)
    done = False
    for _ in range(120):
        action = make_action(obs, lift, 1.0, cfg, max_delta=cfg.lift_max_delta)
        obs, done, info = step_env(env, obs, action, trace, "LIFT_CUBE", lift, step_idx)
        step_idx += 1
        if object_pos(env, obs, CUBE)[2] - cube0[2] >= cfg.lift_continue_threshold or done:
            break
    return obs, step_idx, done


def run_place(env, obs, cfg: PrimitiveConfig, trace: List[Dict[str, Any]], step_idx: int):
    obs, step_idx, done = run_pick_lift(env, obs, cfg, trace, step_idx)
    if done:
        return obs, step_idx, done
    box = object_pos(env, obs, BOX).copy()
    above_box = box + np.array([0.0, 0.0, cfg.box_above_z], dtype=np.float32)
    release = box + np.array([0.0, -0.005, cfg.release_z], dtype=np.float32)
    obs, step_idx, done = move_to(env, obs, above_box, 1.0, cfg, trace, "MOVE_ABOVE_BOX", step_idx, max_steps=320, max_delta=0.035)
    if done:
        return obs, step_idx, done
    obs, step_idx, done = move_to(env, obs, release, 1.0, cfg, trace, "LOWER_TO_BOX", step_idx, max_steps=220, max_delta=0.02)
    if done:
        return obs, step_idx, done
    obs, step_idx, done = hold(env, obs, 18, -1.0, trace, "OPEN_GRIPPER", step_idx)
    if done:
        return obs, step_idx, done
    obs, step_idx, done = hold(env, obs, cfg.settle_steps, -1.0, trace, "WAIT_SETTLE", step_idx)
    return obs, step_idx, done


def unique_stages(trace: List[Dict[str, Any]]) -> List[str]:
    stages: List[str] = []
    for item in trace:
        stage = item["stage"]
        if not stages or stages[-1] != stage:
            stages.append(stage)
    return stages


def first_pose(trace: List[Dict[str, Any]], stage: str) -> Optional[np.ndarray]:
    for item in trace:
        if item["stage"] == stage:
            return np.asarray(item["cube_position"], dtype=np.float32)
    return None


def last_pose(trace: List[Dict[str, Any]], stage: str) -> Optional[np.ndarray]:
    for item in reversed(trace):
        if item["stage"] == stage:
            return np.asarray(item["cube_position"], dtype=np.float32)
    return None


def cube_follows_eef(trace: List[Dict[str, Any]], lift_baseline: np.ndarray) -> bool:
    lift = [item for item in trace if item["stage"] == "LIFT_CUBE"]
    if len(lift) < 5:
        return False
    cube_z = np.asarray([item["cube_position"][2] for item in lift], dtype=np.float32)
    eef_xy_dist = np.asarray([item["cube_eef_xy_distance"] for item in lift], dtype=np.float32)
    return bool(float(cube_z.max() - lift_baseline[2]) >= 0.04 and float(np.median(eef_xy_dist[-10:])) <= 0.055)


def last_item(trace: List[Dict[str, Any]], stage: str) -> Optional[Dict[str, Any]]:
    for item in reversed(trace):
        if item["stage"] == stage:
            return item
    return None


def first_item(trace: List[Dict[str, Any]], stage: str) -> Optional[Dict[str, Any]]:
    for item in trace:
        if item["stage"] == stage:
            return item
    return None


def pick_lift_diagnostics(trace: List[Dict[str, Any]], initial_cube: np.ndarray) -> Dict[str, Any]:
    eef_before_descent = last_item(trace, "MOVE_ABOVE_CUBE")
    descend_first = first_item(trace, "DESCEND_TO_CUBE")
    descend_last = last_item(trace, "DESCEND_TO_CUBE")
    close_last = last_item(trace, "CLOSE_GRIPPER_AND_WAIT")
    hold_last = last_item(trace, "POST_CLOSE_HOLD")
    lift_items = [item for item in trace if item["stage"] == "LIFT_CUBE"]
    lift_last = lift_items[-1] if lift_items else None

    if close_last is not None:
        eef_close = np.asarray(close_last["eef_position"], dtype=np.float32)
        cube_close = np.asarray(close_last["cube_position"], dtype=np.float32)
        xy_error = float(np.linalg.norm(eef_close[:2] - cube_close[:2]))
        z_error = float(eef_close[2] - cube_close[2])
        geom_positions_at_close = close_last.get("geom_positions", {})
    else:
        xy_error = None
        z_error = None
        geom_positions_at_close = {}

    if descend_first is not None and descend_last is not None:
        cube_desc_start = np.asarray(descend_first["cube_position"], dtype=np.float32)
        cube_desc_end = np.asarray(descend_last["cube_position"], dtype=np.float32)
        cube_xy_shift = float(np.linalg.norm(cube_desc_end[:2] - cube_desc_start[:2]))
    else:
        cube_xy_shift = None

    lift_distances = [item["cube_eef_distance"] for item in lift_items]
    close_contacts = [
        pair
        for item in trace
        if item["stage"] in {"CLOSE_GRIPPER_AND_WAIT", "POST_CLOSE_HOLD"}
        for pair in item.get("contact_pairs", [])
    ]
    gripper_cube_contacts = [
        pair for pair in close_contacts if any(CUBE in str(part) for part in pair) and any(("gripper" in str(part) or "finger" in str(part)) for part in pair)
    ]
    after_lift = np.asarray(lift_last["cube_position"], dtype=np.float32) if lift_last is not None else None
    lift_baseline_item = hold_last if hold_last is not None else close_last
    lift_baseline = (
        np.asarray(lift_baseline_item["cube_position"], dtype=np.float32)
        if lift_baseline_item is not None
        else initial_cube
    )
    return {
        "cube_asset": CUBE_ASSET,
        "cube_size_m": CUBE_SIZE_M,
        "cube_init_pos": initial_cube.astype(float).round(6).tolist(),
        "cube_pos_before_grasp": None if descend_last is None else descend_last["cube_position"],
        "eef_pos_before_descent": None if eef_before_descent is None else eef_before_descent["eef_position"],
        "eef_pos_at_gripper_close": None if close_last is None else close_last["eef_position"],
        "cube_pos_at_gripper_close": None if close_last is None else close_last["cube_position"],
        "xy_error_at_close": xy_error,
        "z_error_at_close": z_error,
        "cube_pos_after_close_hold": None if hold_last is None else hold_last["cube_position"],
        "cube_pos_after_lift": None if lift_last is None else lift_last["cube_position"],
        "cube_z_increase": None if after_lift is None else float(after_lift[2] - lift_baseline[2]),
        "cube_z_increase_from_initial": None if after_lift is None else float(after_lift[2] - initial_cube[2]),
        "cube_xy_shift_during_descent": cube_xy_shift,
        "contact_pairs_during_close": close_contacts,
        "gripper_contacted_cube": bool(gripper_cube_contacts),
        "geom_positions_at_close": geom_positions_at_close,
        "cube_to_eef_distance_during_lift": lift_distances,
    }


def classify_failure(mode: str, trace: List[Dict[str, Any]], initial_cube: np.ndarray, box: np.ndarray, direct_writes: int, helper_used: bool) -> str:
    if direct_writes or helper_used:
        return "oracle_or_direct_pose_write_used"
    stages = unique_stages(trace)
    final_cube = np.asarray(trace[-1]["cube_position"], dtype=np.float32) if trace else initial_cube
    after_lift = last_pose(trace, "LIFT_CUBE")
    lift_baseline = last_pose(trace, "POST_CLOSE_HOLD")
    if lift_baseline is None:
        lift_baseline = last_pose(trace, "CLOSE_GRIPPER_AND_WAIT")
    if lift_baseline is None:
        lift_baseline = initial_cube
    z_inc = float((after_lift if after_lift is not None else final_cube)[2] - lift_baseline[2])
    if mode in {"pick_lift_only", "place_only", "full"}:
        if "LIFT_CUBE" not in stages or z_inc < 0.04:
            return "grasp_failed_cube_not_lifted"
        if not cube_follows_eef(trace, lift_baseline):
            return "grasp_slip_during_lift"
    if mode in {"place_only", "full"}:
        if "WAIT_SETTLE" not in stages:
            return "success_before_settle_invalid"
        if not inside_box(final_cube, box):
            return "place_failed_not_inside_box"
    return ""


def run_seed(env, mode: str, seed: int, cfg: PrimitiveConfig, randomization_level: str = "debug_small") -> Dict[str, Any]:
    env.seed(seed)
    obs = env.reset()
    try:
        env.env.sim.forward()
    except Exception:
        pass
    obs, _reset_info = apply_button_box_reset_randomization(env, obs, seed, settle_steps=cfg.reset_settle_steps, randomization_level=randomization_level)
    initial_cube = object_pos(env, obs, CUBE).copy()
    initial_box = object_pos(env, obs, BOX).copy()
    trace: List[Dict[str, Any]] = []
    object_joints = []
    for name in [BUTTON, CUBE, BOX]:
        try:
            object_joints.extend(env.env.get_object(name).joints)
        except Exception:
            pass

    step_idx = 0
    with JointWriteGuard(env, object_joints) as guard:
        if mode == "press_only":
            button = object_pos(env, obs, BUTTON)
            obs, step_idx, done = move_to(env, obs, button + [0.0, 0.0, 0.135], -1.0, cfg, trace, "MOVE_ABOVE_BUTTON", step_idx)
            obs, step_idx, done = move_to(env, obs, button + [0.0, 0.0, 0.033], -1.0, cfg, trace, "PRESS_BUTTON", step_idx, max_delta=0.02)
            obs, step_idx, done = hold(env, obs, 8, -1.0, trace, "HOLD_BUTTON_PRESS", step_idx)
        elif mode == "pick_lift_only":
            obs, step_idx, done = run_pick_lift(env, obs, cfg, trace, step_idx)
        elif mode == "place_only":
            obs, step_idx, done = run_place(env, obs, cfg, trace, step_idx)
        elif mode == "full":
            button = object_pos(env, obs, BUTTON)
            obs, step_idx, done = move_to(env, obs, button + [0.0, 0.0, 0.135], -1.0, cfg, trace, "MOVE_ABOVE_BUTTON", step_idx)
            obs, step_idx, done = move_to(env, obs, button + [0.0, 0.0, 0.033], -1.0, cfg, trace, "PRESS_BUTTON", step_idx, max_delta=0.02)
            obs, step_idx, done = hold(env, obs, 8, -1.0, trace, "HOLD_BUTTON_PRESS", step_idx)
            obs, step_idx, done = move_to(env, obs, button + [0.0, 0.0, 0.135], -1.0, cfg, trace, "RETRACT_FROM_BUTTON", step_idx)
            obs, step_idx, done = run_place(env, obs, cfg, trace, step_idx)
        else:
            raise ValueError(mode)

    final_cube = object_pos(env, obs, CUBE).copy()
    final_box = object_pos(env, obs, BOX).copy()
    before_grasp = first_pose(trace, "DESCEND_TO_CUBE")
    after_close = last_pose(trace, "CLOSE_GRIPPER_AND_WAIT")
    after_close_hold = last_pose(trace, "POST_CLOSE_HOLD")
    after_lift = last_pose(trace, "LIFT_CUBE")
    lift_baseline = after_close_hold if after_close_hold is not None else (after_close if after_close is not None else initial_cube)
    failure = classify_failure(mode, trace, initial_cube, final_box, guard.count, False)
    passed = not failure
    if mode == "pick_lift_only":
        passed = passed and float((after_lift if after_lift is not None else final_cube)[2] - lift_baseline[2]) >= 0.04
    if mode == "press_only":
        button = object_pos(env, obs, BUTTON)
        press_hits = [
            np.linalg.norm(np.asarray(item["eef_position"][:2], dtype=np.float32) - button[:2]) <= 0.032
            and item["eef_position"][2] <= button[2] + 0.045
            for item in trace
        ]
        passed = bool(any(press_hits) and guard.count == 0)

    return {
        "seed": seed,
        "mode": mode,
        "passed": bool(passed),
        "failure_reason": "" if passed else failure or "primitive_acceptance_failed",
        "stage_timeline": unique_stages(trace),
        "eef_pose_over_time": [item["eef_position"] for item in trace],
        "gripper_command_over_time": [item["gripper_command"] for item in trace],
        "cube_initial_pose": initial_cube.astype(float).round(6).tolist(),
        "cube_asset": CUBE_ASSET,
        "cube_size_m": CUBE_SIZE_M,
        "cube_pose_before_grasp": None if before_grasp is None else before_grasp.astype(float).round(6).tolist(),
        "cube_pose_after_gripper_close": None if after_close is None else after_close.astype(float).round(6).tolist(),
        "cube_pose_after_lift": None if after_lift is None else after_lift.astype(float).round(6).tolist(),
        "cube_final_pose": final_cube.astype(float).round(6).tolist(),
        "cube_z_increase": float(final_cube[2] - lift_baseline[2]),
        "cube_z_increase_from_initial": float(final_cube[2] - initial_cube[2]),
        "cube_z_increase_after_lift": None if after_lift is None else float(after_lift[2] - lift_baseline[2]),
        "cube_eef_distance_after_grasp": None
        if after_close is None
        else next((item["cube_eef_distance"] for item in reversed(trace) if item["stage"] == "CLOSE_GRIPPER_AND_WAIT"), None),
        "cube_follows_eef_during_lift": cube_follows_eef(trace, lift_baseline),
        "pick_lift_diagnostics": pick_lift_diagnostics(trace, initial_cube),
        "box_pose": final_box.astype(float).round(6).tolist(),
        "inside_box_result": inside_box(final_cube, final_box),
        "oracle_helper_used": False,
        "direct_object_pose_writes_during_rollout": int(guard.count),
        "trace": trace,
    }


def print_running_summary(per_seed: List[Dict[str, Any]]) -> None:
    completed = len(per_seed)
    pass_count = sum(1 for item in per_seed if item["passed"])
    fail_count = completed - pass_count
    failures = Counter(item["failure_reason"] or "passed" for item in per_seed if not item["passed"])
    common = failures.most_common(1)[0][0] if failures else "none"
    rate = pass_count / completed if completed else 0.0
    print(
        f"running: completed={completed} pass={pass_count} fail={fail_count} "
        f"success_rate={rate:.3f} most_common_failure={common}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["press_only", "pick_lift_only", "place_only", "full"], required=True)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--output-dir", default="reports/button_box_primitive_debug")
    parser.add_argument("--grasp-z-offset", type=float, default=None)
    parser.add_argument("--pregrasp-z-offset", type=float, default=None)
    parser.add_argument("--close-hold-steps", type=int, default=None)
    parser.add_argument("--descent-max-delta", type=float, default=None)
    parser.add_argument("--lift-max-delta", type=float, default=None)
    parser.add_argument("--grasp-x-offset", type=float, default=None)
    parser.add_argument("--grasp-y-offset", type=float, default=None)
    parser.add_argument(
        "--randomization-level",
        default="debug_small",
        choices=["debug_small", "medium", "final", "diverse", "diverse_v2"],
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    cfg = PrimitiveConfig()
    if args.grasp_z_offset is not None:
        cfg.grasp_z = args.grasp_z_offset
    if args.pregrasp_z_offset is not None:
        cfg.approach_z = args.pregrasp_z_offset
    if args.close_hold_steps is not None:
        cfg.close_steps = args.close_hold_steps
    if args.descent_max_delta is not None:
        cfg.descent_max_delta = args.descent_max_delta
    if args.lift_max_delta is not None:
        cfg.lift_max_delta = args.lift_max_delta
    if args.grasp_x_offset is not None:
        cfg.grasp_x_offset = args.grasp_x_offset
    if args.grasp_y_offset is not None:
        cfg.grasp_y_offset = args.grasp_y_offset
    env = OffScreenRenderEnv(
        bddl_file_name=resolve_bddl_path(custom_task="button_box"),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        camera_names=["agentview"],
        horizon=1400,
        ignore_done=True,
    )
    try:
        per_seed = []
        for idx in range(args.num_seeds):
            seed = args.seed + idx
            if not args.quiet:
                print(f"[primitive {args.mode} seed {idx + 1}/{args.num_seeds}] seed={seed} start", flush=True)
            result = run_seed(env, args.mode, seed, cfg, randomization_level=args.randomization_level)
            per_seed.append(result)
            if not args.quiet:
                status = "PASS" if result["passed"] else "FAIL"
                print(
                    "  seed={seed} {status} reason={reason} cube_z_lift={lift:.4f}".format(
                        seed=result["seed"],
                        status=status,
                        reason=result["failure_reason"] or "ok",
                        lift=float(result.get("cube_z_increase_after_lift") or 0.0),
                    ),
                    flush=True,
                )
                if (idx + 1) % 3 == 0 or idx + 1 == args.num_seeds:
                    print_running_summary(per_seed)
    finally:
        env.close()

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "num_seeds": args.num_seeds,
        "passed": all(item["passed"] for item in per_seed),
        "per_seed": per_seed,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.mode}.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.mode == "pick_lift_only":
        (out_dir / "pick_lift_diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    console_report = {
        **{k: v for k, v in report.items() if k != "per_seed"},
        "per_seed": [{k: v for k, v in item.items() if k not in {"trace", "eef_pose_over_time", "gripper_command_over_time"}} for item in per_seed],
    }
    print(json.dumps(console_report, indent=2, sort_keys=True))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
