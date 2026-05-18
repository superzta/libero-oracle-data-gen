"""Strict non-video QA for the button_box oracle rollout."""

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

from controllers.button_box_controller import ButtonBoxController
from button_box_reset_utils import apply_button_box_reset_randomization
from libero_env_utils import configure_runtime_env, resolve_bddl_path

configure_runtime_env()


def object_pos(env, obs: Dict[str, Any], name: str) -> np.ndarray:
    key = f"{name}_pos"
    if key in obs:
        return np.asarray(obs[key], dtype=np.float32)
    return np.asarray(env.env.sim.data.body_xpos[env.env.obj_body_id[name]], dtype=np.float32)


def inside_box(cube: np.ndarray, box: np.ndarray) -> bool:
    return bool(np.linalg.norm(cube[:2] - box[:2]) <= 0.095 and box[2] - 0.025 <= cube[2] <= box[2] + 0.17)


def primitive_failure_code(
    stages: List[str],
    button_pressed: bool,
    cube_z_increase: float,
    cube_final_inside_box: bool,
    release_step,
    success_step,
    settle_frames: int,
    cube_initial_xy_std: float = 1.0,
) -> str:
    if cube_initial_xy_std < 0.01:
        return "randomization_not_applied"
    if not button_pressed or "HOLD_BUTTON_PRESS" not in stages:
        return "button_press_failed"
    if "LIFT_CUBE" in stages and cube_z_increase < 0.030:
        return "grasp_failed_cube_not_lifted"
    if "MOVE_ABOVE_BOX" in stages and cube_z_increase >= 0.04 and "OPEN_GRIPPER" not in stages:
        return "grasp_slip_during_lift"
    if release_step is not None and success_step is not None and success_step <= release_step:
        return "success_before_settle_invalid"
    if release_step is not None and settle_frames < 30:
        return "success_before_settle_invalid"
    if "VERIFY_FINAL_STATE" in stages and not cube_final_inside_box:
        return "place_failed_not_inside_box"
    return ""


class JointWriteGuard:
    def __init__(self, env, joint_names: List[str]):
        self.env = env
        self.joint_names = set(joint_names)
        self.count = 0
        self.records: List[str] = []
        self.original = env.env.sim.data.set_joint_qpos

    def __enter__(self):
        def guarded(name, qpos):
            if name in self.joint_names:
                self.count += 1
                self.records.append(str(name))
            return self.original(name, qpos)

        self.env.env.sim.data.set_joint_qpos = guarded
        return self

    def __exit__(self, exc_type, exc, tb):
        self.env.env.sim.data.set_joint_qpos = self.original


def run_seed(env, seed: int, horizon: int, allow_helper: bool, randomization_level: str = "debug_small") -> Dict[str, Any]:
    env.seed(seed)
    obs = env.reset()
    try:
        env.env.sim.forward()
    except Exception:
        pass
    obs, _reset_info = apply_button_box_reset_randomization(env, obs, seed, settle_steps=20, randomization_level=randomization_level)
    controller = ButtonBoxController()
    controller.reset(env, obs, {"allow_oracle_state_helper": allow_helper})
    object_joints = []
    for name in [controller.button_name, controller.cube_name, controller.box_name]:
        try:
            object_joints.extend(env.env.get_object(name).joints)
        except Exception:
            pass

    initial_button = object_pos(env, obs, controller.button_name).copy()
    initial_cube = object_pos(env, obs, controller.cube_name).copy()
    initial_box = object_pos(env, obs, controller.box_name).copy()
    timeline = []
    success = False
    done = False
    release_step = None

    with JointWriteGuard(env, object_joints) as guard:
        for step in range(horizon):
            action = controller.act(obs)
            obs, reward, done, info = env.step(action)
            setattr(controller, "_last_debug_obs", obs)
            debug = controller.get_debug_state()
            debug["step"] = step
            timeline.append(debug)
            if debug["stage"] == "OPEN_GRIPPER" and release_step is None:
                release_step = step
            success = controller.is_success(obs, info, env)
            if success:
                break

    final_button = object_pos(env, obs, controller.button_name).copy()
    final_cube = object_pos(env, obs, controller.cube_name).copy()
    final_box = object_pos(env, obs, controller.box_name).copy()
    stages = [item["stage"] for item in timeline]
    unique_stages = []
    for stage in stages:
        if not unique_stages or unique_stages[-1] != stage:
            unique_stages.append(stage)

    button_xy_drift = float(np.linalg.norm(final_button[:2] - initial_button[:2]))
    cube_movement = float(np.linalg.norm(final_cube - initial_cube))
    cube_xy_movement = float(np.linalg.norm(final_cube[:2] - initial_cube[:2]))
    cube_z_increase = float(final_cube[2] - initial_cube[2])
    lift_z_values = [
        float(item["cube_position"][2])
        for item in timeline
        if item.get("stage") in {"LIFT_CUBE", "MOVE_ABOVE_BOX", "LOWER_TO_BOX"} and "cube_position" in item
    ]
    cube_lift_z_increase = float((max(lift_z_values) if lift_z_values else final_cube[2]) - initial_cube[2])
    final_inside = inside_box(final_cube, final_box)
    settle_frames = int(getattr(controller, "settle_steps", 0))
    primitive_failure = primitive_failure_code(
        stages,
        bool(getattr(controller, "button_pressed", False)),
        cube_lift_z_increase,
        final_inside,
        release_step,
        getattr(controller, "success_step", None),
        settle_frames,
    )
    reasons = []
    if primitive_failure:
        reasons.append(primitive_failure)
    if button_xy_drift > 0.002:
        reasons.append(f"button xy drift {button_xy_drift:.5f} > 0.002")
    if cube_xy_movement < 0.018:
        reasons.append(f"cube xy movement {cube_xy_movement:.5f} < 0.018")
    if not final_inside:
        reasons.append("cube final pose is not inside box")
    if "VERIFY_FINAL_STATE" not in stages or stages[-1] not in {"VERIFY_FINAL_STATE", "DONE"}:
        reasons.append("rollout ended before VERIFY_FINAL_STATE")
    if release_step is None or getattr(controller, "success_step", None) is None:
        reasons.append("success did not occur after OPEN_GRIPPER")
    elif getattr(controller, "success_step", 0) <= release_step:
        reasons.append("success occurred before release")
    if settle_frames < 30:
        reasons.append(f"settle frames {settle_frames} < 30")
    if allow_helper or getattr(controller, "oracle_helper_used", False):
        reasons.append("oracle helper was used")
    if guard.count:
        reasons.append(f"direct object joint pose writes during rollout: {guard.count}")
    if not success:
        reasons.append("controller did not report physical success")

    return {
        "seed": seed,
        "passed": not reasons,
        "reasons": reasons,
        "success": bool(success),
        "done": bool(done),
        "total_steps": len(timeline),
        "stage_timeline": unique_stages,
        "button_initial_pose": initial_button.astype(float).round(6).tolist(),
        "button_final_pose": final_button.astype(float).round(6).tolist(),
        "cube_initial_pose": initial_cube.astype(float).round(6).tolist(),
        "cube_final_pose": final_cube.astype(float).round(6).tolist(),
        "box_pose": final_box.astype(float).round(6).tolist(),
        "button_xy_drift": button_xy_drift,
        "cube_movement_distance": cube_movement,
        "cube_xy_movement_distance": cube_xy_movement,
        "cube_z_increase": cube_z_increase,
        "cube_lift_z_increase": cube_lift_z_increase,
        "cube_final_inside_box": final_inside,
        "subprimitive_failure": primitive_failure,
        "gripper_opened_before_success": bool(release_step is not None and getattr(controller, "success_step", None) is not None and controller.success_step > release_step),
        "settle_frames_after_release": settle_frames,
        "oracle_helper_used": bool(getattr(controller, "oracle_helper_used", False)),
        "direct_object_pose_writes_during_rollout": int(guard.count),
        "final_debug": timeline[-1] if timeline else {},
        "timeline": timeline,
    }


def compute_diversity_metrics(per_seed: list) -> Dict[str, Any]:
    """Compute initial-state diversity metrics across seeds."""
    cube_initials = np.asarray([item["cube_initial_pose"][:2] for item in per_seed], dtype=np.float32)
    n = len(cube_initials)
    if n < 2:
        return {"n": n, "cube_x_range": 0.0, "cube_y_range": 0.0, "cube_xy_std": [0.0, 0.0], "bins_occupied": 0}
    x_range = float(cube_initials[:, 0].max() - cube_initials[:, 0].min())
    y_range = float(cube_initials[:, 1].max() - cube_initials[:, 1].min())
    xy_std = cube_initials.std(axis=0).astype(float).round(6).tolist()
    # Count occupied 3x3 grid bins over the sampled extent.
    bins = set()
    x_edges = np.linspace(float(cube_initials[:, 0].min()), float(cube_initials[:, 0].max()), 4)
    y_edges = np.linspace(float(cube_initials[:, 1].min()), float(cube_initials[:, 1].max()), 4)
    for xy in cube_initials:
        bx = int(np.clip(np.searchsorted(x_edges, xy[0], side="right") - 1, 0, 2))
        by = int(np.clip(np.searchsorted(y_edges, xy[1], side="right") - 1, 0, 2))
        bins.add((bx, by))
    return {
        "n": n,
        "cube_x_range": round(x_range, 5),
        "cube_y_range": round(y_range, 5),
        "cube_xy_std": xy_std,
        "bins_occupied": len(bins),
        "cube_initial_positions": cube_initials.astype(float).round(5).tolist(),
    }


def primary_failure(item: Dict[str, Any]) -> str:
    reasons = item.get("reasons") or []
    return str(reasons[0]) if reasons else "passed"


def print_running_summary(per_seed: List[Dict[str, Any]]) -> None:
    completed = len(per_seed)
    pass_count = sum(1 for item in per_seed if item["passed"])
    fail_count = completed - pass_count
    failures = Counter(primary_failure(item) for item in per_seed if not item["passed"])
    common = failures.most_common(1)[0][0] if failures else "none"
    rate = pass_count / completed if completed else 0.0
    print(
        f"running: completed={completed} pass={pass_count} fail={fail_count} "
        f"success_rate={rate:.3f} most_common_failure={common}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-oracle-state-helper", action="store_true")
    parser.add_argument("--output-dir", default="reports/button_box_rollout_qa")
    parser.add_argument(
        "--randomization-level",
        default="debug_small",
        choices=["debug_small", "medium", "final", "diverse", "diverse_v2"],
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=resolve_bddl_path(custom_task="button_box"),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        camera_names=["agentview"],
        horizon=args.horizon,
        ignore_done=True,
    )
    try:
        per_seed = []
        for idx in range(args.num_seeds):
            seed = args.seed + idx
            if not args.quiet:
                print(f"[qa seed {idx + 1}/{args.num_seeds}] seed={seed} start", flush=True)
            result = run_seed(env, seed, args.horizon, args.allow_oracle_state_helper, args.randomization_level)
            per_seed.append(result)
            if not args.quiet:
                status = "PASS" if result["passed"] else "FAIL"
                reason = primary_failure(result)
                print(
                    f"[qa seed {idx + 1}/{args.num_seeds}] seed={seed} {status} "
                    f"steps={result['total_steps']} stage={result.get('final_debug', {}).get('stage')} "
                    f"reason={reason} cube_lift={result.get('cube_lift_z_increase', 0.0):.4f}",
                    flush=True,
                )
                if (idx + 1) % 3 == 0 or idx + 1 == args.num_seeds:
                    print_running_summary(per_seed)
    finally:
        env.close()

    diversity = compute_diversity_metrics(per_seed)
    cube_initials = np.asarray([item["cube_initial_pose"][:2] for item in per_seed], dtype=np.float32)
    cube_initial_std = cube_initials.std(axis=0) if len(cube_initials) else np.zeros(2, dtype=np.float32)
    cross_seed_reasons = []
    if args.num_seeds > 1 and float(np.max(cube_initial_std)) < 0.01:
        cross_seed_reasons.append(f"cube initial xy std too low: {cube_initial_std.round(5).tolist()}")
    # Diversity gate per level
    _DIV_GATES = {
        "medium": {"min_x_range": 0.04, "min_y_range": 0.03, "min_bins": 3},
        "final":  {"min_x_range": 0.06, "min_y_range": 0.04, "min_bins": 4},
        "diverse":{"min_x_range": 0.12, "min_y_range": 0.06, "min_bins": 6},
    }
    gate = _DIV_GATES.get(args.randomization_level)
    if gate and args.num_seeds >= 4:
        if diversity["cube_x_range"] < gate["min_x_range"]:
            cross_seed_reasons.append(f"cube x_range {diversity['cube_x_range']:.4f} < {gate['min_x_range']} ({args.randomization_level} requirement)")
        if diversity["cube_y_range"] < gate["min_y_range"]:
            cross_seed_reasons.append(f"cube y_range {diversity['cube_y_range']:.4f} < {gate['min_y_range']} ({args.randomization_level} requirement)")
        if diversity["bins_occupied"] < gate["min_bins"]:
            cross_seed_reasons.append(f"bins_occupied {diversity['bins_occupied']} < {gate['min_bins']} ({args.randomization_level} requirement)")
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strict": bool(args.strict),
        "num_seeds": args.num_seeds,
        "randomization_level": args.randomization_level,
        "cube_initial_xy_std": cube_initial_std.astype(float).round(6).tolist(),
        "diversity": diversity,
        "cross_seed_reasons": cross_seed_reasons,
        "per_seed": per_seed,
    }
    report["passed"] = bool(not cross_seed_reasons and all(item["passed"] for item in per_seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "qa_button_box_rollout.json"
    md_path = out_dir / "qa_button_box_rollout.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Button Box Rollout QA", "", f"Passed: `{report['passed']}`", "", "```json", json.dumps({k: v for k, v in report.items() if k not in ("per_seed",)}, indent=2, sort_keys=True), "```"]
    for item in per_seed:
        lines.extend(["", f"## Seed {item['seed']}", "", f"Passed: `{item['passed']}`", "", "Reasons: " + (", ".join(item["reasons"]) if item["reasons"] else "none"), "", "Stages: " + " -> ".join(item["stage_timeline"])])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console_report = {
        **{k: v for k, v in report.items() if k != "per_seed"},
        "per_seed": [
            {k: v for k, v in item.items() if k != "timeline"}
            for item in per_seed
        ],
    }
    print(json.dumps(console_report, indent=2, sort_keys=True))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
