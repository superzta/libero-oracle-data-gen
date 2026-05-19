"""Strict non-video QA for the peg_insertion oracle rollout."""

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

from controllers.peg_insertion_controller import PegInsertionController
from libero_env_utils import configure_runtime_env, resolve_bddl_path
from peg_insertion_reset_utils import apply_peg_insertion_reset_randomization
from qa_button_box_rollout import JointWriteGuard

configure_runtime_env()


def object_pos(env, obs: Dict[str, Any], name: str) -> np.ndarray:
    key = f"{name}_pos"
    if key in obs:
        return np.asarray(obs[key], dtype=np.float32)
    return np.asarray(env.env.sim.data.body_xpos[env.env.obj_body_id[name]], dtype=np.float32)


def object_quat(env, obs: Dict[str, Any], name: str) -> np.ndarray:
    key = f"{name}_quat"
    if key in obs:
        return np.asarray(obs[key], dtype=np.float32)
    return np.asarray(env.env.sim.data.body_xquat[env.env.obj_body_id[name]], dtype=np.float32)


def upright_score(quat: np.ndarray) -> float:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-6:
        return 0.0
    quat = quat / norm
    _, x, y, _ = quat
    return float(1.0 - 2.0 * (x * x + y * y))


def hole_pos(block: np.ndarray) -> np.ndarray:
    return np.asarray(block, dtype=np.float32) + np.array([0.0, 0.0, 0.021], dtype=np.float32)


def primary_failure(item: Dict[str, Any]) -> str:
    reasons = item.get("reasons") or []
    return str(reasons[0]) if reasons else "passed"


def _items_for_stage(timeline: List[Dict[str, Any]], stages: set) -> List[Dict[str, Any]]:
    return [item for item in timeline if item.get("stage") in stages]


def run_seed(env, seed: int, horizon: int, allow_helper: bool, randomization_level: str, progress_interval: int = 0) -> Dict[str, Any]:
    env.seed(seed)
    obs = env.reset()
    env.env.sim.forward()
    obs, reset_info = apply_peg_insertion_reset_randomization(env, obs, seed, settle_steps=20, randomization_level=randomization_level)
    controller = PegInsertionController()
    controller.reset(env, obs, {"allow_oracle_state_helper": allow_helper})
    object_joints = []
    for name in [controller.peg_name, controller.block_name]:
        object_joints.extend(env.env.get_object(name).joints)

    initial_peg = object_pos(env, obs, controller.peg_name).copy()
    initial_block = object_pos(env, obs, controller.block_name).copy()
    initial_hole = hole_pos(initial_block)
    timeline: List[Dict[str, Any]] = []
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
            if progress_interval and step > 0 and step % progress_interval == 0:
                print(
                    f"  [peg qa seed={seed}] step={step} stage={debug.get('stage')} "
                    f"reason={debug.get('transition_reason')}",
                    flush=True,
                )
            if debug["stage"] == "OPEN_GRIPPER" and release_step is None:
                release_step = step
            success = controller.is_success(obs, info, env)
            if success:
                break

    final_peg = object_pos(env, obs, controller.peg_name).copy()
    final_peg_quat = object_quat(env, obs, controller.peg_name).copy()
    final_block = object_pos(env, obs, controller.block_name).copy()
    h = hole_pos(final_block)
    stages = [item["stage"] for item in timeline]
    unique_stages: List[str] = []
    for stage in stages:
        if not unique_stages or unique_stages[-1] != stage:
            unique_stages.append(stage)

    lift_z_values = [
        float(item["peg_position"][2])
        for item in timeline
        if item.get("stage") in {"LIFT_PEG", "MOVE_ABOVE_HOLE", "ALIGN_WITH_HOLE", "LOWER_INSERT"} and "peg_position" in item
    ]
    peg_lift_z_increase = float((max(lift_z_values) if lift_z_values else final_peg[2]) - initial_peg[2])
    lift_items = _items_for_stage(timeline, {"LIFT_PEG"})
    transport_items = _items_for_stage(timeline, {"MOVE_ABOVE_HOLE", "ALIGN_WITH_HOLE", "LOWER_INSERT"})
    peg_pose_after_lift = lift_items[-1].get("peg_position") if lift_items else None
    peg_pose_before_insertion = transport_items[-1].get("peg_position") if transport_items else None
    eef_pose_before_insertion = transport_items[-1].get("eef_position") if transport_items else None
    transport_peg_z = np.asarray([item["peg_position"][2] for item in transport_items if "peg_position" in item], dtype=np.float32)
    transport_eef_dist = np.asarray(
        [
            float(np.linalg.norm(np.asarray(item["peg_position"], dtype=np.float32) - np.asarray(item["eef_position"], dtype=np.float32)))
            for item in transport_items
            if "peg_position" in item and "eef_position" in item
        ],
        dtype=np.float32,
    )
    after_lift_z = float(peg_pose_after_lift[2]) if peg_pose_after_lift is not None else float(initial_peg[2])
    peg_z_drop_transport = None if transport_peg_z.size == 0 else float(after_lift_z - transport_peg_z.min())
    peg_xy_movement = float(np.linalg.norm(final_peg[:2] - initial_peg[:2]))
    peg_hole_xy_error = float(np.linalg.norm(final_peg[:2] - h[:2]))
    peg_final_z_error = float(final_peg[2] - (h[2] + 0.034))
    peg_upright_score = upright_score(final_peg_quat)
    block_xy_drift = float(np.linalg.norm(final_block[:2] - initial_block[:2]))
    settle_frames = int(getattr(controller, "settle_steps", 0))

    reasons = []
    if not success:
        reasons.append("controller did not report physical success")
    if "VERIFY_FINAL_STATE" not in stages or unique_stages[-1] not in {"VERIFY_FINAL_STATE", "DONE"}:
        reasons.append("rollout ended before VERIFY_FINAL_STATE")
    if release_step is None or getattr(controller, "success_step", None) is None:
        reasons.append("success did not occur after OPEN_GRIPPER")
    elif getattr(controller, "success_step", 0) <= release_step:
        reasons.append("success occurred before release")
    if settle_frames < 35:
        reasons.append(f"settle frames {settle_frames} < 35")
    if peg_lift_z_increase < 0.040:
        reasons.append(f"peg lift {peg_lift_z_increase:.5f} < 0.040")
    if peg_xy_movement < 0.050:
        reasons.append(f"peg xy movement {peg_xy_movement:.5f} < 0.050")
    if peg_hole_xy_error > 0.045:
        reasons.append(f"peg-hole xy error {peg_hole_xy_error:.5f} > 0.045")
    if abs(peg_final_z_error) > 0.055:
        reasons.append(f"peg final z error {peg_final_z_error:.5f} outside tolerance")
    if peg_upright_score < 0.94:
        reasons.append(f"peg upright score {peg_upright_score:.5f} < 0.94")
    if block_xy_drift > 0.018:
        reasons.append(f"block xy drift {block_xy_drift:.5f} > 0.018")
    if allow_helper or getattr(controller, "oracle_helper_used", False):
        reasons.append("oracle helper was used")
    if guard.count:
        reasons.append(f"direct object joint pose writes during rollout: {guard.count}")

    return {
        "seed": seed,
        "passed": not reasons,
        "reasons": reasons,
        "success": bool(success),
        "done": bool(done),
        "total_steps": len(timeline),
        "stage_timeline": unique_stages,
        "peg_initial_pose": initial_peg.astype(float).round(6).tolist(),
        "peg_final_pose": final_peg.astype(float).round(6).tolist(),
        "peg_final_quat": final_peg_quat.astype(float).round(6).tolist(),
        "block_initial_pose": initial_block.astype(float).round(6).tolist(),
        "block_final_pose": final_block.astype(float).round(6).tolist(),
        "hole_final_pose": h.astype(float).round(6).tolist(),
        "peg_lift_z_increase": peg_lift_z_increase,
        "peg_hole_initial_xy_distance": float(np.linalg.norm(initial_peg[:2] - initial_hole[:2])),
        "transport_distance": float(np.linalg.norm(initial_peg[:2] - initial_hole[:2])),
        "peg_pose_after_lift": peg_pose_after_lift,
        "peg_pose_before_insertion": peg_pose_before_insertion,
        "eef_pose_before_insertion": eef_pose_before_insertion,
        "peg_eef_distance_transport_max": None if transport_eef_dist.size == 0 else float(transport_eef_dist.max()),
        "peg_z_drop_during_transport": peg_z_drop_transport,
        "peg_xy_movement_distance": peg_xy_movement,
        "peg_hole_xy_error": peg_hole_xy_error,
        "peg_final_z_error": peg_final_z_error,
        "peg_upright_score": peg_upright_score,
        "block_xy_drift": block_xy_drift,
        "gripper_opened_before_success": bool(release_step is not None and getattr(controller, "success_step", None) is not None and controller.success_step > release_step),
        "settle_frames_after_release": settle_frames,
        "oracle_helper_used": bool(getattr(controller, "oracle_helper_used", False)),
        "direct_object_pose_writes_during_rollout": int(guard.count),
        "reset_info": reset_info,
        "final_debug": timeline[-1] if timeline else {},
        "timeline": timeline,
    }


def compute_diversity_metrics(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    peg_initials = np.asarray([item["peg_initial_pose"][:2] for item in per_seed], dtype=np.float32)
    block_initials = np.asarray([item["block_initial_pose"][:2] for item in per_seed], dtype=np.float32)
    if len(peg_initials) < 2:
        return {"n": len(peg_initials)}
    return {
        "n": len(peg_initials),
        "peg_x_range": float(peg_initials[:, 0].max() - peg_initials[:, 0].min()),
        "peg_y_range": float(peg_initials[:, 1].max() - peg_initials[:, 1].min()),
        "block_x_range": float(block_initials[:, 0].max() - block_initials[:, 0].min()),
        "block_y_range": float(block_initials[:, 1].max() - block_initials[:, 1].min()),
        "peg_xy_std": peg_initials.std(axis=0).astype(float).round(6).tolist(),
        "block_xy_std": block_initials.std(axis=0).astype(float).round(6).tolist(),
        "peg_initial_positions": peg_initials.astype(float).round(5).tolist(),
        "block_initial_positions": block_initials.astype(float).round(5).tolist(),
    }


def print_running_summary(per_seed: List[Dict[str, Any]]) -> None:
    pass_count = sum(1 for item in per_seed if item["passed"])
    fail_count = len(per_seed) - pass_count
    failures = Counter(primary_failure(item) for item in per_seed if not item["passed"])
    common = failures.most_common(1)[0][0] if failures else "none"
    print(f"running: completed={len(per_seed)} pass={pass_count} fail={fail_count} success_rate={pass_count / len(per_seed):.3f} most_common_failure={common}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-oracle-state-helper", action="store_true")
    parser.add_argument("--output-dir", default="reports/peg_insertion_rollout_qa")
    parser.add_argument("--randomization-level", default="debug_small", choices=["debug_small", "medium", "final", "diverse", "diverse_v2"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=resolve_bddl_path(custom_task="peg_insertion"),
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
                print(f"[peg qa seed {idx + 1}/{args.num_seeds}] seed={seed} start", flush=True)
            result = run_seed(
                env,
                seed,
                args.horizon,
                args.allow_oracle_state_helper,
                args.randomization_level,
                progress_interval=0 if args.quiet else 100,
            )
            per_seed.append(result)
            if not args.quiet:
                status = "PASS" if result["passed"] else "FAIL"
                print(
                    f"[peg qa seed {idx + 1}/{args.num_seeds}] seed={seed} {status} "
                    f"dist={result['transport_distance']:.3f} xy_err={result['peg_hole_xy_error']:.3f} "
                    f"steps={result['total_steps']} stage={result.get('final_debug', {}).get('stage')} "
                    f"reason={primary_failure(result)}",
                    flush=True,
                )
                print_running_summary(per_seed)
    finally:
        env.close()

    diversity = compute_diversity_metrics(per_seed)
    cross_seed_reasons = []
    if args.num_seeds > 1:
        if max(diversity.get("peg_xy_std", [0.0, 0.0])) < 0.010:
            cross_seed_reasons.append("peg initial xy std too low")
        if max(diversity.get("block_xy_std", [0.0, 0.0])) < 0.010:
            cross_seed_reasons.append("block initial xy std too low")
    if args.randomization_level == "diverse_v2" and args.num_seeds >= 10:
        for key, minimum in [("peg_x_range", 0.16), ("peg_y_range", 0.18), ("block_x_range", 0.16), ("block_y_range", 0.18)]:
            if diversity.get(key, 0.0) < minimum:
                cross_seed_reasons.append(f"{key} {diversity.get(key, 0.0):.4f} < {minimum}")

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strict": bool(args.strict),
        "num_seeds": args.num_seeds,
        "randomization_level": args.randomization_level,
        "diversity": diversity,
        "cross_seed_reasons": cross_seed_reasons,
        "per_seed": per_seed,
    }
    report["passed"] = bool(not cross_seed_reasons and all(item["passed"] for item in per_seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qa_peg_insertion_rollout.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    console_report = {**{k: v for k, v in report.items() if k != "per_seed"}, "per_seed": [{k: v for k, v in item.items() if k != "timeline"} for item in per_seed]}
    print(json.dumps(console_report, indent=2, sort_keys=True))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
