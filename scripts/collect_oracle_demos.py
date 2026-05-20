"""Collect oracle rollouts from LIBERO environments."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controllers import CONTROLLER_REGISTRY
from button_box_reset_utils import apply_button_box_reset_randomization
from peg_insertion_reset_utils import apply_peg_insertion_reset_randomization
from tool_sweep_reset_utils import apply_tool_sweep_reset_randomization
from libero_env_utils import configure_runtime_env, get_task_language, resolve_bddl_path

configure_runtime_env()


def numeric_obs(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out = {}
    for key, value in obs.items():
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            out[key] = arr.copy()
    return out


def make_env(args):
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=args.bddl_path,
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        camera_names=args.camera_names.split(","),
        horizon=args.horizon,
        ignore_done=True,
    )


def render_frame(env, camera_name: str, height: int, width: int):
    try:
        return env.env.sim.render(camera_name=camera_name, height=height, width=width)[::-1]
    except Exception:
        return None


def save_episode_hdf5(path: Path, episode: Dict[str, Any]) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["metadata"] = json.dumps(episode["metadata"], sort_keys=True)
        h5.attrs["success"] = bool(episode["success"])
        h5.attrs["failure_reason"] = episode.get("failure_reason", "")
        for key in ["actions", "rewards", "dones"]:
            h5.create_dataset(key, data=np.asarray(episode[key]), compression="gzip")
        if episode.get("env_states"):
            h5.create_dataset("env_states", data=np.asarray(episode["env_states"]), compression="gzip")
        obs_group = h5.create_group("observations")
        for key, values in episode["observations"].items():
            try:
                obs_group.create_dataset(key, data=np.asarray(values), compression="gzip")
            except TypeError:
                pass
        debug_group = h5.create_group("debug")
        debug_group.attrs["states"] = json.dumps(episode["debug"], sort_keys=True)


def collect_one(env, controller, args, seed: int, attempt: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
    env.seed(seed)
    obs = env.reset()
    try:
        env.env.sim.forward()
    except Exception:
        pass
    reset_info: Dict[str, Any] = {}
    if args.custom_task == "button_box":
        obs, reset_info = apply_button_box_reset_randomization(
            env, obs, seed,
            settle_steps=20,
            randomization_level=args.randomization_level,
        )
    elif args.custom_task == "peg_insertion":
        obs, reset_info = apply_peg_insertion_reset_randomization(
            env, obs, seed,
            settle_steps=20,
            randomization_level=args.randomization_level,
        )
    elif args.custom_task == "tool_sweep":
        obs, reset_info = apply_tool_sweep_reset_randomization(
            env, obs, seed,
            settle_steps=20,
            randomization_level=args.randomization_level,
        )
    controller.reset(env, obs, metadata)
    observations: Dict[str, List[np.ndarray]] = {key: [] for key in numeric_obs(obs)}
    actions, rewards, dones, env_states, debug = [], [], [], [], []
    frames = []
    initial_obs = numeric_obs(obs)
    success = False
    success_seen = False
    final_hold_remaining = 0
    failure_reason = "max_steps"
    info = {}
    stage_trace = []

    for step in range(args.horizon):
        for key, value in numeric_obs(obs).items():
            observations.setdefault(key, []).append(value)
        try:
            env_states.append(np.asarray(env.get_sim_state()).copy())
        except Exception:
            pass
        if args.save_video:
            frame = render_frame(env, args.video_camera, args.camera_size, args.camera_size)
            if frame is not None:
                frames.append(frame)
        action = np.asarray(controller.act(obs), dtype=np.float32)
        obs, reward, done, info = env.step(action)
        actions.append(action.copy())
        rewards.append(float(reward))
        dones.append(bool(done))
        setattr(controller, "_last_debug_obs", obs)
        debug_state = controller.get_debug_state()
        debug.append(debug_state)
        if args.save_stage_trace:
            stage_trace.append(debug_state)
        if not args.quiet and step > 0 and step % 100 == 0:
            print(
                f"  [collect seed={seed}] step={step} stage={debug_state.get('stage')} "
                f"reason={debug_state.get('transition_reason')}",
                flush=True,
            )
        physical_success = controller.is_success(obs, info, env)
        if args.require_physical_success:
            physical_success = physical_success and passes_task_physical_checks(args.custom_task, controller, obs)
        if physical_success and not success_seen:
            success = True
            success_seen = True
            failure_reason = ""
            final_hold_remaining = int(args.final_hold_steps)
        if success_seen:
            if final_hold_remaining <= 0:
                break
            final_hold_remaining -= 1
        if done and args.custom_task not in ("button_box", "tool_sweep"):
            failure_reason = "env_done"
            break
        if getattr(controller, "stage", "") == "DONE" and not success_seen:
            debug_reason = debug_state.get("transition_reason") if debug else ""
            failure_reason = debug_reason or "controller_done_without_success"
            break

    final_obs = numeric_obs(obs)
    episode_length = len(actions)
    video_frame_count = len(frames)
    # video_stride=1 means every step has a corresponding frame
    video_stride = 1
    if video_frame_count > 0 and abs(video_frame_count - episode_length) > 2:
        video_stride = max(1, round(episode_length / video_frame_count))

    return {
        "success": success,
        "failure_reason": failure_reason,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "env_states": env_states,
        "frames": frames,
        "debug": debug,
        "initial_obs": initial_obs,
        "metadata": {
            **metadata,
            "seed": seed,
            "attempt": attempt,
            "success": success,
            "failure_reason": failure_reason,
            "episode_length": episode_length,
            "video_frame_count": video_frame_count,
            "video_stride": video_stride,
            "stage_trace": stage_trace,
            "randomization_level": args.randomization_level,
            "cube_bin_id": reset_info.get("cube_bin_id", -1),
            "box_bin_id": reset_info.get("box_bin_id", -1),
            "button_bin_id": reset_info.get("button_bin_id", -1),
            "peg_bin_id": reset_info.get("peg_bin_id", -1),
            "block_bin_id": reset_info.get("block_bin_id", -1),
            "pusher_bin_id": reset_info.get("pusher_bin_id", -1),
            "dustpan_bin_id": reset_info.get("dustpan_bin_id", -1),
            "lane_angle_deg": reset_info.get("lane_angle_deg", None),
            "reset_info": reset_info,
            "initial_positions": {
                key: value.reshape(-1).astype(float).tolist()
                for key, value in initial_obs.items()
                if key.endswith("_pos")
            },
            "final_positions": {
                key: value.reshape(-1).astype(float).tolist()
                for key, value in final_obs.items()
                if key.endswith("_pos")
            },
            "oracle_state_helper_used": bool(getattr(controller, "oracle_helper_used", False)),
            "direct_pose_writes_during_rollout": int(getattr(controller, "direct_pose_writes_during_rollout", 0)),
            "physical_success_required": bool(args.require_physical_success),
            "final_hold_steps": int(args.final_hold_steps),
        },
    }


def passes_task_physical_checks(custom_task: str, controller, obs: Dict[str, Any]) -> bool:
    if getattr(controller, "allow_oracle_state_helper", False) or getattr(controller, "oracle_helper_used", False):
        return False
    if getattr(controller, "stage", "") != "DONE":
        return False
    try:
        if custom_task in {"button_box", "peg_insertion", "tool_sweep"}:
            return bool(controller._physical_final_state(obs))
        return bool(controller.is_success(obs, {}, None))
    except Exception:
        return False


def save_video(path: Path, frames: List[np.ndarray], fps: int = 20) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def save_run_manifest(
    out_dir: Path,
    video_dir: Path,
    args,
    language: str,
    success_hdf5_files: List[str],
    success_video_files: List[str],
    seeds: List[int],
    timestamp: str,
) -> None:
    manifest = {
        "timestamp": timestamp,
        "dataset_dir": str(out_dir),
        "video_dir": str(video_dir) if args.save_video else None,
        "task_name": Path(args.bddl_path).stem,
        "task_language": language,
        "bddl_file": args.bddl_path,
        "controller": args.controller,
        "camera_size": args.camera_size,
        "horizon": args.horizon,
        "final_hold_steps": args.final_hold_steps,
        "randomization_level": args.randomization_level,
        "seed_start": args.seed,
        "seed_list": seeds,
        "success_hdf5_files": success_hdf5_files,
        "success_video_files": success_video_files if args.save_video else [],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--bddl-file", default=None)
    parser.add_argument("--custom-task", default=None, help="Named custom task from bddl_files/, e.g. button_box")
    parser.add_argument("--controller", default="noop", choices=sorted(CONTROLLER_REGISTRY))
    parser.add_argument("--controller-metadata", default="{}", help="JSON object passed to controller.reset")
    parser.add_argument("--allow-oracle-state-helper", action="store_true", help="Allow controllers to set object poses directly for debug collection.")
    parser.add_argument("--disallow-oracle-helper", action="store_true", default=True)
    parser.add_argument("--require-physical-success", action="store_true", default=True)
    parser.add_argument("--save-stage-trace", action="store_true")
    parser.add_argument("--final-hold-steps", type=int, default=30)
    parser.add_argument("--min-horizon", type=int, default=0)
    parser.add_argument("--num-successes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--camera-names", default="agentview,robot0_eye_in_hand")
    parser.add_argument("--save-video", action="store_true")
    parser.set_defaults(video_success_precheck=True)
    parser.add_argument(
        "--video-success-precheck",
        dest="video_success_precheck",
        action="store_true",
        help=(
            "When saving videos, first run each seed without frame capture and only "
            "replay successful seeds with video. This avoids buffering long failed videos."
        ),
    )
    parser.add_argument(
        "--no-video-success-precheck",
        dest="video_success_precheck",
        action="store_false",
        help="Disable the successful-seed replay path and buffer frames during every attempt.",
    )
    parser.add_argument("--video-camera", default="agentview")
    parser.add_argument("--output-dir", default="datasets")
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument(
        "--randomization-level",
        default="debug_small",
        choices=["debug_small", "medium", "final", "diverse", "diverse_v2"],
        help="Initial state randomization level for button_box task.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.min_horizon and args.horizon < args.min_horizon:
        args.horizon = args.min_horizon
    if args.disallow_oracle_helper and args.allow_oracle_state_helper:
        raise SystemExit("--allow-oracle-state-helper conflicts with --disallow-oracle-helper")

    args.bddl_path = resolve_bddl_path(args.task_name, args.suite, args.task_id, args.bddl_file, args.custom_task)
    language = get_task_language(args.bddl_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{Path(args.bddl_path).stem}_{args.controller}_{timestamp}"
    video_dir = Path("videos") / out_dir.name
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    controller_cls = CONTROLLER_REGISTRY[args.controller]
    controller = controller_cls()
    controller_metadata = json.loads(args.controller_metadata)
    if args.allow_oracle_state_helper:
        controller_metadata["allow_oracle_state_helper"] = True
    base_metadata = {
        "task_language": language,
        "bddl_file": args.bddl_path,
        "controller": args.controller,
        "controller_metadata": controller_metadata,
    }
    env = make_env(args)
    successes = 0
    attempts = 0
    seeds = []
    failure_reasons: Dict[str, int] = {}
    success_hdf5_files: List[str] = []
    success_video_files: List[str] = []

    try:
        while successes < args.num_successes and attempts < args.max_attempts:
            seed = args.seed + attempts
            if not args.quiet:
                print(f"[collect attempt {attempts + 1}/{args.max_attempts}] seed={seed} start", flush=True)
            if args.save_video and args.video_success_precheck:
                precheck_args = copy.copy(args)
                precheck_args.save_video = False
                episode = collect_one(
                    env, controller, precheck_args, seed, attempts, {**base_metadata, **controller_metadata}
                )
                if episode["success"]:
                    if not args.quiet:
                        print(f"[collect attempt {attempts + 1}/{args.max_attempts}] seed={seed} replay video", flush=True)
                    video_episode = collect_one(env, controller, args, seed, attempts, {**base_metadata, **controller_metadata})
                    if video_episode["success"]:
                        episode = video_episode
                    else:
                        episode = video_episode
                        episode["failure_reason"] = f"video_replay_failed:{episode['failure_reason']}"
            else:
                episode = collect_one(env, controller, args, seed, attempts, {**base_metadata, **controller_metadata})
            attempts += 1
            seeds.append(seed)
            if episode["success"]:
                successes += 1
                ep_name = f"success_{successes:03d}_seed_{seed}.hdf5"
                save_episode_hdf5(out_dir / ep_name, episode)
                success_hdf5_files.append(str(out_dir / ep_name))
                if args.save_video:
                    video_path = video_dir / ep_name.replace(".hdf5", ".mp4")
                    save_video(video_path, episode["frames"])
                    success_video_files.append(str(video_path))
                # Warn if frame count doesn't match episode length without a known stride
                ep_len = episode["metadata"]["episode_length"]
                vf_count = episode["metadata"]["video_frame_count"]
                v_stride = episode["metadata"]["video_stride"]
                if args.save_video and vf_count > 0 and abs(vf_count - ep_len) > 2 and v_stride == 1:
                    print(
                        f"WARNING: success_{successes:03d}_seed_{seed}: "
                        f"episode_length={ep_len} but video_frames={vf_count} "
                        f"with video_stride=1 — mismatch without known stride"
                    )
            else:
                failure_reasons[episode["failure_reason"]] = failure_reasons.get(episode["failure_reason"], 0) + 1
                if args.keep_failures:
                    save_episode_hdf5(out_dir / "failures" / f"attempt_{attempts:04d}_seed_{seed}.hdf5", episode)
            if not args.quiet:
                status = "PASS" if episode["success"] else "FAIL"
                reason = episode["failure_reason"] or "ok"
                print(
                    f"[collect attempt {attempts}/{args.max_attempts}] seed={seed} {status} "
                    f"successes={successes}/{args.num_successes} reason={reason}",
                    flush=True,
                )
                if attempts % 3 == 0 or episode["success"] or attempts == args.max_attempts:
                    fail_count = attempts - successes
                    rate = successes / attempts if attempts else 0.0
                    common = Counter(failure_reasons).most_common(1)
                    common_reason = common[0][0] if common else "none"
                    print(
                        f"running: attempts={attempts} pass={successes} fail={fail_count} "
                        f"success_rate={rate:.3f} most_common_failure={common_reason}",
                        flush=True,
                    )
            del episode
            gc.collect()
    finally:
        env.close()

    summary = {
        "task_language": language,
        "bddl_file": args.bddl_path,
        "output_dir": str(out_dir),
        "successes": successes,
        "attempts": attempts,
        "success_rate": successes / attempts if attempts else 0.0,
        "seeds": seeds,
        "failure_reasons": failure_reasons,
        "randomization_level": args.randomization_level,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    save_run_manifest(
        out_dir=out_dir,
        video_dir=video_dir,
        args=args,
        language=language,
        success_hdf5_files=success_hdf5_files,
        success_video_files=success_video_files,
        seeds=seeds,
        timestamp=timestamp,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
