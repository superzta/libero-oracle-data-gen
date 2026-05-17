"""Collect oracle rollouts from LIBERO environments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controllers import CONTROLLER_REGISTRY
from button_box_reset_utils import apply_button_box_reset_randomization
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
    if args.custom_task == "button_box":
        obs = apply_button_box_reset_randomization(env, obs, seed, settle_steps=20)
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

    for _ in range(args.horizon):
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
        physical_success = controller.is_success(obs, info, env)
        if args.require_physical_success:
            physical_success = physical_success and passes_button_box_physical_checks(controller, obs)
        if physical_success and not success_seen:
            success = True
            success_seen = True
            failure_reason = ""
            final_hold_remaining = int(args.final_hold_steps)
        if success_seen:
            if final_hold_remaining <= 0:
                break
            final_hold_remaining -= 1
        if done:
            failure_reason = "env_done"
            break

    final_obs = numeric_obs(obs)
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
            "episode_length": len(actions),
            "stage_trace": stage_trace,
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


def passes_button_box_physical_checks(controller, obs: Dict[str, Any]) -> bool:
    if getattr(controller, "allow_oracle_state_helper", False) or getattr(controller, "oracle_helper_used", False):
        return False
    if getattr(controller, "stage", "") != "DONE":
        return False
    try:
        return bool(controller._physical_final_state(obs))
    except Exception:
        return False


def save_video(path: Path, frames: List[np.ndarray], fps: int = 20) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


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
    parser.add_argument("--video-camera", default="agentview")
    parser.add_argument("--output-dir", default="datasets")
    parser.add_argument("--keep-failures", action="store_true")
    args = parser.parse_args()
    if args.min_horizon and args.horizon < args.min_horizon:
        args.horizon = args.min_horizon
    if args.disallow_oracle_helper and args.allow_oracle_state_helper:
        raise SystemExit("--allow-oracle-state-helper conflicts with --disallow-oracle-helper")

    args.bddl_path = resolve_bddl_path(args.task_name, args.suite, args.task_id, args.bddl_file, args.custom_task)
    language = get_task_language(args.bddl_path)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{Path(args.bddl_path).stem}_{args.controller}_{run_id}"
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

    try:
        while successes < args.num_successes and attempts < args.max_attempts:
            seed = args.seed + attempts
            episode = collect_one(env, controller, args, seed, attempts, {**base_metadata, **controller_metadata})
            attempts += 1
            seeds.append(seed)
            if episode["success"]:
                successes += 1
                ep_name = f"success_{successes:03d}_seed_{seed}.hdf5"
                save_episode_hdf5(out_dir / ep_name, episode)
                if args.save_video:
                    save_video(video_dir / ep_name.replace(".hdf5", ".mp4"), episode["frames"])
            else:
                failure_reasons[episode["failure_reason"]] = failure_reasons.get(episode["failure_reason"], 0) + 1
                if args.keep_failures:
                    save_episode_hdf5(out_dir / "failures" / f"attempt_{attempts:04d}_seed_{seed}.hdf5", episode)
            print(f"attempt={attempts} seed={seed} success={episode['success']} successes={successes}/{args.num_successes}")
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
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
