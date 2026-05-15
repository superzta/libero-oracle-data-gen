"""Inspect button-box object motion across seeds without trusting success flags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controllers.button_box_controller import ButtonBoxController
from libero_env_utils import configure_runtime_env, resolve_bddl_path

configure_runtime_env()


def object_pos(env, obs, name: str) -> np.ndarray:
    key = f"{name}_pos"
    if key in obs:
        return np.asarray(obs[key], dtype=np.float32)
    return np.asarray(env.env.sim.data.body_xpos[env.env.obj_body_id[name]], dtype=np.float32)


def run_seed(env, seed: int, horizon: int, allow_helper: bool) -> dict:
    env.seed(seed)
    obs = env.reset()
    controller = ButtonBoxController()
    metadata = {"allow_oracle_state_helper": allow_helper}
    controller.reset(env, obs, metadata)
    initial = {
        "button": object_pos(env, obs, controller.button_name),
        "cube": object_pos(env, obs, controller.cube_name),
        "box": object_pos(env, obs, controller.box_name),
    }
    success = False
    for _ in range(horizon):
        action = controller.act(obs)
        obs, reward, done, info = env.step(action)
        success = controller.is_success(obs, info, env)
        if success or done:
            break
    final = {
        "button": object_pos(env, obs, controller.button_name),
        "cube": object_pos(env, obs, controller.cube_name),
        "box": object_pos(env, obs, controller.box_name),
    }
    return {
        "seed": seed,
        "button_name": controller.button_name,
        "cube_name": controller.cube_name,
        "box_name": controller.box_name,
        "initial_button_pos": initial["button"].round(4).tolist(),
        "initial_cube_pos": initial["cube"].round(4).tolist(),
        "initial_box_pos": initial["box"].round(4).tolist(),
        "final_button_pos": final["button"].round(4).tolist(),
        "final_cube_pos": final["cube"].round(4).tolist(),
        "final_box_pos": final["box"].round(4).tolist(),
        "cube_moved_distance": float(np.linalg.norm(final["cube"][:2] - initial["cube"][:2])),
        "cube_physically_moved": bool(np.linalg.norm(final["cube"][:2] - initial["cube"][:2]) > 0.01),
        "oracle_helper_used": allow_helper,
        "controller_success": bool(success),
        "debug": controller.get_debug_state(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=350)
    parser.add_argument("--camera-size", type=int, default=96)
    parser.add_argument("--allow-oracle-state-helper", action="store_true")
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=resolve_bddl_path(custom_task="button_box"),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        horizon=args.horizon,
    )
    try:
        results = [
            run_seed(env, args.seed + idx, args.horizon, args.allow_oracle_state_helper)
            for idx in range(args.num_seeds)
        ]
    finally:
        env.close()

    cube_initials = [tuple(item["initial_cube_pos"]) for item in results]
    report = {
        "results": results,
        "object_positions_differ_across_seeds": len(set(cube_initials)) > 1,
        "all_cubes_physically_moved": all(item["cube_physically_moved"] for item in results),
        "oracle_helper_used": bool(args.allow_oracle_state_helper),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

