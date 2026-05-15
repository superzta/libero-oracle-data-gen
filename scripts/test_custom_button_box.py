"""Smoke test for the custom button-box LIBERO task."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controllers.button_box_controller import ButtonBoxController
from libero_env_utils import configure_runtime_env, resolve_bddl_path

configure_runtime_env()


def object_pos(env, obs, name):
    key = f"{name}_pos"
    if key in obs:
        return np.asarray(obs[key])
    return np.asarray(env.env.sim.data.body_xpos[env.env.obj_body_id[name]])


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = resolve_bddl_path(custom_task="button_box")
    print("BDDL file:", bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=96,
        camera_widths=96,
        camera_names=["agentview", "robot0_eye_in_hand"],
        horizon=200,
    )
    try:
        env.seed(0)
        obs = env.reset()
        print("Reset OK")
        print("Observation keys:", list(obs.keys()))
        print("Button pos:", object_pos(env, obs, "red_button_1").round(4).tolist())
        print("Blue cube pos:", object_pos(env, obs, "blue_cube_1").round(4).tolist())
        print("Open box pos:", object_pos(env, obs, "open_box_1").round(4).tolist())

        controller = ButtonBoxController()
        controller.reset(env, obs, {})
        for step in range(20):
            action = controller.act(obs)
            obs, reward, done, info = env.step(action)
            if step < 5 or step % 5 == 0:
                print(f"step={step:03d} action={np.asarray(action).round(3).tolist()} debug={controller.get_debug_state()}")
            if done:
                print("Environment ended early at step", step)
                break
        print("Custom button-box smoke test OK")
    finally:
        env.close()


if __name__ == "__main__":
    main()
