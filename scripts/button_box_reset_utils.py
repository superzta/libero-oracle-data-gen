"""Reset-time initialization helpers for the custom button_box task."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


CUBE = "blue_cube_1"
BOX = "open_box_1"
BUTTON = "red_button_1"
UPRIGHT_CUBE_QUAT_WXYZ = np.asarray([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float64)
IDENTITY_QUAT_WXYZ = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def numeric_wait_action(gripper: float = -1.0) -> np.ndarray:
    return np.asarray([0, 0, 0, 0, 0, 0, gripper], dtype=np.float32)


def apply_button_box_reset_randomization(
    env: Any,
    obs: Dict[str, Any],
    seed: int,
    settle_steps: int = 20,
) -> Dict[str, Any]:
    """Apply feasible cube randomization before rollout begins.

    Direct simulator writes here are reset-time initialization, not rollout
    oracle actions. Rollout write guards are installed after this helper.
    """

    rng = np.random.default_rng(seed + 7919)
    cube_x = float(rng.uniform(-0.030, 0.030))
    cube_y = float(rng.uniform(-0.035, -0.005))
    cube_z = 0.990
    cube = env.env.get_object(CUBE)
    box = env.env.get_object(BOX)
    try:
        button = env.env.get_object(BUTTON)
        body_id = env.env.sim.model.body_name2id(button.root_body)
        env.env.sim.model.body_pos[body_id] = np.asarray([-0.155, -0.155, 0.900], dtype=np.float64)
        env.env.sim.model.body_quat[body_id] = IDENTITY_QUAT_WXYZ
    except Exception:
        pass
    env.env.sim.data.set_joint_qpos(
        box.joints[0],
        np.concatenate([np.asarray([0.055, 0.095, 0.900], dtype=np.float64), IDENTITY_QUAT_WXYZ]),
    )
    env.env.sim.data.set_joint_qpos(
        cube.joints[0],
        np.concatenate([np.asarray([cube_x, cube_y, cube_z], dtype=np.float64), UPRIGHT_CUBE_QUAT_WXYZ]),
    )
    env.env.sim.forward()
    for _ in range(settle_steps):
        env.env.sim.step()
        try:
            env.env._post_process()
        except Exception:
            pass
    try:
        env.env._update_observables(force=True)
        return env.env._get_observations(force_update=True)
    except Exception:
        env.env.sim.forward()
        return obs
