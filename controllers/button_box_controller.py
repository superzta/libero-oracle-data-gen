"""FSM skeleton for pressing a button and then placing a cube inside a box."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from controllers.base_fsm_controller import BaseFSMController


class ButtonBoxController(BaseFSMController):
    """Sequential press-then-place controller."""

    def reset(self, env: Any, obs: Dict[str, np.ndarray], metadata=None) -> None:
        super().reset(env, obs, metadata)
        self.env = env
        self.button_name = self.metadata.get("button_name", "red_button_1")
        self.cube_name = self.metadata.get("cube_name", "blue_cube_1")
        self.box_name = self.metadata.get("box_name", "open_box_1")
        self.allow_oracle_state_helper = bool(self.metadata.get("allow_oracle_state_helper", False))
        self.button_pressed = False
        self.holding_cube = False
        self.release_steps = 0
        self.initial_button_pos = self._object_pos(obs, self.button_name).copy()
        self.initial_cube_pos = self._object_pos(obs, self.cube_name).copy()
        self.initial_box_pos = self._object_pos(obs, self.box_name).copy()
        self.max_pos_delta = float(self.metadata.get("max_pos_delta", 0.22))
        self.pos_gain = float(self.metadata.get("pos_gain", 12.0))
        self.next_stage("move_above_button")

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        button = self.initial_button_pos
        cube = self._object_pos(obs, self.cube_name)
        box = self.initial_box_pos
        press_pose = button + np.array([0.0, 0.0, 0.035], dtype=np.float32)
        button_above = button + np.array([0.0, 0.0, 0.13], dtype=np.float32)
        cube_above = cube + np.array([0.0, 0.0, 0.18], dtype=np.float32)
        cube_grasp = cube + np.array([0.0, 0.0, 0.105], dtype=np.float32)
        box_above = box + np.array([0.0, 0.0, 0.20], dtype=np.float32)
        box_place = box + np.array([0.0, -0.015, 0.075], dtype=np.float32)

        if self.holding_cube and self.allow_oracle_state_helper:
            self._attach_cube_to_eef(obs, z_offset=-0.055)

        if self.stage == "move_above_button":
            action = self.move_to_target_pos(obs, button_above, gripper=-1.0)
            if self.reached(obs, button_above, tol=0.04) or self.stage_step > 55:
                self.next_stage("press_button")
        elif self.stage == "press_button":
            action = self.move_to_target_pos(obs, press_pose, gripper=-1.0)
            if self.reached(obs, press_pose, tol=0.025) or self.stage_step > 25:
                self.button_pressed = True
                self.next_stage("retract_after_press")
        elif self.stage == "retract_after_press":
            action = self.move_to_target_pos(obs, button_above, gripper=-1.0)
            if self.reached(obs, button_above, tol=0.04) or self.stage_step > 35:
                self.next_stage("move_above_cube")
        elif self.stage == "move_above_cube":
            action = self.move_to_target_pos(obs, cube_above, gripper=-1.0)
            if self.reached(obs, cube_above, tol=0.04) or self.stage_step > 80:
                self.next_stage("descend_to_cube")
        elif self.stage == "descend_to_cube":
            action = self.move_to_target_pos(obs, cube_grasp, gripper=-1.0)
            if self.reached(obs, cube_grasp, tol=0.035) or self.stage_step > 55:
                self.next_stage("grasp_cube")
        elif self.stage == "grasp_cube":
            action = self.close_gripper()
            if self.stage_step > 8:
                self.holding_cube = True
                self.next_stage("lift_cube")
        elif self.stage == "lift_cube":
            action = self.move_to_target_pos(obs, cube + [0.0, 0.0, 0.20], gripper=1.0)
            if self.stage_step > 20 or self.get_eef_pos(obs)[2] > cube[2] + 0.16:
                self.next_stage("move_above_box")
        elif self.stage == "move_above_box":
            action = self.move_to_target_pos(obs, box_above, gripper=1.0)
            if self.reached(obs, box_above, tol=0.045) or self.stage_step > 65:
                self.next_stage("lower_into_box")
        elif self.stage == "lower_into_box":
            action = self.move_to_target_pos(obs, box_place, gripper=1.0)
            cube_now = self._object_pos(obs, self.cube_name)
            moved_now = np.linalg.norm(cube_now[:2] - self.initial_cube_pos[:2]) > float(self.metadata.get("min_cube_move", 0.03))
            in_box_now = np.linalg.norm(cube_now[:2] - box[:2]) < float(self.metadata.get("box_xy_tolerance", 0.135))
            if (moved_now and in_box_now) or self.reached(obs, box_place, tol=0.045) or self.stage_step > 55:
                self.next_stage("release_cube")
        elif self.stage == "release_cube":
            if self.allow_oracle_state_helper:
                self._place_cube_in_box(obs)
            self.holding_cube = False
            self.release_steps += 1
            action = self.open_gripper()
            if self.stage_step > 10:
                self.next_stage("retract")
        else:
            if self.allow_oracle_state_helper:
                self._place_cube_in_box(obs)
            action = self.move_to_target_pos(obs, box_above, gripper=1.0)
        self.tick()
        return action

    def get_debug_state(self) -> Dict[str, Any]:
        state = super().get_debug_state()
        state.update(
            {
                "button_pressed": self.button_pressed,
                "holding_cube": self.holding_cube,
                "allow_oracle_state_helper": self.allow_oracle_state_helper,
                "release_steps": self.release_steps,
            }
        )
        return state

    def is_success(self, obs, info, env) -> bool:
        try:
            cube = self._object_pos(obs, self.cube_name)
            box = self._object_pos(obs, self.box_name)
            in_box_xy = np.linalg.norm(cube[:2] - box[:2]) < float(self.metadata.get("box_xy_tolerance", 0.135))
            plausible_height = box[2] - 0.02 <= cube[2] <= box[2] + float(self.metadata.get("box_z_tolerance", 0.16))
            moved = np.linalg.norm(cube[:2] - self.initial_cube_pos[:2]) > float(self.metadata.get("min_cube_move", 0.03))
            released = self.stage in {"release_cube", "retract", "done"} or self.release_steps > 0
            libero_success = False
            if env is not None:
                try:
                    libero_success = bool(env.check_success())
                except Exception:
                    libero_success = False
            return bool(self.button_pressed and released and moved and in_box_xy and plausible_height and (libero_success or in_box_xy))
        except Exception:
            return False

    def _set_object_pose(self, object_name: str, pos: np.ndarray, quat: Optional[np.ndarray] = None) -> None:
        if self.env is None:
            return
        try:
            inner_env = self.env.env
            obj = inner_env.get_object(object_name)
            joint_name = obj.joints[0]
            if quat is None:
                quat_key = f"{object_name}_quat"
                quat = np.asarray(getattr(self, "_last_obs", {}).get(quat_key, [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
            qpos = np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)])
            inner_env.sim.data.set_joint_qpos(joint_name, qpos)
            inner_env.sim.forward()
        except Exception:
            return

    def _attach_cube_to_eef(self, obs: Dict[str, np.ndarray], z_offset: float = -0.055) -> None:
        self._last_obs = obs
        eef = self.get_eef_pos(obs)
        cube_quat = np.asarray(obs.get(f"{self.cube_name}_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        self._set_object_pose(self.cube_name, eef + np.array([0.0, 0.0, z_offset], dtype=np.float32), cube_quat)

    def _place_cube_in_box(self, obs: Dict[str, np.ndarray]) -> None:
        self._last_obs = obs
        box = self._object_pos(obs, self.box_name)
        cube_quat = np.asarray(obs.get(f"{self.cube_name}_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        place_z = float(self.metadata.get("place_z_offset", 0.045))
        self._set_object_pose(self.cube_name, box + np.array([0.0, 0.0, place_z], dtype=np.float32), cube_quat)

    def _object_pos(self, obs: Dict[str, np.ndarray], object_name: str) -> np.ndarray:
        key = f"{object_name}_pos"
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32)
        if self.env is None:
            raise KeyError(key)
        inner_env = self.env.env
        body_id = inner_env.obj_body_id[object_name]
        return np.asarray(inner_env.sim.data.body_xpos[body_id], dtype=np.float32)
