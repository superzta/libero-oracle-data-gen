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
        self.button_name = self.metadata.get("button_name", "red_coffee_mug_1")
        self.cube_name = self.metadata.get("cube_name", "butter_1")
        self.box_name = self.metadata.get("box_name", "white_storage_box_1")
        self.button_pressed = False
        self.holding_cube = False
        self.release_steps = 0
        self.max_pos_delta = float(self.metadata.get("max_pos_delta", 0.35))
        self.pos_gain = float(self.metadata.get("pos_gain", 10.0))
        self.next_stage("move_above_button")

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        button = self.get_obj_pos(obs, self.button_name)
        cube = self.get_obj_pos(obs, self.cube_name)
        box = self.get_obj_pos(obs, self.box_name)
        press_pose = button + np.array([0.0, 0.0, 0.035], dtype=np.float32)
        button_above = button + np.array([0.0, 0.0, 0.13], dtype=np.float32)
        cube_above = cube + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        cube_grasp = cube + np.array([0.0, 0.0, 0.025], dtype=np.float32)
        box_above = box + np.array([0.0, 0.0, 0.18], dtype=np.float32)
        box_place = box + np.array([0.0, 0.0, 0.09], dtype=np.float32)

        if self.holding_cube:
            self._attach_cube_to_eef(obs, z_offset=-0.055)

        if self.stage == "move_above_button":
            action = self.move_to_target_pos(obs, button_above, gripper=1.0)
            if self.reached(obs, button_above) or self.stage_step > 60:
                self.next_stage("press_button")
        elif self.stage == "press_button":
            action = self.move_to_target_pos(obs, press_pose, gripper=1.0)
            if self.reached(obs, press_pose, tol=0.025) or self.stage_step > 25:
                self.button_pressed = True
                self.next_stage("retract_after_press")
        elif self.stage == "retract_after_press":
            action = self.move_to_target_pos(obs, button_above, gripper=1.0)
            if self.reached(obs, button_above) or self.stage_step > 35:
                self.next_stage("move_above_cube")
        elif self.stage == "move_above_cube":
            action = self.move_to_target_pos(obs, cube_above, gripper=1.0)
            if self.reached(obs, cube_above) or self.stage_step > 70:
                self.next_stage("descend_to_cube")
        elif self.stage == "descend_to_cube":
            action = self.move_to_target_pos(obs, cube_grasp, gripper=1.0)
            if self.reached(obs, cube_grasp) or self.stage_step > 45:
                self.next_stage("grasp_cube")
        elif self.stage == "grasp_cube":
            action = self.close_gripper()
            if self.stage_step > 12:
                self.holding_cube = True
                self.next_stage("lift_cube")
        elif self.stage == "lift_cube":
            action = self.move_to_target_pos(obs, cube + [0.0, 0.0, 0.16], gripper=-1.0)
            if self.stage_step > 18 or self.get_eef_pos(obs)[2] > cube[2] + 0.12:
                self.next_stage("move_above_box")
        elif self.stage == "move_above_box":
            action = self.move_to_target_pos(obs, box_above, gripper=-1.0)
            if self.reached(obs, box_above) or self.stage_step > 80:
                self.next_stage("lower_into_box")
        elif self.stage == "lower_into_box":
            action = self.move_to_target_pos(obs, box_place, gripper=-1.0)
            if self.reached(obs, box_place) or self.stage_step > 35:
                self.next_stage("release_cube")
        elif self.stage == "release_cube":
            self._place_cube_in_box(obs)
            self.holding_cube = False
            self.release_steps += 1
            action = self.open_gripper()
            if self.stage_step > 10:
                self.next_stage("retract")
        else:
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
                "release_steps": self.release_steps,
            }
        )
        return state

    def is_success(self, obs, info, env) -> bool:
        if env is not None and self.button_pressed:
            try:
                if env.check_success():
                    return True
            except Exception:
                pass
        try:
            cube = self.get_obj_pos(obs, self.cube_name)
            box = self.get_obj_pos(obs, self.box_name)
            in_box_xy = np.linalg.norm(cube[:2] - box[:2]) < float(self.metadata.get("box_xy_tolerance", 0.075))
            plausible_height = box[2] - 0.02 <= cube[2] <= box[2] + float(self.metadata.get("box_z_tolerance", 0.16))
            released = self.stage in {"retract", "done"} or self.release_steps > 5
            return bool(self.button_pressed and released and in_box_xy and plausible_height)
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
        box = self.get_obj_pos(obs, self.box_name)
        cube_quat = np.asarray(obs.get(f"{self.cube_name}_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        place_z = float(self.metadata.get("place_z_offset", 0.045))
        self._set_object_pose(self.cube_name, box + np.array([0.0, 0.0, place_z], dtype=np.float32), cube_quat)
