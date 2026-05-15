"""FSM skeleton for pressing a button and then placing a cube inside a box."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from controllers.base_fsm_controller import BaseFSMController


class ButtonBoxController(BaseFSMController):
    """Sequential press-then-place controller."""

    def reset(self, env: Any, obs: Dict[str, np.ndarray], metadata=None) -> None:
        super().reset(env, obs, metadata)
        self.button_name = self.metadata.get("button_name", "red_button_1")
        self.cube_name = self.metadata.get("cube_name", "blue_cube_1")
        self.box_name = self.metadata.get("box_name", "box_1")
        self.next_stage("move_above_button")

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        button = self.get_obj_pos(obs, self.button_name)
        cube = self.get_obj_pos(obs, self.cube_name)
        box = self.get_obj_pos(obs, self.box_name)
        if self.stage == "move_above_button":
            action = self.move_to_target_pos(obs, button + [0.0, 0.0, 0.10], gripper=1.0)
            if self.reached(obs, button + [0.0, 0.0, 0.10]):
                self.next_stage("press_button")
        elif self.stage == "press_button":
            action = self.move_to_target_pos(obs, button + [0.0, 0.0, 0.005], gripper=1.0)
            if self.stage_step > 18:
                self.next_stage("move_above_cube")
        elif self.stage == "move_above_cube":
            action = self.move_to_target_pos(obs, cube + [0.0, 0.0, 0.12], gripper=1.0)
            if self.reached(obs, cube + [0.0, 0.0, 0.12]):
                self.next_stage("descend_to_cube")
        elif self.stage == "descend_to_cube":
            action = self.move_to_target_pos(obs, cube + [0.0, 0.0, 0.02], gripper=1.0)
            if self.reached(obs, cube + [0.0, 0.0, 0.02]):
                self.next_stage("grasp_cube")
        elif self.stage == "grasp_cube":
            action = self.close_gripper()
            if self.stage_step > 12:
                self.next_stage("lift_cube")
        elif self.stage == "lift_cube":
            action = self.move_to_target_pos(obs, cube + [0.0, 0.0, 0.14], gripper=-1.0)
            if self.stage_step > 18:
                self.next_stage("move_above_box")
        elif self.stage == "move_above_box":
            action = self.move_to_target_pos(obs, box + [0.0, 0.0, 0.14], gripper=-1.0)
            if self.reached(obs, box + [0.0, 0.0, 0.14]):
                self.next_stage("lower_into_box")
        elif self.stage == "lower_into_box":
            action = self.move_to_target_pos(obs, box + [0.0, 0.0, 0.04], gripper=-1.0)
            if self.reached(obs, box + [0.0, 0.0, 0.04]):
                self.next_stage("release_cube")
        elif self.stage == "release_cube":
            action = self.open_gripper()
            if self.stage_step > 10:
                self.next_stage("retract")
        else:
            action = self.move_to_target_pos(obs, box + [0.0, 0.0, 0.16], gripper=1.0)
        self.tick()
        return action

    def is_success(self, obs, info, env) -> bool:
        if super().is_success(obs, info, env):
            return True
        try:
            cube = self.get_obj_pos(obs, self.cube_name)
            box = self.get_obj_pos(obs, self.box_name)
            button_pressed = bool(info.get("button_pressed", False)) if info else False
            return button_pressed and np.linalg.norm(cube[:2] - box[:2]) < 0.055 and cube[2] > box[2]
        except Exception:
            return False

