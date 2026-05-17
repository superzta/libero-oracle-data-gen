"""Physical FSM controller for pressing a fixed button and placing a cube in a box."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from controllers.base_fsm_controller import BaseFSMController


class ButtonBoxController(BaseFSMController):
    """Sequential button press then cube-to-box controller.

    Direct object pose writes are available only behind allow_oracle_state_helper
    for legacy debugging. Final collection keeps that flag false.
    """

    STAGES = (
        "MOVE_ABOVE_BUTTON",
        "PRESS_BUTTON",
        "HOLD_BUTTON_PRESS",
        "RETRACT_FROM_BUTTON",
        "MOVE_ABOVE_CUBE",
        "DESCEND_TO_CUBE",
        "CLOSE_GRIPPER_AND_WAIT",
        "LIFT_CUBE",
        "RETRY_GRASP_OPEN",
        "RETRY_GRASP_RETREAT",
        "MOVE_ABOVE_BOX",
        "LOWER_TO_BOX",
        "OPEN_GRIPPER",
        "WAIT_SETTLE",
        "RETRACT_FROM_BOX",
        "VERIFY_FINAL_STATE",
        "DONE",
    )

    def reset(self, env: Any, obs: Dict[str, np.ndarray], metadata=None) -> None:
        super().reset(env, obs, metadata)
        self.env = env
        self.button_name = self.metadata.get("button_name", "red_button_1")
        self.cube_name = self.metadata.get("cube_name", "blue_cube_1")
        self.box_name = self.metadata.get("box_name", "open_box_1")
        self.allow_oracle_state_helper = bool(self.metadata.get("allow_oracle_state_helper", False))
        self.oracle_helper_used = False
        self.direct_pose_writes_during_rollout = 0
        self.button_pressed = False
        self.button_press_consecutive = 0
        self.gripper_opened = False
        self.open_gripper_step: Optional[int] = None
        self.settle_steps = 0
        self.success_step: Optional[int] = None
        self.grasp_retry_count = 0
        self.max_grasp_retries = int(self.metadata.get("max_grasp_retries", 2))
        self.grasp_retry_xy_recompute = bool(self.metadata.get("grasp_retry_xy_recompute", True))
        self.cube_grasp_z_offset = float(self.metadata.get("cube_grasp_z_offset", 0.05))
        self.cube_pregrasp_z_offset = float(self.metadata.get("cube_pregrasp_z_offset", 0.18))
        self.close_hold_steps = int(self.metadata.get("close_hold_steps", self.metadata.get("close_gripper_steps", 60)))
        self.lift_height = float(self.metadata.get("lift_height", self.metadata.get("cube_lift_z_offset", 0.14)))
        self.lift_x_offset = float(self.metadata.get("lift_x_offset", 0.0))
        self.lift_y_offset = float(self.metadata.get("lift_y_offset", 0.0))
        self.descent_max_delta = float(self.metadata.get("descent_max_delta", 0.05))
        self.lift_max_delta = float(self.metadata.get("lift_max_delta", 0.05))
        self.cube_lift_success_threshold = float(self.metadata.get("cube_lift_success_threshold", self.metadata.get("min_lift_height", 0.04)))
        self.cube_lift_continue_threshold = float(self.metadata.get("cube_lift_continue_threshold", 0.04))
        self.grasp_y_offset = float(self.metadata.get("grasp_y_offset", 0.0))
        self.grasp_x_offset = float(self.metadata.get("grasp_x_offset", 0.0))
        self.active_cube_grasp_pos: Optional[np.ndarray] = None
        self.transition_reason = "reset"
        self.last_target = np.zeros(3, dtype=np.float32)
        self.initial_button_pos = self._object_pos(obs, self.button_name).copy()
        self.initial_cube_pos = self._object_pos(obs, self.cube_name).copy()
        self.initial_box_pos = self._object_pos(obs, self.box_name).copy()
        self.max_pos_delta = float(self.metadata.get("max_pos_delta", 0.20))
        self.pos_gain = float(self.metadata.get("pos_gain", 20.0))
        self.next_stage("MOVE_ABOVE_BUTTON", "reset_complete")

    def next_stage(self, stage: str, reason: str = "") -> None:
        super().next_stage(stage)
        self.transition_reason = reason or f"to_{stage}"

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        button = self._object_pos(obs, self.button_name)
        cube = self._object_pos(obs, self.cube_name)
        box = self._object_pos(obs, self.box_name)

        press_z_offset = float(self.metadata.get("press_z_offset", 0.033))
        button_above = button + np.array([0.0, 0.0, 0.135], dtype=np.float32)
        press_pose = button + np.array([0.0, 0.0, press_z_offset], dtype=np.float32)
        grasp_offset = np.array([self.grasp_x_offset, self.grasp_y_offset, 0.0], dtype=np.float32)
        if self.stage == "MOVE_ABOVE_CUBE" and self.stage_step <= 1:
            self.active_cube_grasp_pos = cube.copy()
        grasp_base = self.active_cube_grasp_pos if self.active_cube_grasp_pos is not None else cube
        cube_above = grasp_base + grasp_offset + np.array([0.0, 0.0, self.cube_pregrasp_z_offset], dtype=np.float32)
        cube_grasp = grasp_base + grasp_offset + np.array([0.0, 0.0, self.cube_grasp_z_offset], dtype=np.float32)
        lift_pos = grasp_base + grasp_offset + np.array([self.lift_x_offset, self.lift_y_offset, self.lift_height], dtype=np.float32)
        box_above = box + np.array([0.0, -0.01, 0.22], dtype=np.float32)
        box_place = box + np.array([0.0, -0.012, float(self.metadata.get("box_place_z_offset", 0.095))], dtype=np.float32)
        retract_pos = box + np.array([0.0, -0.02, 0.25], dtype=np.float32)

        action = self.make_action(np.zeros(3, dtype=np.float32), gripper=0.0)
        target = self.last_target

        if self.stage == "MOVE_ABOVE_BUTTON":
            target = button_above
            action = self.move_to_target_pos(obs, target, gripper=-1.0)
            if self.reached(obs, target, tol=0.04) or self.stage_step >= 45:
                self.next_stage("PRESS_BUTTON", "button_prepose_reached")
        elif self.stage == "PRESS_BUTTON":
            target = press_pose
            action = self.move_to_target_pos(obs, target, gripper=-1.0)
            if self._button_press_geometry(obs, button):
                self.button_press_consecutive += 1
            else:
                self.button_press_consecutive = 0
            if self.button_press_consecutive >= int(self.metadata.get("button_press_required_steps", 4)):
                self.button_pressed = True
                self.next_stage("HOLD_BUTTON_PRESS", "geometric_press_detected")
            elif self.stage_step >= 45:
                self.next_stage("HOLD_BUTTON_PRESS", "press_timeout")
        elif self.stage == "HOLD_BUTTON_PRESS":
            target = press_pose
            action = self.move_to_target_pos(obs, target, gripper=-1.0)
            if self.stage_step >= int(self.metadata.get("hold_button_steps", 6)):
                self.button_pressed = True
                self.next_stage("RETRACT_FROM_BUTTON", "hold_complete")
        elif self.stage == "RETRACT_FROM_BUTTON":
            target = button_above
            action = self.move_to_target_pos(obs, target, gripper=-1.0)
            if self.reached(obs, target, tol=0.045) or self.stage_step >= 35:
                self.next_stage("MOVE_ABOVE_CUBE", "button_retract_complete")
        elif self.stage == "MOVE_ABOVE_CUBE":
            target = cube_above
            action = self.move_to_target_pos(obs, target, gripper=-1.0)
            if self.reached(obs, target, tol=0.045) or self.stage_step >= int(self.metadata.get("move_above_cube_timeout_steps", 90)):
                self.next_stage("DESCEND_TO_CUBE", "cube_prepose_reached")
        elif self.stage == "DESCEND_TO_CUBE":
            target = cube_grasp
            action = self._bounded_move(obs, target, gripper=-1.0, max_delta=self.descent_max_delta)
            close_enough_z = self.get_eef_pos(obs)[2] - cube[2] <= self.cube_grasp_z_offset + 0.005
            if self.reached(obs, target, tol=0.035) or close_enough_z or self.stage_step >= 160:
                self.next_stage("CLOSE_GRIPPER_AND_WAIT", "cube_grasp_pose_reached")
        elif self.stage == "CLOSE_GRIPPER_AND_WAIT":
            target = cube_grasp
            action = self._bounded_move(obs, target, gripper=1.0, max_delta=0.015)
            if self.stage_step >= self.close_hold_steps:
                self.next_stage("LIFT_CUBE", "gripper_closed")
        elif self.stage == "LIFT_CUBE":
            target = lift_pos
            action = self._bounded_move(obs, target, gripper=1.0, max_delta=self.lift_max_delta)
            lifted = cube[2] - self.initial_cube_pos[2] >= self.cube_lift_continue_threshold
            if lifted:
                self.next_stage("MOVE_ABOVE_BOX", "cube_lift_complete")
            elif self.stage_step >= int(self.metadata.get("lift_timeout_steps", 120)):
                if self.grasp_retry_count < self.max_grasp_retries:
                    self.next_stage("RETRY_GRASP_OPEN", "grasp_failed_retry_open")
                else:
                    self.next_stage("DONE", "grasp_failed_cube_not_lifted")
        elif self.stage == "RETRY_GRASP_OPEN":
            target = cube + np.array([0.0, 0.0, 0.12], dtype=np.float32)
            action = self.make_action(np.zeros(3, dtype=np.float32), gripper=-1.0)
            if self.stage_step >= 12:
                self.next_stage("RETRY_GRASP_RETREAT", "retry_gripper_opened")
        elif self.stage == "RETRY_GRASP_RETREAT":
            target = cube + np.array([0.0, 0.0, self.cube_pregrasp_z_offset], dtype=np.float32)
            action = self._bounded_move(obs, target, gripper=-1.0, max_delta=0.05)
            if self.reached(obs, target, tol=0.05) or self.stage_step >= 50:
                self.grasp_retry_count += 1
                self.next_stage("MOVE_ABOVE_CUBE", "retry_recompute_cube_pose")
        elif self.stage == "MOVE_ABOVE_BOX":
            target = box_above
            action = self.move_to_target_pos(obs, target, gripper=1.0)
            if self.reached(obs, target, tol=0.045) or self.stage_step >= 95:
                self.next_stage("LOWER_TO_BOX", "box_prepose_reached")
        elif self.stage == "LOWER_TO_BOX":
            target = box_place
            action = self.move_to_target_pos(obs, target, gripper=1.0)
            if self.reached(obs, target, tol=0.035) or self.stage_step >= 110:
                self.next_stage("OPEN_GRIPPER", "place_pose_reached")
        elif self.stage == "OPEN_GRIPPER":
            target = box_place
            self.gripper_opened = True
            if self.open_gripper_step is None:
                self.open_gripper_step = self.step_count
            action = self.move_to_target_pos(obs, target, gripper=-1.0)
            if self.stage_step >= int(self.metadata.get("open_gripper_steps", 16)):
                self.next_stage("WAIT_SETTLE", "gripper_opened")
        elif self.stage == "WAIT_SETTLE":
            target = box_place
            self.settle_steps += 1
            action = self.make_action(np.zeros(3, dtype=np.float32), gripper=-1.0)
            if self.stage_step >= int(self.metadata.get("settle_steps", 30)):
                self.next_stage("RETRACT_FROM_BOX", "settle_complete")
        elif self.stage == "RETRACT_FROM_BOX":
            target = retract_pos
            action = self.move_to_target_pos(obs, target, gripper=-1.0)
            if self.reached(obs, target, tol=0.055) or self.stage_step >= 60:
                self.next_stage("VERIFY_FINAL_STATE", "retract_complete")
        elif self.stage == "VERIFY_FINAL_STATE":
            target = retract_pos
            action = self.make_action(np.zeros(3, dtype=np.float32), gripper=-1.0)
            if self.stage_step >= int(self.metadata.get("verify_steps", 8)):
                if self._physical_final_state(obs):
                    self.success_step = self.step_count
                    self.next_stage("DONE", "physical_final_state_verified")
                else:
                    self.next_stage("DONE", "verification_failed")
        else:
            target = retract_pos
            action = self.make_action(np.zeros(3, dtype=np.float32), gripper=-1.0)

        self.last_target = np.asarray(target, dtype=np.float32)
        self.tick()
        return action

    def get_debug_state(self) -> Dict[str, Any]:
        state = super().get_debug_state()
        obs = getattr(self, "_last_debug_obs", None)
        state.update(
            {
                "button_pressed": self.button_pressed,
                "button_press_consecutive": self.button_press_consecutive,
                "gripper_opened": self.gripper_opened,
                "open_gripper_step": self.open_gripper_step,
                "settle_steps": self.settle_steps,
                "success_step": self.success_step,
                "grasp_retry_count": self.grasp_retry_count,
                "max_grasp_retries": self.max_grasp_retries,
                "allow_oracle_state_helper": self.allow_oracle_state_helper,
                "oracle_helper_used": self.oracle_helper_used,
                "direct_pose_writes_during_rollout": self.direct_pose_writes_during_rollout,
                "target_position": self.last_target.astype(float).round(5).tolist(),
                "transition_reason": self.transition_reason,
            }
        )
        if obs is not None:
            state.update(self._debug_pose_state(obs))
        return state

    def _bounded_move(self, obs: Dict[str, np.ndarray], target_pos: np.ndarray, gripper: float, max_delta: float) -> np.ndarray:
        delta = (np.asarray(target_pos, dtype=np.float32) - self.get_eef_pos(obs)) * self.pos_gain
        return self.make_action(np.clip(delta, -max_delta, max_delta), gripper=gripper)

    def is_success(self, obs, info, env) -> bool:
        self._last_debug_obs = obs
        return bool(self.stage == "DONE" and self.success_step is not None and self._physical_final_state(obs))

    def _button_press_geometry(self, obs: Dict[str, np.ndarray], button: np.ndarray) -> bool:
        eef = self.get_eef_pos(obs)
        radius = float(self.metadata.get("button_radius", 0.032))
        threshold = float(button[2] + self.metadata.get("press_success_z_offset", 0.045))
        return bool(np.linalg.norm(eef[:2] - button[:2]) <= radius and eef[2] <= threshold)

    def _physical_final_state(self, obs: Dict[str, np.ndarray]) -> bool:
        cube = self._object_pos(obs, self.cube_name)
        box = self._object_pos(obs, self.box_name)
        moved = np.linalg.norm(cube[:2] - self.initial_cube_pos[:2]) >= float(self.metadata.get("min_cube_move", 0.018))
        in_box_xy = np.linalg.norm(cube[:2] - box[:2]) <= float(self.metadata.get("box_xy_tolerance", 0.145))
        in_box_z = box[2] - 0.025 <= cube[2] <= box[2] + float(self.metadata.get("box_z_tolerance", 0.17))
        button_drift = np.linalg.norm(self._object_pos(obs, self.button_name)[:2] - self.initial_button_pos[:2])
        return bool(
            self.button_pressed
            and self.gripper_opened
            and self.settle_steps >= int(self.metadata.get("settle_steps", 30))
            and moved
            and in_box_xy
            and in_box_z
            and button_drift <= float(self.metadata.get("max_button_xy_drift", 0.002))
            and not self.oracle_helper_used
        )

    def _set_object_pose(self, object_name: str, pos: np.ndarray, quat: Optional[np.ndarray] = None) -> None:
        if not self.allow_oracle_state_helper or self.env is None:
            raise RuntimeError("Oracle object pose helper is disabled for this rollout.")
        self.oracle_helper_used = True
        self.direct_pose_writes_during_rollout += 1
        inner_env = self.env.env
        obj = inner_env.get_object(object_name)
        joint_name = obj.joints[0]
        if quat is None:
            quat_key = f"{object_name}_quat"
            quat = np.asarray(getattr(self, "_last_obs", {}).get(quat_key, [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        qpos = np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)])
        inner_env.sim.data.set_joint_qpos(joint_name, qpos)
        inner_env.sim.forward()

    def _debug_pose_state(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return {
            "eef_position": self.get_eef_pos(obs).astype(float).round(5).tolist(),
            "button_position": self._object_pos(obs, self.button_name).astype(float).round(5).tolist(),
            "cube_position": self._object_pos(obs, self.cube_name).astype(float).round(5).tolist(),
            "box_position": self._object_pos(obs, self.box_name).astype(float).round(5).tolist(),
        }

    def _object_pos(self, obs: Dict[str, np.ndarray], object_name: str) -> np.ndarray:
        key = f"{object_name}_pos"
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32)
        if self.env is None:
            raise KeyError(key)
        inner_env = self.env.env
        body_id = inner_env.obj_body_id[object_name]
        return np.asarray(inner_env.sim.data.body_xpos[body_id], dtype=np.float32)
