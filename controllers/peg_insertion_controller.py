"""Physical FSM controller for inserting a green peg into a wooden socket."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from controllers.base_fsm_controller import BaseFSMController


class PegInsertionController(BaseFSMController):
    """State-based peg insertion controller with strict physical success."""

    STAGES = (
        "MOVE_ABOVE_PEG",
        "DESCEND_TO_PEG",
        "CLOSE_GRIPPER_AND_WAIT",
        "LIFT_PEG",
        "MOVE_ABOVE_HOLE",
        "ALIGN_WITH_HOLE",
        "LOWER_INSERT",
        "HOLD_INSERT",
        "OPEN_GRIPPER",
        "WAIT_SETTLE",
        "RETRACT",
        "VERIFY_FINAL_STATE",
        "DONE",
    )

    def reset(self, env: Any, obs: Dict[str, np.ndarray], metadata=None) -> None:
        super().reset(env, obs, metadata)
        self.env = env
        self.peg_name = self.metadata.get("peg_name", "green_peg_1")
        self.block_name = self.metadata.get("block_name", "wooden_hole_block_1")
        self.allow_oracle_state_helper = bool(self.metadata.get("allow_oracle_state_helper", False))
        self.oracle_helper_used = False
        self.direct_pose_writes_during_rollout = 0
        self.gripper_opened = False
        self.open_gripper_step: Optional[int] = None
        self.settle_steps = 0
        self.success_step: Optional[int] = None
        self.initial_peg_pos = self._object_pos(obs, self.peg_name).copy()
        self.initial_block_pos = self._object_pos(obs, self.block_name).copy()
        self.active_grasp_xy = self.initial_peg_pos[:2].copy()
        self.active_grasp_z = float(self.initial_peg_pos[2])
        self.last_target = self.initial_peg_pos.copy()
        self.transition_reason = "reset"
        self.pos_gain = float(self.metadata.get("pos_gain", 20.0))
        self.max_pos_delta = float(self.metadata.get("max_pos_delta", 0.18))
        self.next_stage("MOVE_ABOVE_PEG", "reset_complete")

    def next_stage(self, stage: str, reason: str = "") -> None:
        super().next_stage(stage)
        self.transition_reason = reason or f"to_{stage}"

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        self._last_debug_obs = obs
        peg = self._object_pos(obs, self.peg_name)
        block = self._object_pos(obs, self.block_name)
        hole = self._hole_center(block)

        grasp_xy = self.active_grasp_xy
        above_peg = np.array([grasp_xy[0], grasp_xy[1], self.active_grasp_z + float(self.metadata.get("peg_pregrasp_z_offset", 0.135))], dtype=np.float32)
        grasp = np.array([grasp_xy[0], grasp_xy[1], self.active_grasp_z + float(self.metadata.get("peg_grasp_z_offset", 0.005))], dtype=np.float32)
        lift_pos = np.array([grasp_xy[0], grasp_xy[1], self.active_grasp_z + float(self.metadata.get("lift_z_offset", 0.115))], dtype=np.float32)
        above_hole = hole + np.array([0.0, 0.0, float(self.metadata.get("hole_above_z_offset", 0.160))], dtype=np.float32)
        align = hole + np.array([0.0, 0.0, float(self.metadata.get("hole_align_z_offset", 0.090))], dtype=np.float32)
        insert = hole + np.array([0.0, 0.0, float(self.metadata.get("insert_z_offset", 0.040))], dtype=np.float32)
        retract = self._retract_target(hole)
        above_hole_with_peg_offset = self._target_with_held_peg_offset(obs, peg, above_hole)
        insertion_bias = self._insertion_direction_bias(hole)
        align_with_peg_offset = self._target_with_held_peg_offset(obs, peg, align)
        insert_with_peg_offset = self._target_with_held_peg_offset(obs, peg, insert, insertion_bias)

        action = self.make_action(np.zeros(3, dtype=np.float32), gripper=-1.0)
        target = self.last_target

        if self.stage == "MOVE_ABOVE_PEG":
            target = above_peg
            action = self._bounded_move(obs, target, gripper=-1.0, max_delta=0.22)
            if self.reached(obs, target, tol=0.045) or self.stage_step >= 120:
                self.next_stage("DESCEND_TO_PEG", "peg_prepose_reached")
        elif self.stage == "DESCEND_TO_PEG":
            target = grasp
            action = self._bounded_move(obs, target, gripper=-1.0, max_delta=0.055)
            close_enough_z = self.get_eef_pos(obs)[2] - peg[2] <= float(self.metadata.get("peg_grasp_z_offset", 0.005)) + 0.020
            if close_enough_z or self.stage_step >= 180:
                self.next_stage("CLOSE_GRIPPER_AND_WAIT", "peg_grasp_pose_reached")
        elif self.stage == "CLOSE_GRIPPER_AND_WAIT":
            target = grasp
            action = self._bounded_move(obs, target, gripper=1.0, max_delta=0.010)
            if self.stage_step >= int(self.metadata.get("close_gripper_steps", 40)):
                self.next_stage("LIFT_PEG", "gripper_closed")
        elif self.stage == "LIFT_PEG":
            target = lift_pos
            action = self._bounded_move(obs, target, gripper=1.0, max_delta=0.065)
            lifted = peg[2] - self.initial_peg_pos[2] >= float(self.metadata.get("peg_lift_continue_threshold", 0.045))
            if lifted:
                self.next_stage("MOVE_ABOVE_HOLE", "peg_lift_complete")
            elif self.stage_step >= 140:
                self.next_stage("DONE", "grasp_failed_peg_not_lifted")
        elif self.stage == "MOVE_ABOVE_HOLE":
            target = above_hole_with_peg_offset
            action = self._bounded_move(obs, target, gripper=1.0, max_delta=0.080)
            if self.reached(obs, target, tol=0.065) or self.stage_step >= 165:
                self.next_stage("ALIGN_WITH_HOLE", "hole_prepose_reached")
        elif self.stage == "ALIGN_WITH_HOLE":
            target = align_with_peg_offset
            action = self._bounded_move(obs, target, gripper=1.0, max_delta=0.050)
            if self.reached(obs, target, tol=0.040) or self.stage_step >= 105:
                self.next_stage("LOWER_INSERT", "aligned_above_socket")
        elif self.stage == "LOWER_INSERT":
            target = insert_with_peg_offset
            action = self._bounded_move(obs, target, gripper=1.0, max_delta=0.026)
            if self.reached(obs, target, tol=0.040) or self.stage_step >= 140:
                self.next_stage("HOLD_INSERT", "insert_pose_reached")
        elif self.stage == "HOLD_INSERT":
            target = insert_with_peg_offset
            action = self._bounded_move(obs, target, gripper=1.0, max_delta=0.006)
            if self.stage_step >= int(self.metadata.get("hold_insert_steps", 8)):
                self.next_stage("OPEN_GRIPPER", "insert_hold_complete")
        elif self.stage == "OPEN_GRIPPER":
            target = insert_with_peg_offset
            self.gripper_opened = True
            if self.open_gripper_step is None:
                self.open_gripper_step = self.step_count
            action = self._bounded_move(obs, target, gripper=-1.0, max_delta=0.006)
            if self.stage_step >= int(self.metadata.get("open_gripper_steps", 12)):
                self.next_stage("WAIT_SETTLE", "gripper_opened")
        elif self.stage == "WAIT_SETTLE":
            target = insert
            self.settle_steps += 1
            action = self.make_action(np.zeros(3, dtype=np.float32), gripper=-1.0)
            if self.stage_step >= int(self.metadata.get("settle_steps", 35)):
                self.next_stage("RETRACT", "settle_complete")
        elif self.stage == "RETRACT":
            target = retract
            action = self._bounded_move(obs, target, gripper=-1.0, max_delta=0.120)
            if self.reached(obs, target, tol=0.070) or self.stage_step >= 65:
                self.next_stage("VERIFY_FINAL_STATE", "retract_complete")
        elif self.stage == "VERIFY_FINAL_STATE":
            target = retract
            action = self.make_action(np.zeros(3, dtype=np.float32), gripper=-1.0)
            if self.stage_step >= int(self.metadata.get("verify_steps", 5)):
                if self._physical_final_state(obs):
                    self.success_step = self.step_count
                    self.next_stage("DONE", "physical_final_state_verified")
                else:
                    self.next_stage("DONE", "verification_failed")
        else:
            target = retract
            action = self.make_action(np.zeros(3, dtype=np.float32), gripper=-1.0)

        self.last_target = np.asarray(target, dtype=np.float32)
        self.tick()
        return action

    def get_debug_state(self) -> Dict[str, Any]:
        state = super().get_debug_state()
        obs = getattr(self, "_last_debug_obs", None)
        state.update(
            {
                "gripper_opened": self.gripper_opened,
                "open_gripper_step": self.open_gripper_step,
                "settle_steps": self.settle_steps,
                "success_step": self.success_step,
                "allow_oracle_state_helper": self.allow_oracle_state_helper,
                "oracle_helper_used": self.oracle_helper_used,
                "direct_pose_writes_during_rollout": self.direct_pose_writes_during_rollout,
                "target_position": self.last_target.astype(float).round(5).tolist(),
                "transition_reason": self.transition_reason,
            }
        )
        if obs is not None:
            peg = self._object_pos(obs, self.peg_name)
            block = self._object_pos(obs, self.block_name)
            state.update(
                {
                    "eef_position": self.get_eef_pos(obs).astype(float).round(5).tolist(),
                    "peg_position": peg.astype(float).round(5).tolist(),
                    "block_position": block.astype(float).round(5).tolist(),
                    "hole_position": self._hole_center(block).astype(float).round(5).tolist(),
                }
            )
        return state

    def _bounded_move(self, obs: Dict[str, np.ndarray], target_pos: np.ndarray, gripper: float, max_delta: float) -> np.ndarray:
        delta = (np.asarray(target_pos, dtype=np.float32) - self.get_eef_pos(obs)) * self.pos_gain
        return self.make_action(np.clip(delta, -max_delta, max_delta), gripper=gripper)

    def _target_with_held_peg_offset(
        self,
        obs: Dict[str, np.ndarray],
        peg_pos: np.ndarray,
        target_for_peg: np.ndarray,
        xy_bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Move the wrist so the physically held peg center, not the wrist, reaches the socket."""
        target = np.asarray(target_for_peg, dtype=np.float32).copy()
        eef_pos = self.get_eef_pos(obs)
        xy_offset = np.clip(eef_pos[:2] - peg_pos[:2], -0.065, 0.065)
        target[:2] += xy_offset
        if xy_bias is not None:
            target[:2] += np.asarray(xy_bias, dtype=np.float32)
        return target

    def _insertion_direction_bias(self, hole_pos: np.ndarray) -> np.ndarray:
        direction = np.asarray(hole_pos[:2] - self.initial_peg_pos[:2], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return direction / norm * float(self.metadata.get("insert_xy_overdrive", 0.004))

    def _retract_target(self, hole_pos: np.ndarray) -> np.ndarray:
        target = np.asarray(hole_pos, dtype=np.float32).copy()
        away = self.initial_peg_pos[:2] - target[:2]
        norm = float(np.linalg.norm(away))
        if norm < 1e-6:
            away = np.array([0.0, -1.0], dtype=np.float32)
        else:
            away = away / norm
        target[:2] += away * float(self.metadata.get("retract_xy_distance", 0.115))
        target[0] = float(np.clip(target[0], -0.140, 0.140))
        target[1] = float(np.clip(target[1], -0.155, 0.255))
        target[2] += float(self.metadata.get("retract_z_offset", 0.190))
        return target

    def is_success(self, obs, info, env) -> bool:
        self._last_debug_obs = obs
        return bool(self.stage == "DONE" and self.success_step is not None and self._physical_final_state(obs))

    def _physical_final_state(self, obs: Dict[str, np.ndarray]) -> bool:
        peg = self._object_pos(obs, self.peg_name)
        block = self._object_pos(obs, self.block_name)
        hole = self._hole_center(block)
        moved = np.linalg.norm(peg[:2] - self.initial_peg_pos[:2]) >= float(self.metadata.get("min_peg_move", 0.050))
        in_hole_xy = np.linalg.norm(peg[:2] - hole[:2]) <= float(self.metadata.get("hole_xy_tolerance", 0.018))
        in_hole_z = abs(float(peg[2] - (hole[2] + float(self.metadata.get("final_peg_z_above_hole", 0.034))))) <= float(self.metadata.get("hole_z_tolerance", 0.050))
        upright = self._peg_upright_score(obs) >= float(self.metadata.get("min_peg_upright_score", 0.94))
        stable_block = np.linalg.norm(block[:2] - self.initial_block_pos[:2]) <= float(self.metadata.get("max_block_xy_drift", 0.018))
        return bool(
            self.gripper_opened
            and self.settle_steps >= int(self.metadata.get("settle_steps", 35))
            and moved
            and in_hole_xy
            and in_hole_z
            and upright
            and stable_block
            and not self.oracle_helper_used
            and self.direct_pose_writes_during_rollout == 0
        )

    def _hole_center(self, block_pos: np.ndarray) -> np.ndarray:
        return np.asarray(block_pos, dtype=np.float32) + np.array([0.0, 0.0, float(self.metadata.get("hole_z_offset", 0.021))], dtype=np.float32)

    def _object_pos(self, obs: Dict[str, np.ndarray], object_name: str) -> np.ndarray:
        key = f"{object_name}_pos"
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32)
        if self.env is None:
            raise KeyError(key)
        return np.asarray(self.env.env.sim.data.body_xpos[self.env.env.obj_body_id[object_name]], dtype=np.float32)

    def _object_quat(self, obs: Dict[str, np.ndarray], object_name: str) -> np.ndarray:
        key = f"{object_name}_quat"
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32)
        if self.env is None:
            raise KeyError(key)
        return np.asarray(self.env.env.sim.data.body_xquat[self.env.env.obj_body_id[object_name]], dtype=np.float32)

    def _peg_upright_score(self, obs: Dict[str, np.ndarray]) -> float:
        quat = self._object_quat(obs, self.peg_name)
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-6:
            return 0.0
        quat = quat / norm
        _, x, y, _ = quat
        return float(1.0 - 2.0 * (x * x + y * y))
