"""FSM skeleton for inserting a green peg into a matching hole."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from controllers.base_fsm_controller import BaseFSMController


class PegInsertionController(BaseFSMController):
    """Precision insertion controller.

    Expected metadata keys: `peg_name`, `hole_name` or `hole_pos`.
    """

    def reset(self, env: Any, obs: Dict[str, np.ndarray], metadata=None) -> None:
        super().reset(env, obs, metadata)
        self.peg_name = self.metadata.get("peg_name", "green_peg_1")
        self.hole_name = self.metadata.get("hole_name", "wooden_block_hole")
        self.next_stage("move_above_peg")

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        peg = self.get_obj_pos(obs, self.peg_name)
        hole = np.asarray(self.metadata.get("hole_pos", obs.get(f"{self.hole_name}_pos", peg)), dtype=np.float32)
        above_peg = peg + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        grasp = peg + np.array([0.0, 0.0, 0.02], dtype=np.float32)
        above_hole = hole + np.array([0.0, 0.0, 0.14], dtype=np.float32)
        insert = hole + np.array([0.0, 0.0, 0.025], dtype=np.float32)

        if self.stage == "move_above_peg":
            action = self.move_to_target_pos(obs, above_peg, gripper=1.0)
            if self.reached(obs, above_peg):
                self.next_stage("descend_to_peg")
        elif self.stage == "descend_to_peg":
            action = self.move_to_target_pos(obs, grasp, gripper=1.0)
            if self.reached(obs, grasp):
                self.next_stage("grasp_peg")
        elif self.stage == "grasp_peg":
            action = self.close_gripper()
            if self.stage_step > 12:
                self.next_stage("lift_peg")
        elif self.stage == "lift_peg":
            action = self.move_to_target_pos(obs, above_peg, gripper=-1.0)
            if self.reached(obs, above_peg):
                self.next_stage("move_above_hole")
        elif self.stage == "move_above_hole":
            action = self.move_to_target_pos(obs, above_hole, gripper=-1.0)
            if self.reached(obs, above_hole):
                self.next_stage("slow_insert")
        elif self.stage == "slow_insert":
            action = self.move_to_target_pos(obs, insert, gripper=-1.0)
            action[:3] *= 0.35
            if self.reached(obs, insert, tol=0.018):
                self.next_stage("release")
        elif self.stage == "release":
            action = self.open_gripper()
            if self.stage_step > 10:
                self.next_stage("retract")
        else:
            action = self.move_to_target_pos(obs, above_hole, gripper=1.0)
        self.tick()
        return action

    def is_success(self, obs, info, env) -> bool:
        if super().is_success(obs, info, env):
            return True
        try:
            peg = self.get_obj_pos(obs, self.peg_name)
            hole = np.asarray(self.metadata.get("hole_pos", obs[f"{self.hole_name}_pos"]), dtype=np.float32)
            return np.linalg.norm(peg[:2] - hole[:2]) < 0.025 and abs(float(peg[2] - hole[2])) < 0.06
        except Exception:
            return False

