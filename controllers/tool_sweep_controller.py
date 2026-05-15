"""FSM skeleton for tool-mediated sweeping into a dustpan."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from controllers.base_fsm_controller import BaseFSMController


class ToolSweepController(BaseFSMController):
    """Grasp a pusher and sweep a red block into a dustpan target region."""

    def reset(self, env: Any, obs: Dict[str, np.ndarray], metadata=None) -> None:
        super().reset(env, obs, metadata)
        self.pusher_name = self.metadata.get("pusher_name", "pusher_1")
        self.block_name = self.metadata.get("block_name", "red_block_1")
        self.dustpan_name = self.metadata.get("dustpan_name", "dustpan_1")
        self.next_stage("move_above_pusher")

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        pusher = self.get_obj_pos(obs, self.pusher_name)
        block = self.get_obj_pos(obs, self.block_name)
        dustpan = self.get_obj_pos(obs, self.dustpan_name)
        sweep_dir = dustpan[:2] - block[:2]
        norm = np.linalg.norm(sweep_dir) or 1.0
        sweep_dir = sweep_dir / norm
        behind = block.copy()
        behind[:2] -= sweep_dir * 0.08
        behind[2] = block[2] + 0.035
        contact = behind.copy()
        contact[2] = block[2] + 0.01
        finish = dustpan.copy()
        finish[:2] -= sweep_dir * 0.015
        finish[2] = contact[2]

        if self.stage == "move_above_pusher":
            action = self.move_to_target_pos(obs, pusher + [0.0, 0.0, 0.12], gripper=1.0)
            if self.reached(obs, pusher + [0.0, 0.0, 0.12]):
                self.next_stage("grasp_pusher")
        elif self.stage == "grasp_pusher":
            action = self.move_to_target_pos(obs, pusher + [0.0, 0.0, 0.025], gripper=1.0)
            if self.reached(obs, pusher + [0.0, 0.0, 0.025]):
                self.next_stage("close")
        elif self.stage == "close":
            action = self.close_gripper()
            if self.stage_step > 12:
                self.next_stage("move_behind_block")
        elif self.stage == "move_behind_block":
            action = self.move_to_target_pos(obs, behind + [0.0, 0.0, 0.08], gripper=-1.0)
            if self.reached(obs, behind + [0.0, 0.0, 0.08]):
                self.next_stage("lower_to_contact")
        elif self.stage == "lower_to_contact":
            action = self.move_to_target_pos(obs, contact, gripper=-1.0)
            if self.reached(obs, contact):
                self.next_stage("sweep")
        elif self.stage == "sweep":
            action = self.move_to_target_pos(obs, finish, gripper=-1.0)
            action[:3] *= 0.55
            if self.is_success(obs, {}, None) or self.reached(obs, finish, tol=0.035):
                self.next_stage("retract")
        else:
            action = self.move_to_target_pos(obs, finish + [0.0, 0.0, 0.12], gripper=-1.0)
        self.tick()
        return action

    def is_success(self, obs, info, env) -> bool:
        if env is not None and super().is_success(obs, info, env):
            return True
        try:
            block = self.get_obj_pos(obs, self.block_name)
            dustpan = self.get_obj_pos(obs, self.dustpan_name)
            return np.linalg.norm(block[:2] - dustpan[:2]) < float(self.metadata.get("dustpan_radius", 0.07))
        except Exception:
            return False

