"""Stretch FSM skeleton for hanging a ring on a hook."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from controllers.base_fsm_controller import BaseFSMController


class RingHookController(BaseFSMController):
    """Place a grasped ring over a hook and release."""

    def reset(self, env: Any, obs: Dict[str, np.ndarray], metadata=None) -> None:
        super().reset(env, obs, metadata)
        self.ring_name = self.metadata.get("ring_name", "ring_1")
        self.hook_name = self.metadata.get("hook_name", "hook_1")
        self.next_stage("move_above_ring")

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        ring = self.get_obj_pos(obs, self.ring_name)
        hook = self.get_obj_pos(obs, self.hook_name)
        pre_hook = hook + np.array([-0.035, 0.0, 0.08], dtype=np.float32)
        place = hook + np.array([0.0, 0.0, 0.015], dtype=np.float32)
        if self.stage == "move_above_ring":
            action = self.move_to_target_pos(obs, ring + [0.0, 0.0, 0.12], gripper=1.0)
            if self.reached(obs, ring + [0.0, 0.0, 0.12]):
                self.next_stage("grasp_ring")
        elif self.stage == "grasp_ring":
            action = self.move_to_target_pos(obs, ring + [0.0, 0.0, 0.02], gripper=1.0)
            if self.reached(obs, ring + [0.0, 0.0, 0.02]):
                self.next_stage("close")
        elif self.stage == "close":
            action = self.close_gripper()
            if self.stage_step > 12:
                self.next_stage("lift")
        elif self.stage == "lift":
            action = self.move_to_target_pos(obs, ring + [0.0, 0.0, 0.14], gripper=-1.0)
            if self.stage_step > 18:
                self.next_stage("pre_hook")
        elif self.stage == "pre_hook":
            action = self.move_to_target_pos(obs, pre_hook, gripper=-1.0)
            if self.reached(obs, pre_hook):
                self.next_stage("place_on_hook")
        elif self.stage == "place_on_hook":
            action = self.move_to_target_pos(obs, place, gripper=-1.0)
            action[:3] *= 0.4
            if self.reached(obs, place, tol=0.02):
                self.next_stage("release")
        elif self.stage == "release":
            action = self.open_gripper()
            if self.stage_step > 10:
                self.next_stage("retract")
        else:
            action = self.move_to_target_pos(obs, hook + [0.0, 0.0, 0.12], gripper=1.0)
        self.tick()
        return action

    def is_success(self, obs, info, env) -> bool:
        if super().is_success(obs, info, env):
            return True
        try:
            ring = self.get_obj_pos(obs, self.ring_name)
            hook = self.get_obj_pos(obs, self.hook_name)
            return np.linalg.norm(ring[:2] - hook[:2]) < 0.04 and abs(float(ring[2] - hook[2])) < 0.08
        except Exception:
            return False

