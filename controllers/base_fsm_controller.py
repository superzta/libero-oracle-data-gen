"""Finite-state-machine helpers for ground-truth LIBERO oracle controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BaseFSMController:
    """Base class for state-based Cartesian controllers.

    The action convention targets LIBERO / robosuite OSC_POSE environments:
    delta xyz, delta rotation axis-angle, gripper command.
    """

    pos_gain: float = 8.0
    max_pos_delta: float = 0.08
    max_rot_delta: float = 0.25
    position_tolerance: float = 0.025
    stage: str = "reset"
    step_count: int = 0
    stage_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reset(self, env: Any, obs: Dict[str, np.ndarray], metadata: Optional[Dict[str, Any]] = None) -> None:
        self.stage = "start"
        self.step_count = 0
        self.stage_step = 0
        self.metadata = metadata or {}

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def is_success(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], env: Any) -> bool:
        try:
            return bool(env.check_success())
        except Exception:
            return False

    def get_debug_state(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "step_count": self.step_count,
            "stage_step": self.stage_step,
        }

    def next_stage(self, stage: str) -> None:
        self.stage = stage
        self.stage_step = 0

    def tick(self) -> None:
        self.step_count += 1
        self.stage_step += 1

    def get_eef_pos(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float32)

    def get_obj_pos(self, obs: Dict[str, np.ndarray], name: str) -> np.ndarray:
        key = name if name.endswith("_pos") else f"{name}_pos"
        if key not in obs:
            raise KeyError(f"Observation key not found: {key}")
        return np.asarray(obs[key], dtype=np.float32)

    def distance(self, obs: Dict[str, np.ndarray], target_pos: np.ndarray) -> float:
        return float(np.linalg.norm(self.get_eef_pos(obs) - np.asarray(target_pos)))

    def reached(self, obs: Dict[str, np.ndarray], target_pos: np.ndarray, tol: Optional[float] = None) -> bool:
        return self.distance(obs, target_pos) <= (tol or self.position_tolerance)

    def make_action(self, xyz_delta: np.ndarray, gripper: float = 0.0, rot_delta: Optional[np.ndarray] = None) -> np.ndarray:
        rot = np.zeros(3, dtype=np.float32) if rot_delta is None else np.asarray(rot_delta, dtype=np.float32)
        xyz = np.asarray(xyz_delta, dtype=np.float32)
        action = np.concatenate([xyz, rot, np.asarray([gripper], dtype=np.float32)])
        return self.clip_action(action)

    def move_to_target_pos(self, obs: Dict[str, np.ndarray], target_pos: np.ndarray, gripper: float = 0.0) -> np.ndarray:
        delta = (np.asarray(target_pos, dtype=np.float32) - self.get_eef_pos(obs)) * self.pos_gain
        return self.make_action(delta, gripper=gripper)

    def open_gripper(self) -> np.ndarray:
        return self.make_action(np.zeros(3, dtype=np.float32), gripper=-1.0)

    def close_gripper(self) -> np.ndarray:
        return self.make_action(np.zeros(3, dtype=np.float32), gripper=1.0)

    def wait_steps(self, n_steps: int, gripper: float = 0.0) -> np.ndarray:
        if self.stage_step >= n_steps:
            self.next_stage("done")
        return self.make_action(np.zeros(3, dtype=np.float32), gripper=gripper)

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        clipped = np.asarray(action, dtype=np.float32).copy()
        clipped[:3] = np.clip(clipped[:3], -self.max_pos_delta, self.max_pos_delta)
        clipped[3:6] = np.clip(clipped[3:6], -self.max_rot_delta, self.max_rot_delta)
        clipped[6] = np.clip(clipped[6], -1.0, 1.0)
        return clipped


class NoOpController(BaseFSMController):
    """Controller used to verify reset, stepping, serialization, and reporting."""

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        self.tick()
        return self.make_action(np.zeros(3, dtype=np.float32), gripper=0.0)
