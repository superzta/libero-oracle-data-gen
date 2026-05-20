"""FSM oracle controller for the tool_sweep task.

Stages:
  MOVE_ABOVE_PUSHER → DESCEND_TO_PUSHER → CLOSE_GRIPPER_AND_WAIT →
  LIFT_PUSHER → MOVE_BEHIND_BLOCK → LOWER_TO_SWEEP →
  SWEEP → WAIT_SETTLE → RETRACT → VERIFY_FINAL_STATE → DONE
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from controllers.base_fsm_controller import BaseFSMController


class ToolSweepController(BaseFSMController):
    """Grasp a pusher tool and sweep the red block into the dustpan."""

    STAGES = (
        "MOVE_ABOVE_PUSHER",
        "DESCEND_TO_PUSHER",
        "CLOSE_GRIPPER_AND_WAIT",
        "LIFT_PUSHER",
        "MOVE_BEHIND_BLOCK",
        "LOWER_TO_SWEEP",
        "SWEEP",
        "WAIT_SETTLE",
        "RETRACT",
        "VERIFY_FINAL_STATE",
        "DONE",
    )

    def reset(self, env: Any, obs: Dict[str, np.ndarray], metadata=None) -> None:
        super().reset(env, obs, metadata)
        self.env = env
        self.pusher_name = self.metadata.get("pusher_name", "pusher_tool_1")
        self.block_name = self.metadata.get("block_name", "red_block_1")
        self.dustpan_name = self.metadata.get("dustpan_name", "dustpan_1")

        self.allow_oracle_state_helper = bool(self.metadata.get("allow_oracle_state_helper", False))
        self.oracle_helper_used = False
        self.direct_pose_writes_during_rollout = 0

        self.pos_gain = float(self.metadata.get("pos_gain", 20.0))
        self.max_pos_delta = float(self.metadata.get("max_pos_delta", 0.10))
        self.position_tolerance = float(self.metadata.get("position_tolerance", 0.025))

        # Record initial positions from post-randomization obs
        self.initial_pusher_pos = self._obj_pos(obs, self.pusher_name).copy()
        self.initial_block_pos = self._obj_pos(obs, self.block_name).copy()
        self.initial_dustpan_pos = self._obj_pos(obs, self.dustpan_name).copy()

        # Lane direction (sweep direction: pusher → dustpan)
        diff = self.initial_dustpan_pos[:2] - self.initial_pusher_pos[:2]
        norm = float(np.linalg.norm(diff))
        self.lane_dir_2d = diff / norm if norm > 1e-6 else np.array([0.0, 1.0], dtype=np.float32)

        # State tracking
        self.pusher_grasped = False
        self.gripper_opened = False
        self.max_pusher_z_delta = 0.0     # max z increase of pusher during LIFT stage
        self.sweep_started_step = None
        self.settle_steps_done = 0
        self.success_step: Optional[int] = None
        self.transition_reason = "reset"
        self.last_target = self.initial_pusher_pos.copy()

        self.next_stage("MOVE_ABOVE_PUSHER", "reset_complete")

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        self._last_debug_obs = obs
        pusher = self._obj_pos(obs, self.pusher_name)
        block = self._obj_pos(obs, self.block_name)
        dustpan = self._obj_pos(obs, self.dustpan_name)

        # --- Derived targets ---
        p0 = self.initial_pusher_pos
        b0 = self.initial_block_pos
        d0 = self.initial_dustpan_pos
        lane = np.array([self.lane_dir_2d[0], self.lane_dir_2d[1], 0.0], dtype=np.float32)

        pregrasp_z = float(self.metadata.get("pregrasp_z_offset", 0.120))
        grasp_z = float(self.metadata.get("grasp_z_offset", 0.005))
        lift_z = float(self.metadata.get("lift_z_offset", 0.050))
        # sweep_z_offset keeps blade ~10 mm above table (blade_bottom = pusher_center - 0.060;
        # at sweep_z=0.031 → pusher_center ≈ 0.970 → blade_bottom ≈ 0.910).
        sweep_z = float(self.metadata.get("sweep_z_offset", 0.031))
        behind_dist = float(self.metadata.get("behind_dist", 0.130))
        overshoot = float(self.metadata.get("dustpan_overshoot", 0.040))

        above_pusher = p0 + np.array([0.0, 0.0, pregrasp_z], dtype=np.float32)
        grasp_target = p0 + np.array([0.0, 0.0, grasp_z], dtype=np.float32)
        lift_target = p0 + np.array([0.0, 0.0, grasp_z + lift_z], dtype=np.float32)
        behind_xy = b0[:2] - behind_dist * self.lane_dir_2d
        behind_high = np.array([behind_xy[0], behind_xy[1], grasp_z + lift_z + p0[2]], dtype=np.float32)
        behind_sweep = np.array([behind_xy[0], behind_xy[1], sweep_z + p0[2]], dtype=np.float32)
        sweep_end = np.array([
            d0[0] + overshoot * self.lane_dir_2d[0],
            d0[1] + overshoot * self.lane_dir_2d[1],
            sweep_z + p0[2],
        ], dtype=np.float32)
        retract_target = np.array([
            d0[0] - 0.050 * self.lane_dir_2d[0],
            d0[1] - 0.050 * self.lane_dir_2d[1],
            grasp_z + p0[2] + float(self.metadata.get("retract_z_offset", 0.160)),
        ], dtype=np.float32)

        action = self.make_action(np.zeros(3, dtype=np.float32), gripper=-1.0)
        target = self.last_target

        if self.stage == "MOVE_ABOVE_PUSHER":
            target = above_pusher
            action = self._bounded_move(obs, target, gripper=-1.0, max_delta=0.22)
            # 120 steps is insufficient when the pusher is far in x from the EEF home position;
            # 300 allows full convergence for any reachable workspace position.
            if self.reached(obs, target, tol=0.040) or self.stage_step >= int(self.metadata.get("move_above_pusher_max_steps", 300)):
                self.next_stage("DESCEND_TO_PUSHER", "pregrasp_reached")

        elif self.stage == "DESCEND_TO_PUSHER":
            target = grasp_target
            action = self._bounded_move(obs, target, gripper=-1.0, max_delta=0.050)
            eef_z = self.get_eef_pos(obs)[2]
            close_enough = eef_z - p0[2] <= grasp_z + 0.018
            if close_enough or self.stage_step >= int(self.metadata.get("descend_max_steps", 300)):
                self.next_stage("CLOSE_GRIPPER_AND_WAIT", "descend_complete")

        elif self.stage == "CLOSE_GRIPPER_AND_WAIT":
            target = grasp_target
            action = self._bounded_move(obs, target, gripper=1.0, max_delta=0.010)
            if self.stage_step >= int(self.metadata.get("close_steps", 30)):
                self.next_stage("LIFT_PUSHER", "gripper_closed")

        elif self.stage == "LIFT_PUSHER":
            target = lift_target
            action = self._bounded_move(obs, target, gripper=1.0, max_delta=0.060)
            pusher_z_delta = float(pusher[2] - p0[2])
            self.max_pusher_z_delta = max(self.max_pusher_z_delta, pusher_z_delta)
            # Always run the full lift budget so the pusher reaches true lift height.
            # Early exit via distance tolerance fires before the pusher is properly lifted.
            if self.stage_step >= int(self.metadata.get("lift_max_steps", 80)):
                if self.max_pusher_z_delta >= float(self.metadata.get("lift_threshold", 0.008)):
                    self.pusher_grasped = True
                    self.next_stage("MOVE_BEHIND_BLOCK", "pusher_lifted")
                else:
                    self.next_stage("DONE", "grasp_failed_pusher_not_lifted")

        elif self.stage == "MOVE_BEHIND_BLOCK":
            target = behind_high
            action = self._bounded_move(obs, target, gripper=1.0, max_delta=0.080)
            if self.reached(obs, target, tol=0.040) or self.stage_step >= int(self.metadata.get("move_behind_max_steps", 200)):
                self.next_stage("LOWER_TO_SWEEP", "behind_block_reached")

        elif self.stage == "LOWER_TO_SWEEP":
            target = behind_sweep
            action = self._bounded_move(obs, target, gripper=1.0, max_delta=0.040)
            # Tighter tolerance: must actually descend; 30 mm tol fires immediately from lift height.
            if self.reached(obs, target, tol=0.012) or self.stage_step >= int(self.metadata.get("lower_to_sweep_max_steps", 100)):
                self.sweep_started_step = self.step_count
                self.next_stage("SWEEP", "at_sweep_height")

        elif self.stage == "SWEEP":
            target = sweep_end
            sweep_gain = float(self.metadata.get("sweep_pos_gain", 15.0))
            # 0.018 gives ≈0.15 mm/step — too slow to reach the block in 450 steps.
            # 0.060 gives ≈0.5 mm/step, reaching the block (~93 mm) in ~185 steps.
            sweep_delta_cap = float(self.metadata.get("sweep_max_delta", 0.060))
            delta = (np.asarray(target, dtype=np.float32) - self.get_eef_pos(obs)) * sweep_gain
            action = self.make_action(np.clip(delta, -sweep_delta_cap, sweep_delta_cap), gripper=1.0)
            block_in = self._block_inside_dustpan(block, dustpan)
            swept_far_enough = self.reached(obs, target, tol=0.045)
            if block_in or swept_far_enough or self.stage_step >= int(self.metadata.get("sweep_max_steps", 700)):
                self.next_stage("WAIT_SETTLE", f"sweep_done_block_in={block_in}")

        elif self.stage == "WAIT_SETTLE":
            target = sweep_end
            self.settle_steps_done += 1
            action = self.make_action(np.zeros(3, dtype=np.float32), gripper=1.0)
            if self.settle_steps_done >= int(self.metadata.get("settle_steps", 40)):
                self.next_stage("RETRACT", "settle_done")

        elif self.stage == "RETRACT":
            target = retract_target
            self.gripper_opened = True
            action = self._bounded_move(obs, target, gripper=-1.0, max_delta=0.120)
            if self.reached(obs, target, tol=0.060) or self.stage_step >= int(self.metadata.get("retract_max_steps", 100)):
                self.next_stage("VERIFY_FINAL_STATE", "retract_done")

        elif self.stage == "VERIFY_FINAL_STATE":
            target = retract_target
            action = self.make_action(np.zeros(3, dtype=np.float32), gripper=-1.0)
            if self.stage_step >= int(self.metadata.get("verify_steps", 10)):
                if self._physical_final_state(obs):
                    self.success_step = self.step_count
                    self.next_stage("DONE", "physical_final_state_verified")
                else:
                    self.next_stage("DONE", "verification_failed")
        else:
            action = self.make_action(np.zeros(3, dtype=np.float32), gripper=-1.0)

        self.last_target = np.asarray(target, dtype=np.float32)
        self.tick()
        return action

    # ------------------------------------------------------------------
    # Success / physical checks
    # ------------------------------------------------------------------

    def is_success(self, obs, info, env) -> bool:
        self._last_debug_obs = obs
        return bool(self.stage == "DONE" and self.success_step is not None and self._physical_final_state(obs))

    def _physical_final_state(self, obs: Dict[str, np.ndarray]) -> bool:
        block = self._obj_pos(obs, self.block_name)
        dustpan = self._obj_pos(obs, self.dustpan_name)
        b0 = self.initial_block_pos

        block_moved = float(np.linalg.norm(block[:2] - b0[:2])) >= float(self.metadata.get("min_block_move", 0.080))
        block_inside = self._block_inside_dustpan(block, dustpan)
        block_z_ok = float(block[2]) <= float(dustpan[2]) + 0.070  # not flying
        return bool(
            self.pusher_grasped
            and self.settle_steps_done >= int(self.metadata.get("settle_steps", 40))
            and block_moved
            and block_inside
            and block_z_ok
            and not self.oracle_helper_used
            and self.direct_pose_writes_during_rollout == 0
        )

    def _block_inside_dustpan(self, block_pos: np.ndarray, dustpan_pos: np.ndarray) -> bool:
        """True if block XY is within tray bounds and not in front of opening."""
        xy_dist = float(np.linalg.norm(block_pos[:2] - dustpan_pos[:2]))
        radius = float(self.metadata.get("dustpan_xy_radius", 0.095))
        if xy_dist > radius:
            return False
        # Check block is inside the tray (y_local ≥ -0.015, not more than 1.5 cm in front)
        diff_xy = block_pos[:2] - dustpan_pos[:2]
        y_local = float(np.dot(diff_xy, self.lane_dir_2d))
        return y_local >= -0.015

    # ------------------------------------------------------------------
    # Debug state
    # ------------------------------------------------------------------

    def get_debug_state(self) -> Dict[str, Any]:
        state = super().get_debug_state()
        obs = getattr(self, "_last_debug_obs", None)
        state.update({
            "pusher_grasped": self.pusher_grasped,
            "max_pusher_z_delta": round(self.max_pusher_z_delta, 5),
            "settle_steps_done": self.settle_steps_done,
            "success_step": self.success_step,
            "oracle_helper_used": self.oracle_helper_used,
            "direct_pose_writes_during_rollout": self.direct_pose_writes_during_rollout,
            "target_position": self.last_target.astype(float).round(5).tolist(),
            "transition_reason": self.transition_reason,
        })
        if obs is not None:
            try:
                pusher = self._obj_pos(obs, self.pusher_name)
                block = self._obj_pos(obs, self.block_name)
                dustpan = self._obj_pos(obs, self.dustpan_name)
                state.update({
                    "eef_position": self.get_eef_pos(obs).astype(float).round(5).tolist(),
                    "pusher_position": pusher.astype(float).round(5).tolist(),
                    "block_position": block.astype(float).round(5).tolist(),
                    "dustpan_position": dustpan.astype(float).round(5).tolist(),
                    "block_moved_xy": float(np.linalg.norm(block[:2] - self.initial_block_pos[:2])),
                    "block_inside_dustpan": self._block_inside_dustpan(block, dustpan),
                })
            except Exception:
                pass
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def next_stage(self, stage: str, reason: str = "") -> None:
        super().next_stage(stage)
        self.transition_reason = reason or f"to_{stage}"

    def _bounded_move(self, obs, target_pos, gripper: float, max_delta: float):
        delta = (np.asarray(target_pos, dtype=np.float32) - self.get_eef_pos(obs)) * self.pos_gain
        return self.make_action(np.clip(delta, -max_delta, max_delta), gripper=gripper)

    def _obj_pos(self, obs: Dict[str, np.ndarray], name: str) -> np.ndarray:
        key = f"{name}_pos"
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32)
        if self.env is None:
            raise KeyError(key)
        return np.asarray(
            self.env.env.sim.data.body_xpos[self.env.env.obj_body_id[name]], dtype=np.float32
        )
