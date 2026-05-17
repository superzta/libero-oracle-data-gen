"""Unit tests for button_box primitive failure diagnostics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from qa_button_box_rollout import primitive_failure_code


class ButtonBoxDiagnosticTests(unittest.TestCase):
    def test_grasp_failed_cube_not_lifted(self):
        code = primitive_failure_code(
            ["MOVE_ABOVE_BUTTON", "HOLD_BUTTON_PRESS", "LIFT_CUBE"],
            button_pressed=True,
            cube_z_increase=0.01,
            cube_final_inside_box=False,
            release_step=None,
            success_step=None,
            settle_frames=0,
        )
        self.assertEqual(code, "grasp_failed_cube_not_lifted")

    def test_place_failed_not_inside_box(self):
        code = primitive_failure_code(
            ["HOLD_BUTTON_PRESS", "LIFT_CUBE", "OPEN_GRIPPER", "WAIT_SETTLE", "VERIFY_FINAL_STATE"],
            button_pressed=True,
            cube_z_increase=0.05,
            cube_final_inside_box=False,
            release_step=100,
            success_step=None,
            settle_frames=50,
        )
        self.assertEqual(code, "place_failed_not_inside_box")

    def test_success_before_settle_invalid(self):
        code = primitive_failure_code(
            ["HOLD_BUTTON_PRESS", "LIFT_CUBE", "OPEN_GRIPPER"],
            button_pressed=True,
            cube_z_increase=0.05,
            cube_final_inside_box=True,
            release_step=100,
            success_step=99,
            settle_frames=0,
        )
        self.assertEqual(code, "success_before_settle_invalid")


if __name__ == "__main__":
    unittest.main()
