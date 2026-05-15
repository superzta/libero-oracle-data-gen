"""Oracle controller implementations for LIBERO data collection."""

from controllers.base_fsm_controller import BaseFSMController, NoOpController
from controllers.button_box_controller import ButtonBoxController
from controllers.peg_insertion_controller import PegInsertionController
from controllers.ring_hook_controller import RingHookController
from controllers.tool_sweep_controller import ToolSweepController

CONTROLLER_REGISTRY = {
    "noop": NoOpController,
    "peg_insertion": PegInsertionController,
    "tool_sweep": ToolSweepController,
    "button_box": ButtonBoxController,
    "ring_hook": RingHookController,
}

