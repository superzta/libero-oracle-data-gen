"""Custom primitive objects for the LIBERO oracle task suite.

Importing this module registers the classes with LIBERO's OBJECTS_DICT.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import register_object

ASSET_ROOT = Path(__file__).resolve().parents[1] / "custom_objects" / "assets"


class OracleXMLObject(MujocoXMLObject):
    """Small XML-backed object whose assets live in this repo."""

    def __init__(self, name: str, obj_name: str, joints="default"):
        if joints == "default":
            joints = [dict(type="free", damping="0.0005")]
        super().__init__(
            str(ASSET_ROOT / obj_name / f"{obj_name}.xml"),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.rotation = (0.0, np.pi / 2)
        self.rotation_axis = "z"
        self.category_name = "_".join(
            re.sub(r"([A-Z0-9])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.object_properties = {"vis_site_names": {}}


@register_object
class RedButton(OracleXMLObject):
    """A short red cylinder used as a press target."""

    def __init__(self, name="red_button", obj_name="red_button", joints=None):
        if joints is None:
            joints = []
        super().__init__(name=name, obj_name=obj_name, joints=joints)


@register_object
class BlueCube(OracleXMLObject):
    """A visually explicit blue cube for pick-and-place."""

    def __init__(self, name="blue_cube", obj_name="blue_cube", joints="default"):
        super().__init__(name=name, obj_name=obj_name, joints=joints)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"


@register_object
class OpenBox(OracleXMLObject):
    """A simple open-top container with a `contain_region` site."""

    def __init__(self, name="open_box", obj_name="open_box", joints="default"):
        if joints == "default":
            joints = [dict(type="free", damping="0.0005")]
        super().__init__(name=name, obj_name=obj_name, joints=joints)


@register_object
class GreenPeg(OracleXMLObject):
    """A graspable upright green peg for insertion tasks."""

    def __init__(self, name="green_peg", obj_name="green_peg", joints="default"):
        super().__init__(name=name, obj_name=obj_name, joints=joints)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"


@register_object
class WoodenHoleBlock(OracleXMLObject):
    """A wooden block with a visible shallow socket and contain_region site."""

    def __init__(self, name="wooden_hole_block", obj_name="wooden_hole_block", joints="default"):
        if joints == "default":
            joints = [dict(type="free", damping="0.0005")]
        super().__init__(name=name, obj_name=obj_name, joints=joints)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"


@register_object
class PusherTool(OracleXMLObject):
    """A T-shaped pusher tool: cylindrical handle + wide blade for non-prehensile sweep."""

    def __init__(self, name="pusher_tool", obj_name="pusher_tool", joints="default"):
        super().__init__(name=name, obj_name=obj_name, joints=joints)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"


@register_object
class RedBlock(OracleXMLObject):
    """A flat red box used as the swept object in the tool_sweep task."""

    def __init__(self, name="red_block", obj_name="red_block", joints="default"):
        super().__init__(name=name, obj_name=obj_name, joints=joints)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"


@register_object
class Dustpan(OracleXMLObject):
    """A shallow open tray with three walls and a contain_region site."""

    def __init__(self, name="dustpan", obj_name="dustpan", joints="default"):
        if joints == "default":
            joints = [dict(type="free", damping="0.0005")]
        super().__init__(name=name, obj_name=obj_name, joints=joints)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
