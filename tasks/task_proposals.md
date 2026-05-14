# Proposed LIBERO Task Extensions

## Candidate Task 1: Precision Insertion

Instruction:

> Insert the green peg into the matching hole on the wooden block.

Novelty:

This task requires precision pose alignment and insertion into a constrained target. It is qualitatively different from loose pick-and-place, putting objects in bowls/baskets/drawers, or placing objects on surfaces.

Planned controller:

State-based finite-state-machine controller:
1. Move above peg.
2. Descend to grasp pose.
3. Close gripper.
4. Lift peg.
5. Move above hole.
6. Align x/y/yaw.
7. Lower slowly into hole.
8. Release and retract.
9. Check insertion success.

---

## Candidate Task 2: Hanging / Hooking

Instruction:

> Hang the ring on the hook.

Novelty:

This task requires hooking/hanging behavior. The object must be supported by the hook after release, which is different from placing an object on a flat surface or inside a container.

Planned controller:

State-based finite-state-machine controller:
1. Move above ring.
2. Grasp ring.
3. Lift ring.
4. Move to pre-hook pose.
5. Align ring center with hook.
6. Lower ring around hook.
7. Release.
8. Check that ring remains supported by hook.

---

## Candidate Task 3: Tool-Mediated Sweeping

Instruction:

> Use the pusher to sweep the red block into the dustpan.

Novelty:

This task requires tool-mediated non-prehensile manipulation. The robot manipulates a tool to move another object rather than directly grasping the target object.

Planned controller:

State-based finite-state-machine controller:
1. Move above pusher handle.
2. Grasp pusher.
3. Move pusher behind red block.
4. Lower pusher to contact height.
5. Sweep red block toward dustpan.
6. Stop once block enters dustpan region.
7. Check success by target block position.