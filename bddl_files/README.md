# Custom BDDL Files

Place candidate custom task BDDL files here once the corresponding LIBERO object categories and problem classes exist.

At the moment, the approved tasks require geometry and/or predicates that are not cleanly represented by stock LIBERO objects:

- Peg and matching socket/hole.
- Pusher, red block, and dustpan target.
- Pressable red button, blue cube, and open box.
- Ring and hook.

Use `python scripts/install_custom_tasks.py --dry-run` to inspect the reproducible copy plan before installing any BDDL files into `~/projects/LIBERO`.

