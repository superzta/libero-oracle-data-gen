# Implementation Notes

## LIBERO Extension Path

LIBERO loads BDDL by mapping the BDDL problem name to a registered Python problem class in `libero.libero.envs.bddl_base_domain.TASK_MAPPING`. Existing problem classes can parse existing object categories and predicates. New objects or new task-level success semantics require adding object definitions, MJCF assets, and possibly problem/predicate classes to the LIBERO checkout.

This repo should remain the source of truth for:

- Controller code.
- Candidate BDDL files.
- Dataset collection scripts.
- Validation and reporting.
- Documentation.

If BDDL files need to be visible inside LIBERO, use:

```bash
python scripts/install_custom_tasks.py --dry-run
python scripts/install_custom_tasks.py --apply
```

## Current Controller Status

The FSM controllers are structured and metadata-driven, but custom tasks require matching observation keys such as `green_peg_1_pos` or `dustpan_1_pos`. Once assets and BDDL exist, tune approach heights, contact heights, tolerances, and success checks from rollouts.

## Dataset Quality Checks

Validation checks:

- Expected number of successful demos.
- No duplicate seeds.
- Action shape `(T, 7)`.
- Required observation keys.
- All success flags true.
- Initial states vary.

Summaries include success count, attempts, success rate, average episode length, failure reasons, and object initial-state distributions.

