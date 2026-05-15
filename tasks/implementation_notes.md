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

The button-box task is the first working custom task. It loads from `bddl_files/button_box.bddl` without copying files into LIBERO because it uses the existing registered problem class `LIBERO_Tabletop_Manipulation` and existing object categories:

- `red_coffee_mug_1 - red_coffee_mug` as the red button proxy.
- `butter_1 - butter` as the blue cube proxy for the debug version.
- `white_storage_box_1 - white_storage_box` as the box.

LIBERO maps BDDL problem names to registered Python classes through `TASK_MAPPING` in `libero.libero.envs.bddl_base_domain`. The stock problem classes are registered by decorators such as `@register_problem` in `libero/libero/envs/problems/*.py`. Object category strings in BDDL map through `OBJECTS_DICT`, populated by `@register_object` decorators in `libero/libero/envs/objects/*.py`.

The current `ButtonBoxController` uses ground-truth poses from observation keys, tracks button press internally when the end-effector reaches the press pose, then moves through the full FSM. For reliable debug collection with proxy assets, it uses an oracle object attachment/place helper after the grasp stage. This should be replaced with a more physical controller once a dedicated cube and button asset are added.

Exact custom task commands:

```bash
python scripts/test_custom_button_box.py

python scripts/collect_oracle_demos.py \
  --custom-task button_box \
  --controller button_box \
  --num-successes 5 \
  --max-attempts 10 \
  --horizon 350 \
  --camera-size 64 \
  --output-dir datasets \
  --keep-failures

python scripts/validate_dataset.py datasets/<button_box_run_dir> --expected-successes 5
python scripts/summarize_dataset.py datasets/<button_box_run_dir>
```

Other FSM controllers are structured and metadata-driven, but custom tasks require matching observation keys such as `green_peg_1_pos` or `dustpan_1_pos`. Once assets and BDDL exist, tune approach heights, contact heights, tolerances, and success checks from rollouts.

## Dataset Quality Checks

Validation checks:

- Expected number of successful demos.
- No duplicate seeds.
- Action shape `(T, 7)`.
- Required observation keys.
- All success flags true.
- Initial states vary.

Summaries include success count, attempts, success rate, average episode length, failure reasons, and object initial-state distributions.
