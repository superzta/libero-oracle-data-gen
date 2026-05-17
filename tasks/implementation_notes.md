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

The button-box task is the active task and is not final-quality yet. It loads from `bddl_files/button_box.bddl` without copying files into LIBERO because it uses the existing registered problem class `LIBERO_Tabletop_Manipulation` plus repo-local custom object registrations imported at runtime:

- `red_button_1 - red_button`
- `blue_cube_1 - blue_cube`
- `open_box_1 - open_box`

LIBERO maps BDDL problem names to registered Python classes through `TASK_MAPPING` in `libero.libero.envs.bddl_base_domain`. The stock problem classes are registered by decorators such as `@register_problem` in `libero/libero/envs/problems/*.py`. Object category strings in BDDL map through `OBJECTS_DICT`, populated by `@register_object` decorators in `libero/libero/envs/objects/*.py`.

The current `ButtonBoxController` uses ground-truth poses from observation keys and object-relative targets. Button press success is geometric: end-effector xy inside the button radius and z below the press threshold for consecutive steps. Success can only be reported after `OPEN_GRIPPER`, `WAIT_SETTLE`, `RETRACT_FROM_BOX`, and `VERIFY_FINAL_STATE`.

Important status from the latest strict probe:

- `python scripts/test_custom_button_box.py` passes.
- `python scripts/debug_button_box_primitives.py --mode pick_lift_only --num-seeds 10 --strict` passes 10/10.
- `python scripts/debug_button_box_primitives.py --mode place_only --num-seeds 10 --strict` passes 10/10.
- `python scripts/qa_button_box_rollout.py --num-seeds 5 --horizon 700 --strict` passes 5/5.
- Button xy drift is 0 in the latest full QA, no direct object pose writes/oracle helper use were detected, and success occurs after release, settle, retract, and final physical verification.
- No 100-demo collection should be run until a 5-success video batch, dataset validation, summary, and strict video QA pass.

Current calibrated button-box settings:

- Red button is a fixed fixture-style pad; press success is geometric from end-effector pose.
- Blue cube/block is a single-body single-geom graspable block with high friction.
- Box is a wide shallow open tray.
- Reset-time cube randomization uses a feasible medium range: x in `[-0.030, 0.030]`, y in `[-0.035, -0.005]`.
- Integrated controller freezes the sampled cube grasp pose during descend/close/lift and recomputes it only on explicit grasp retry.
- Integrated lift-complete threshold is `0.04` m, matching the strict primitive acceptance criterion.

Exact custom task commands:

```bash
python scripts/test_custom_button_box.py

python scripts/debug_button_box_primitives.py --mode pick_lift_only --num-seeds 10 --strict

python scripts/debug_button_box_primitives.py --mode place_only --num-seeds 10 --strict

python scripts/qa_button_box_rollout.py --num-seeds 5 --horizon 700 --strict

python scripts/collect_oracle_demos.py \
  --custom-task button_box \
  --controller button_box \
  --num-successes 5 \
  --max-attempts 25 \
  --horizon 700 \
  --camera-size 128 \
  --save-video \
  --output-dir datasets \
  --keep-failures \
  --final-hold-steps 50 \
  --require-physical-success

python scripts/validate_dataset.py datasets/<button_box_run_dir> --expected-successes 5
python scripts/summarize_dataset.py datasets/<button_box_run_dir>
python scripts/analyze_rollout_video.py videos/<button_box_run_dir>/success_001_seed_<seed>.mp4 --output-dir reports/button_box_video_qa --strict
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
