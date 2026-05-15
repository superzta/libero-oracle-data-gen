# LIBERO Oracle Data Generation

Research engineering project for extending LIBERO with novel manipulation tasks and collecting successful oracle demonstrations with finite-state-machine controllers.

## Project Goal

The target dataset is 100 successful trials per approved task, collected from LIBERO / robosuite environments with ground-truth state controllers. The pipeline is built to save observations, actions, rewards, dones, robot proprioception, object poses, simulator state when available, success flags, language, seeds, metadata, failure reasons, and optional videos.

Approved tasks:

- Insert the green peg into the matching hole on the wooden block.
- Use the pusher to sweep the red block into the dustpan.
- Press the red button, then place the blue cube inside the box.
- Stretch: Hang the ring on the hook.

## Setup

LIBERO is expected to be installed separately:

```bash
cd ~/projects/LIBERO
pip install -e .
```

Run commands from this repository:

```bash
cd ~/projects/libero-oracle-data-gen
```

The scripts set `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and `NUMBA_DISABLE_JIT=1` before importing LIBERO. `NUMBA_DISABLE_JIT=1` avoids a robosuite numba cache failure seen in this environment.

## Stage 1 Status

Current repo inspection found an intentionally small project plus generated-output folders. The installed LIBERO checkout contains existing BDDL suites, object assets, problem classes, and task registration through `TASK_MAPPING`. External BDDL files can be loaded directly only when they use existing LIBERO problem classes and object categories. Brand-new objects or predicates should be installed into the LIBERO checkout through a reviewed, reproducible script rather than ad hoc edits.

## Built-In Checks

Print existing LIBERO tasks:

```bash
python scripts/print_existing_libero_tasks.py
```

Inspect LIBERO paths and registered problem classes:

```bash
python scripts/inspect_libero_structure.py
```

Run a reset/step smoke test on a built-in task:

```bash
python scripts/test_builtin_libero.py
```

## Collect Demos

Run a short pipeline verification on a built-in task with the no-op controller:

```bash
python scripts/collect_oracle_demos.py \
  --suite libero_10 \
  --task-id 0 \
  --controller noop \
  --num-successes 0 \
  --max-attempts 1 \
  --horizon 10
```

Collect successful trials once a task-specific controller and BDDL are ready:

```bash
python scripts/collect_oracle_demos.py \
  --bddl-file bddl_files/<custom_task>.bddl \
  --controller peg_insertion \
  --num-successes 100 \
  --max-attempts 500 \
  --seed 0 \
  --camera-size 128 \
  --save-video \
  --output-dir datasets
```

Controller-specific object names or target positions can be passed as JSON:

```bash
python scripts/collect_oracle_demos.py \
  --bddl-file bddl_files/<custom_task>.bddl \
  --controller tool_sweep \
  --controller-metadata '{"pusher_name":"pusher_1","block_name":"red_block_1","dustpan_name":"dustpan_1"}'
```

## Validate And Summarize

```bash
python scripts/validate_dataset.py datasets/<run_dir> --expected-successes 100
python scripts/summarize_dataset.py datasets/<run_dir>
```

Replay one episode and save a video:

```bash
python scripts/replay_episode.py datasets/<run_dir>/success_001_seed_0.hdf5 \
  --save-video videos/replay_success_001.mp4
```

Create a montage from saved rollout videos:

```bash
python scripts/make_video_montage.py videos/<run_dir>
```

## Custom Task Installation

Use `scripts/install_custom_tasks.py` only when custom BDDL must be copied into the editable LIBERO checkout:

```bash
python scripts/install_custom_tasks.py --dry-run
python scripts/install_custom_tasks.py --apply
```

For the approved tasks, true custom object classes are likely needed for peg/hole, pusher/dustpan, button, box, ring, and hook geometry. The minimal clean path is:

1. Keep design docs, controller code, and candidate BDDL in this repo.
2. Prototype with existing LIBERO objects where possible.
3. Add reviewed object/problem-class patches to LIBERO only when the geometry or predicates cannot be represented by existing categories.
4. Use `install_custom_tasks.py` for reproducible BDDL copies.

## Output Structure

- `datasets/`: HDF5 rollout files and per-run `summary.json`.
- `videos/`: optional per-episode videos and montages.
- `logs/`: collection/failure logs.
- `reports/`: task lists, inspection reports, and dataset summaries.

`datasets/`, `videos/`, and `logs/` are gitignored.

## Current Status / Next Steps

Implemented:

- Built-in task listing and LIBERO structure inspection.
- Import-environment fixes for this robosuite/numba setup.
- Generic oracle collection pipeline with HDF5 serialization.
- FSM base controller and skeleton controllers for all approved tasks.
- Dataset validation, summary, replay, montage, and conservative BDDL install scripts.

Next steps:

- Create or import real MJCF assets for the approved custom objects.
- Add minimal LIBERO object/problem registrations in a reviewed upstream patch if existing problem classes are insufficient.
- Tune each FSM against rendered rollouts until it achieves high success rate.
- Run collection to 100 successful demonstrations per task and validate each dataset.
