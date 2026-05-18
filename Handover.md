# Handover: button_box Oracle Dataset

## Problem Statement

We need a large, diverse oracle demonstration dataset for the `button_box` task in the LIBERO benchmark. The task requires a robot arm to:
1. Press a red button on the table
2. Pick up a blue cube from the table
3. Place the cube inside an open box on the table

The dataset will be used to train a policy. For training to generalise, the **initial scene layout must vary substantially between episodes** — objects should appear anywhere on the table, not in fixed or near-fixed positions.

---

## Goal

Collect **100 successful demonstrations** (`diverse_v2` randomization level) where:
- All three objects (cube, box, red button) are placed **randomly and independently anywhere on the visible table** each episode
- The robot arm physically **closes its gripper on the button** (not just hovering nearby)
- Success is gated on **actual geometric button press detection** — episodes where the arm cannot reach the button simply fail and get retried
- No oracle pose-injection during rollout (`oracle_helper_used = False`)
- All success conditions verified by `_physical_final_state()` which checks button press, cube movement, cube inside box, gripper opened, and button stability

---

## Current State (as of last session)

### What is working

| Component | Status |
|---|---|
| Full-table continuous randomization | ✅ All 3 objects placed uniformly at random across x ∈ [−15, +15] cm, y ∈ [−20, +25] cm |
| Button press (visual) | ✅ Gripper closes (`gripper=1.0`) during `PRESS_BUTTON` and `HOLD_BUTTON_PRESS` — fingers physically contact button |
| Button press (geometric gate) | ✅ `button_geometrically_pressed` flag; requires 2 consecutive steps with EEF xy ≤ 3.2 cm from button AND EEF z ≤ button_z + 4.5 cm |
| Physical success gate | ✅ `_physical_final_state()` requires `button_geometrically_pressed` (was bugged — used to always set True even on timeout) |
| Rejection sampling | ✅ Cube position rejection-sampled to avoid box interior and button proximity |
| Overall success rate | ~73% (15-seed QA: 11/15 PASS) |
| 5-demo visual batch | ✅ Collected; objects appear in clearly different positions each frame |
| QA gate (10-seed) | ✅ 10/10 PASS with horizon=900 |

### What is not working / known limitations

| Issue | Detail |
|---|---|
| Button in back half of table (y > 0) | Arm workspace floor is too high to press at those positions — episode fails, gets retried. About 27% of random button placements fall in this region. |
| Cube in far-back table area | Arm reach limits can cause grasp failure at y > 0.150 — rare, retried. |
| Success rate ~73% | Means collecting 100 demos needs ~137 attempts (max_attempts=200 is sufficient). |
| Horizon must be ≥ 900 | Objects now further apart, arm travels more. Default 500-step horizon was too short. |

---

## What Has Been Tried

### Randomization evolution
- **debug_small / medium / final / diverse**: legacy bin-based levels with small position variation — objects barely moved visually
- **diverse_v2 (bins)**: 6 cube bins × 4 box bins × 4 button bins, decoupled with coprime-stride LCG to prevent correlated motion. Worked but still looked structured
- **diverse_v2 (continuous, per-object zones)**: replaced bins with continuous uniform sampling but kept separate x/y ranges for each object. User objected — wanted fully random, not zone-separated
- **diverse_v2 (continuous, full table)**: current implementation — single shared table area, all three objects sample from same pool, overlap rejection only

### Button press bug fix
- **Bug**: `HOLD_BUTTON_PRESS` stage always set `self.button_pressed = True` at end of hold, regardless of whether the arm actually pressed the button geometrically. Episode could "succeed" without physical button contact.
- **Fix**: Added `button_geometrically_pressed` flag (set only when `button_press_consecutive ≥ 2`). `_physical_final_state()` now checks `button_geometrically_pressed` not `button_pressed`. Episodes that time out on button press now correctly fail.

### Button press reliability improvements
- Increased `MOVE_ABOVE_BUTTON` timeout from 45 → 80 steps (`move_above_button_timeout_steps`)
- Increased `PRESS_BUTTON` timeout from 45 → 80 steps (`press_button_timeout_steps`)
- Reduced required consecutive press steps from 4 → 2 (`button_press_required_steps`)
- Changed gripper from open (−1.0) to closed (+1.0) during press stages — fingers physically close on button

### Infeasible button positions (discovered during testing)
- x = +0.155, y = −0.155: arm workspace floor ≈ z=0.967, above press threshold z=0.945 → always fails
- x = +0.130, y = −0.080: same issue
- Positions near x ≈ 0 with y > −0.110: marginal workspace, unreliable
- Fix: full-table randomization naturally handles this — those seeds just fail and get retried

---

## What to Expect

- **Success rate**: ~70–75% of seeds produce a successful episode
- **Failed episodes**: arm couldn't press button (button too far back/right), or grasp slipped, or cube wasn't placed inside box — all silent failures, retried automatically
- **Episode length**: 550–870 steps depending on object spread (horizon=900 required)
- **Diversity**: every frame visually different — cube, box, button in completely different positions

---

## Git / Data Hygiene

### What is NOT tracked by git (gitignored)
```
datasets/          ← HDF5 files from all collection runs
videos/            ← MP4 per-episode videos
logs/              ← Collection logs
reports/           ← All generated QA/debug reports (except two docs below)
*.hdf5 *.h5        ← Any HDF5 anywhere
*.mp4 *.avi *.mov  ← Any video anywhere
*.npy *.npz        ← Any numpy arrays
```

Two report files ARE tracked (manually authored, not generated):
- `reports/existing_libero_tasks.txt`
- `reports/libero_structure.md`

### Clean commit — what to stage for next commit
```bash
git add .gitignore scripts/button_box_reset_utils.py Handover.md
# The staged deletions of generated report files are already in the index
git commit
```

---

## Collecting 100 Successful Demos

### Command

```bash
conda run -n libero python scripts/collect_oracle_demos.py \
  --custom-task button_box \
  --controller button_box \
  --num-successes 100 \
  --max-attempts 200 \
  --horizon 900 \
  --camera-size 128 \
  --save-video \
  --output-dir datasets \
  --final-hold-steps 30 \
  --require-physical-success \
  --randomization-level diverse_v2
```

Use `--camera-size 256` for higher-resolution videos (4× larger files). Omit `--save-video` to skip video and save disk space.

### Where to find the output

Every run creates a timestamped directory under `datasets/`:

```
datasets/button_box_button_box_YYYYMMDD_HHMMSS/
├── run_manifest.json          ← paths, args, seed list, success/video file lists
├── summary.json               ← successes, attempts, success_rate, failure breakdown
├── success_001_seed_0.hdf5
├── success_002_seed_3.hdf5
│   ...                        ← one HDF5 per successful episode
└── success_100_seed_NNN.hdf5
```

Videos (if `--save-video`) are stored in a parallel directory:

```
videos/button_box_button_box_YYYYMMDD_HHMMSS/
├── success_001_seed_0.mp4
├── success_002_seed_3.mp4
│   ...
```

### HDF5 contents (per episode)

```
observations/          ← robot and object poses per timestep
  robot0_eef_pos       ← end-effector position
  blue_cube_1_pos      ← cube xyz
  open_box_1_pos       ← box xyz
  red_button_1_pos     ← button xyz
  agentview_image      ← camera frames (if camera obs enabled)
actions                ← 7-dim OSC_POSE actions [dx,dy,dz,drot×3,gripper]
rewards                ← per-step reward
dones                  ← episode termination flags
env_states             ← full simulator state for replay
debug/                 ← JSON state trace (stage, button_pressed, etc.)
attrs:
  metadata             ← JSON: seed, cube_bin_id, box_bin_id, button_bin_id,
                          initial_positions, final_positions, episode_length,
                          randomization_level, oracle_helper_used, ...
  success              ← True for all files in the success_*.hdf5 pattern
```

### Monitoring progress

The script prints to stdout in real time:
```
[collect attempt 5/200] seed=4 PASS successes=4/100 reason=ok
running: attempts=5 pass=4 fail=1 success_rate=0.800 most_common_failure=max_steps
```

To save a log file:
```bash
conda run -n libero python scripts/collect_oracle_demos.py ... 2>&1 | tee logs/run_100demo.log
```

### Post-collection validation and visual QA

```bash
# Validate HDF5 integrity and counts
python scripts/validate_dataset.py datasets/<run_dir> --expected-successes 100

# Check diversity (positional range, bin coverage)
python scripts/check_initial_state_diversity.py datasets/<run_dir> --level diverse_v2 --strict

# Generate initial-frame contact sheet (5-col grid, labelled with positions)
python scripts/make_initial_state_contact_sheet.py datasets/<run_dir> \
  --video-dir videos/<run_dir> --cols 10 --thumb-size 128 \
  --output reports/diverse_v2_100demo_contact_sheet.jpg

# Print summary stats
python scripts/summarize_dataset.py datasets/<run_dir>
```
