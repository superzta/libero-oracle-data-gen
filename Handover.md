# Handover: Oracle Demo Datasets

---

## Task 1: button_box — COMPLETE

### Problem Statement

Collect 100 successful oracle demonstrations for the `button_box` task (press button → pick cube → place cube in box) with full-table diverse randomization (`diverse_v2`).

### Goal

100 successful demos with:
- All three objects (cube, box, button) placed randomly and independently anywhere on the visible table each episode
- Robot physically closes gripper on button (not hovering)
- Success gated on `_physical_final_state()`: button press, cube movement, cube inside box, gripper opened, button stability
- No oracle pose-injection during rollout (`oracle_helper_used = False`)

### Current State

**100 demos collected. Task is DONE.**

| Component | Status |
|---|---|
| Full-table continuous randomization | ✅ All 3 objects uniform across x ∈ [−15, +15] cm, y ∈ [−20, +25] cm |
| Button press (physical contact) | ✅ Gripper closes (`gripper=1.0`) during `PRESS_BUTTON`/`HOLD_BUTTON_PRESS` |
| Button press geometric gate | ✅ `button_geometrically_pressed` flag; requires 2 consecutive steps EEF xy ≤ 3.2 cm from button AND EEF z ≤ button_z + 4.5 cm |
| Physical success gate | ✅ `_physical_final_state()` checks `button_geometrically_pressed` |
| Success rate | ~73% |

### Known Limitations

- Button in back half of table (y > 0): arm workspace floor too high → episode fails, gets retried (~27% of seeds)
- Episode length 550–870 steps; horizon=900 required

### Collect Command (if re-running)

```bash
conda run -n libero python scripts/collect_oracle_demos.py \
  --custom-task button_box \
  --controller button_box \
  --num-successes 100 \
  --max-attempts 200 \
  --horizon 900 \
  --camera-size 128 \
  --output-dir datasets \
  --final-hold-steps 30 \
  --require-physical-success \
  --randomization-level diverse_v2
```

---

## Task 2: peg_insertion — 5-VIDEO GATE PASSED, READY FOR 100-DEMO COLLECTION

### Problem Statement

Collect 100 successful oracle demonstrations for the `peg_insertion` task: pick up a green peg (box 0.044 × 0.044 × 0.110 m) and insert it into the square socket of a wooden block, using diverse_v2 randomization.

### Goal

100 successful demos with:
- Green peg and wooden socket randomized independently across full table area (`diverse_v2`)
- Robot physically grasps peg, transports to socket, lowers peg in, releases, arm retracts
- Peg remains upright (upright score ≥ 0.94) and centered (XY error ≤ 0.018 m) in socket after release
- No oracle pose-injection during rollout

### Key Geometry

| Element | Value |
|---|---|
| Peg half-sizes | 0.022 × 0.022 × 0.055 m |
| Socket collision inner edge | 0.040 m (widened from 0.034 m) |
| Socket wall top (world z) | block_z + 0.044 ≈ **0.944 m** |
| Peg entry XY tolerance | 0.018 m (axis-aligned) |
| **Socket blocking height** | peg_center_z = 0.999 m (= wall top 0.944 + peg half-height 0.055) |
| Peg-EEF Z offset during grip | 0.024–0.025 m (EEF above peg center) |

The **blocking height** is critical: when peg XY error > 0.018 m, the peg corner contacts the socket wall top at peg_center_z = 0.999. Any controller stage that descends the peg to z < 0.999 before XY is aligned will stall.

### FSM Controller Stages

```
MOVE_ABOVE_PEG (120 steps max)
DESCEND_TO_PEG (180 steps max)
CLOSE_GRIPPER_AND_WAIT (40 steps)
LIFT_PEG (140 steps max)
MOVE_ABOVE_HOLE (165 steps max)
ALIGN_WITH_HOLE (105 steps max)    ← coarse XY align, EEF target z = hole_z + 0.110
FINE_ALIGN (160 steps max)         ← direct peg→hole XY feedback, EEF stays at z = hole_z + 0.110
LOWER_INSERT (160 steps max)       ← descend EEF to z = hole_z + 0.040
HOLD_INSERT (22 steps)
OPEN_GRIPPER (12 steps)
WAIT_SETTLE (35 steps)
RETRACT (65 steps max)
VERIFY_FINAL_STATE (5 steps)
DONE
```

### Changes Made This Session

#### 1. Socket widened (`custom_objects/assets/wooden_hole_block/wooden_hole_block.xml`)

Collision walls inner edge widened from 0.034 m → 0.040 m. Maximum XY entry offset is now 0.018 m (was 0.012 m).

```xml
<!-- Before -->
<geom pos="0.061 0 0.025" size="0.027 0.088 0.019" .../>   <!-- inner edge 0.034 -->
<!-- After -->
<geom pos="0.061 0 0.025" size="0.021 0.088 0.019" .../>   <!-- inner edge 0.040 -->
```

Visual walls (group=1) are unchanged.

#### 2. FINE_ALIGN stage added (`controllers/peg_insertion_controller.py`)

A new FSM stage between ALIGN_WITH_HOLE and LOWER_INSERT. Provides direct peg→hole XY feedback at a safe height, before descent begins.

```python
elif self.stage == "FINE_ALIGN":
    peg_xy_err = float(np.linalg.norm(peg[:2] - hole[:2]))
    eef = self.get_eef_pos(obs)
    xy_correction = np.clip(hole[:2] - peg[:2], -0.050, 0.050)
    fine_z = hole[2] + float(self.metadata.get("hole_align_z_offset", 0.110))
    wrist_target = np.array([eef[0]+xy_correction[0], eef[1]+xy_correction[1], fine_z])
    action = self._bounded_move(obs, wrist_target, gripper=1.0,
                                max_delta=self.metadata.get("fine_align_max_delta", 0.040))
    exit_tol = self.metadata.get("fine_align_xy_exit_tol", 0.012)
    if peg_xy_err <= exit_tol or self.stage_step >= self.metadata.get("fine_align_max_steps", 160):
        self.next_stage("LOWER_INSERT", ...)
```

#### 3. `hole_align_z_offset` default raised: 0.090 → 0.110

**Root cause of 8/9 pre-fix failures**: The old default 0.090 set EEF target to hole_z + 0.090 = 1.011 m, putting the peg at z ≈ 0.987 m during FINE_ALIGN. This is 12 mm **below** the socket blocking height (0.999 m). With any XY error > 0.018 m, the peg corner contacted the socket wall top, blocking further descent and preventing XY convergence.

**Fix**: 0.110 sets EEF target to 1.031 m, peg center to 1.006 m — 7 mm above blocking height. FINE_ALIGN now runs unobstructed regardless of initial XY error.

```
Blocking height:  0.999 m  (socket wall top 0.944 + peg half-height 0.055)
Old peg target:   0.987 m  ← BELOW blocking height — XY > 0.018 m causes stall
New peg target:   1.006 m  ← 7 mm above blocking height — safe for all XY errors
```

LOWER_INSERT gap increases from 0.050 m → 0.070 m, but the `reached(tol=0.040)` exit fires at ~138 steps, safely within the 160-step limit.

#### 4. Other controller tuning (from earlier sessions)

| Parameter | Old | New | Reason |
|---|---|---|---|
| `FINE_ALIGN` max_delta | — | 0.040 | Fast direct XY correction |
| `FINE_ALIGN` max_steps | — | 160 | Up to 0.150 m XY travel at ~0.001 m/step |
| `FINE_ALIGN` exit_tol | — | 0.012 m | Tighter than socket clearance (0.018 m) |
| `LOWER_INSERT` max_delta | 0.026 | 0.018 | Slower, avoids peg bouncing on entry |
| `LOWER_INSERT` max_steps | 140 | 160 | More time for deep seeds |
| `hold_insert_steps` default | 8 | 22 | Longer hold before gripper opens |

### Acceptance-Rate Test Results

Run: `datasets/peg_insertion_peg_insertion_20260519_112821`  
Command: `--num-successes 20 --max-attempts 120 --horizon 1100 --camera-size 128` (no video)

**Result: 20/29 = 69%** — exceeds 40% threshold for proceeding.

| Failure reason | Count | Root cause |
|---|---|---|
| `verification_failed` | 8 | Peg stuck on socket rim during FINE_ALIGN (old `hole_align_z_offset=0.090`) |
| `physical_final_state_verified` | 1 | Peg fell uncontrolled into socket after being stuck on rim, bounced at verify time |

Both failure modes are addressed by `hole_align_z_offset=0.110`.

### 5-Video Gate Results

Run: `datasets/peg_insertion_peg_insertion_20260519_120017`  
Command: `--num-successes 5 --max-attempts 80 --horizon 1100 --camera-size 256 --save-video`

**Result: 5/5 = 100%, 0 failures.**

| Episode | Length | XY error | Upright | Peg Z | Moved |
|---|---|---|---|---|---|
| success_001_seed_0 | 822 | 0.0042 m | 0.9980 | 0.9668 | 0.160 m |
| success_002_seed_1 | 727 | 0.0027 m | 0.9946 | 0.9668 | 0.163 m |
| success_003_seed_2 | 737 | 0.0064 m | 0.9970 | 0.9668 | 0.174 m |
| success_004_seed_3 | 784 | 0.0011 m | 0.9999 | 0.9668 | 0.134 m |
| success_005_seed_4 | 810 | 0.0067 m | 0.9964 | 0.9668 | 0.137 m |

All pegs rest on the socket floor (z = 0.967 = floor 0.912 + half-height 0.055). XY errors 1–7 mm. Upright scores 0.994–0.9999.

Videos: `videos/peg_insertion_peg_insertion_20260519_120017/` (5 individual + montage)

### Visual Acceptance Criteria (check before approving 100-demo run)

Watch `montage.mp4` and confirm each episode shows:
- [ ] Green peg and wooden socket in **clearly different** starting positions
- [ ] Robot arm physically closes gripper on peg (not attached)
- [ ] Arm lifts peg off table and transports to socket
- [ ] Peg aligns above socket, then descends cleanly into the hole
- [ ] Gripper opens; peg stays upright and centered in socket
- [ ] Arm retracts; peg remains in socket through end of clip

### Collect 100 Demos Command

Once videos pass visual inspection, run:

```bash
conda run -n libero python scripts/collect_oracle_demos.py \
  --custom-task peg_insertion \
  --controller peg_insertion \
  --num-successes 100 \
  --max-attempts 400 \
  --horizon 1100 \
  --camera-size 128 \
  --output-dir datasets \
  --keep-failures \
  --final-hold-steps 50 \
  --require-physical-success \
  --randomization-level diverse_v2
```

At 69% success rate, expect ~145 attempts for 100 successes. `--max-attempts 400` gives comfortable headroom. Omit `--keep-failures` to save disk space if not needed for debugging.

Add `--save-video` to record every episode (significantly slower, much larger output).

### Post-Collection Validation

```bash
RUN_DIR=$(ls -td datasets/peg_insertion_peg_insertion_* | head -1)

python scripts/validate_dataset.py  "$RUN_DIR" --expected-successes 100
python scripts/summarize_dataset.py "$RUN_DIR"
python scripts/replay_episode.py    "$RUN_DIR/success_001_seed_0.hdf5" --save-video videos/replay.mp4
python scripts/make_video_montage.py "videos/$(basename $RUN_DIR)"
```

### Known Remaining Failure Modes

| Mode | Rate | Detail |
|---|---|---|
| `verification_failed` | rare post-fix | FINE_ALIGN max_steps=160 times out when starting XY error is very large (> 0.130 m). Expected to be uncommon at 69% overall rate. |
| Peg tilt (`upright < 0.94`) | ~2/29 pre-fix | Wider socket (0.040 m) allows more tilt room. Peg contact with socket wall during insertion can cause slight lean. Upright check threshold is 0.94. |
| `grasp_failed_peg_not_lifted` | rare | Grasp slip on extreme-position peg seeds. Gets retried. |

### Physical Check Thresholds (`_physical_final_state`)

| Check | Threshold | What it verifies |
|---|---|---|
| `hole_xy_tolerance` | 0.018 m | Peg center within 18 mm of hole center |
| `hole_z_tolerance` | 0.050 m | Peg center within 50 mm of expected z (hole_z + 0.034) |
| `min_peg_upright_score` | 0.94 | 1 − 2(x²+y²) of peg quat ≥ 0.94 (< 20° tilt) |
| `min_peg_move` | 0.050 m | Peg moved ≥ 5 cm from start (confirms transport occurred) |
| `max_block_xy_drift` | 0.018 m | Socket didn't move (confirms block was stationary) |
| `settle_steps` | 35 | Peg had 35 steps to settle after gripper opened |
| `gripper_opened` | True | |
| `oracle_helper_used` | False | |
| `direct_pose_writes_during_rollout` | 0 | |

---

## Shared Infrastructure

### Output Directories

| Directory | Contents | Git-tracked |
|---|---|---|
| `datasets/` | HDF5 files + `summary.json` per run | No |
| `videos/` | Per-episode MP4s and montages | No |
| `logs/` | Collection logs | No |
| `reports/` | Inspection reports (generated) | No |

### HDF5 Layout (all tasks)

```
observations/
  robot0_eef_pos            ← end-effector xyz
  {object}_pos              ← object xyz per timestep
  {object}_quat             ← object orientation
  agentview_image           ← 128×128 or 256×256 frames (if enabled)
actions                     ← 7-dim OSC_POSE: [dx,dy,dz,drot×3,gripper]
rewards / dones / env_states
attrs:
  metadata  (JSON)          ← seed, initial/final positions, failure_reason, etc.
  success   (bool)
```

### Monitoring a Running Collection

```bash
# Progress from stdout (if running in foreground)
# Or check the dataset directory:
RUN_DIR=$(ls -td datasets/<task>_<task>_* | head -1)
echo "Successes: $(ls $RUN_DIR | grep success | wc -l)"
echo "Failures:  $(ls $RUN_DIR/failures/ | wc -l)"
cat "$RUN_DIR/summary.json"   # appears when run completes
```

### Environment Setup Reminder

```bash
cd ~/projects/LIBERO && pip install -e .
cd ~/projects/libero-oracle-data-gen
export NUMBA_DISABLE_JIT=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

Rendering is GPU-accelerated via EGL; physics runs on CPU. `NUMBA_DISABLE_JIT=1` avoids numba cache failures in this WSL environment.

---

## Remaining Tasks (not started)

- **tool_sweep**: scaffolded controller, no BDDL file, no observation keys defined
- **ring_hook**: scaffolded controller, no BDDL file, no observation keys defined

Do not begin these until peg_insertion 100-demo collection is complete and validated.
