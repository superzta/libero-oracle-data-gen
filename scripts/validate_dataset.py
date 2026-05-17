"""Validate collected LIBERO oracle demonstration datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_episode(path: Path) -> Dict:
    import h5py

    with h5py.File(path, "r") as h5:
        metadata = json.loads(h5.attrs.get("metadata", "{}"))
        obs_keys = list(h5["observations"].keys()) if "observations" in h5 else []
        actions = np.asarray(h5["actions"])
        initial = {}
        final = {}
        for key in obs_keys:
            data = np.asarray(h5["observations"][key])
            if data.size and data.ndim >= 2 and key.endswith("_pos"):
                initial[key] = data[0].reshape(-1).tolist()
                final[key] = data[-1].reshape(-1).tolist()
        return {
            "path": str(path),
            "metadata": metadata,
            "success": bool(h5.attrs.get("success", False)),
            "obs_keys": obs_keys,
            "actions_shape": actions.shape,
            "initial": initial,
            "final": final,
        }


def get_video_frame_count(video_path: Path) -> Optional[int]:
    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(video_path)
        count = sum(1 for _ in reader)
        reader.close()
        return count
    except Exception:
        return None


def find_video_dir(dataset_dir: Path, video_dir_arg: Optional[str]) -> Optional[Path]:
    if video_dir_arg:
        return Path(video_dir_arg)
    manifest_path = dataset_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        vd = manifest.get("video_dir")
        if vd:
            return Path(vd)
    candidate = Path("videos") / dataset_dir.name
    if candidate.exists():
        return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--expected-successes", type=int, default=100)
    parser.add_argument("--required-obs-key", action="append", default=["robot0_eef_pos", "robot0_proprio-state"])
    parser.add_argument("--allow-oracle-state-helper", action="store_true", help="Permit episodes marked as helper-assisted.")
    parser.add_argument("--video-dir", default=None, help="Directory containing success_*.mp4 videos for frame count validation.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    episodes = [load_episode(path) for path in sorted(dataset_dir.glob("success_*.hdf5"))]
    errors: List[str] = []
    warnings: List[str] = []

    if len(episodes) != args.expected_successes:
        errors.append(f"expected {args.expected_successes} successful demos, found {len(episodes)}")
    seeds = [ep["metadata"].get("seed") for ep in episodes]
    if len(seeds) != len(set(seeds)):
        errors.append("duplicate seeds found")
    for ep in episodes:
        if not ep["success"] or not ep["metadata"].get("success", False):
            errors.append(f"success flag false: {ep['path']}")
        if ep["metadata"].get("oracle_state_helper_used", False) and not args.allow_oracle_state_helper:
            errors.append(f"oracle state helper used: {ep['path']}")
        if len(ep["actions_shape"]) != 2 or ep["actions_shape"][1] != 7:
            errors.append(f"invalid action shape {ep['actions_shape']}: {ep['path']}")
        missing = [key for key in args.required_obs_key if key not in ep["obs_keys"]]
        if missing:
            errors.append(f"missing observation keys {missing}: {ep['path']}")
        if "blue_cube_1_pos" in ep["initial"] and "open_box_1_pos" in ep["final"]:
            initial_cube = np.asarray(ep["initial"]["blue_cube_1_pos"], dtype=np.float32)
            final_cube = np.asarray(ep["final"]["blue_cube_1_pos"], dtype=np.float32)
            final_box = np.asarray(ep["final"]["open_box_1_pos"], dtype=np.float32)
            if np.linalg.norm(final_cube[:2] - initial_cube[:2]) <= 0.018:
                errors.append(f"blue cube did not move enough: {ep['path']}")
            if np.linalg.norm(final_cube[:2] - final_box[:2]) > 0.135:
                errors.append(f"blue cube final xy outside box region: {ep['path']}")
            if not (final_box[2] - 0.03 <= final_cube[2] <= final_box[2] + 0.18):
                errors.append(f"blue cube final z not box-consistent: {ep['path']}")

    initial_signatures = [json.dumps(ep["initial"], sort_keys=True) for ep in episodes]
    varied_initial_states = len(set(initial_signatures)) > 1 if episodes else False
    if args.expected_successes > 1 and not varied_initial_states:
        errors.append("initial states do not vary across successful episodes")
    cube_initials = [
        tuple(np.round(np.asarray(ep["initial"].get("blue_cube_1_pos", []), dtype=np.float32), 4))
        for ep in episodes
        if "blue_cube_1_pos" in ep["initial"]
    ]
    varied_cube_initial_states = len(set(cube_initials)) > 1 if cube_initials else False
    if args.expected_successes > 1 and cube_initials and not varied_cube_initial_states:
        errors.append("blue cube initial positions do not vary across successful episodes")

    # Video frame count consistency check
    video_dir = find_video_dir(dataset_dir, args.video_dir)
    video_check_results: List[Dict] = []
    if video_dir is not None and video_dir.exists():
        for ep in episodes:
            ep_hdf5 = Path(ep["path"])
            video_path = video_dir / (ep_hdf5.stem + ".mp4")
            if not video_path.exists():
                warnings.append(f"video not found for {ep_hdf5.name}: expected {video_path}")
                continue
            frame_count = get_video_frame_count(video_path)
            ep_len = ep["metadata"].get("episode_length", 0)
            known_stride = ep["metadata"].get("video_stride", 1)
            mismatch = frame_count is not None and abs(frame_count - ep_len) > 2
            if mismatch and known_stride == 1:
                warnings.append(
                    f"{ep_hdf5.name}: episode_length={ep_len} but video_frames={frame_count} "
                    f"with video_stride=1 — mismatch without known stride"
                )
            video_check_results.append({
                "hdf5": ep_hdf5.name,
                "video": video_path.name,
                "episode_length": ep_len,
                "video_frame_count": frame_count,
                "video_stride": known_stride,
                "mismatch": mismatch and known_stride == 1,
            })

    report = {
        "dataset_dir": str(dataset_dir),
        "num_successes": len(episodes),
        "unique_seeds": len(set(seeds)),
        "varied_initial_states": varied_initial_states,
        "varied_cube_initial_states": varied_cube_initial_states,
        "errors": errors,
        "warnings": warnings,
        "video_frame_checks": video_check_results,
        "valid": not errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
