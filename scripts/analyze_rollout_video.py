"""Lightweight visual QA for rollout mp4 files.

Accepts either a single video file or a directory of mp4s.
When given a directory, all *.mp4 files are analyzed and a combined report
is written; each video also gets its own contact sheet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def centroid(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    return [float(xs.mean()), float(ys.mean())]


def color_centroids(frame: np.ndarray) -> Dict[str, List[float]]:
    arr = frame.astype(np.int16)
    red = (arr[:, :, 0] > 120) & (arr[:, :, 0] > arr[:, :, 1] * 1.5) & (arr[:, :, 0] > arr[:, :, 2] * 1.5)
    blue = (arr[:, :, 2] > 120) & (arr[:, :, 2] > arr[:, :, 0] * 1.25) & (arr[:, :, 2] > arr[:, :, 1] * 1.25)
    out: Dict[str, List[float]] = {}
    red_c = centroid(red)
    blue_c = centroid(blue)
    if red_c is not None:
        out["red"] = red_c
    if blue_c is not None:
        out["blue"] = blue_c
    return out


def make_contact_sheet(
    frames: List[np.ndarray],
    output: Path,
    n_frames: int = 25,
    cols: int = 5,
    thumb_size: int = 160,
) -> None:
    from PIL import Image, ImageDraw

    if not frames:
        return
    idxs = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int).tolist()
    thumbs = []
    for idx in idxs:
        img = Image.fromarray(frames[idx]).resize((thumb_size, thumb_size))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 52, 18), fill=(0, 0, 0))
        draw.text((4, 3), str(idx), fill=(255, 255, 255))
        thumbs.append(img)
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (30, 30, 30))
    for i, img in enumerate(thumbs):
        sheet.paste(img, ((i % cols) * thumb_size, (i // cols) * thumb_size))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def analyze_single_video(
    video_path: Path,
    out_dir: Path,
    min_duration: float,
    max_red_drift_px: float,
    contact_sheet_frames: int = 25,
) -> Dict:
    import imageio.v2 as imageio

    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 0.0) or 0.0)
    frames = [frame[:, :, :3] for frame in reader]
    reader.close()
    frame_count = len(frames)
    duration = frame_count / fps if fps else 0.0

    samples = []
    sample_idxs = np.linspace(0, frame_count - 1, min(30, frame_count)).astype(int).tolist() if frames else []
    for idx in sample_idxs:
        samples.append({"frame": idx, "centroids": color_centroids(frames[idx])})

    red_points = np.asarray(
        [s["centroids"]["red"] for s in samples if "red" in s["centroids"]], dtype=np.float32
    )
    blue_points = np.asarray(
        [s["centroids"]["blue"] for s in samples if "blue" in s["centroids"]], dtype=np.float32
    )
    red_drift = float(np.max(np.linalg.norm(red_points - red_points[0], axis=1))) if len(red_points) > 1 else None
    blue_motion = (
        float(np.max(np.linalg.norm(blue_points - blue_points[0], axis=1))) if len(blue_points) > 1 else None
    )

    # Scale thresholds by image width so they work at both 64px and 256px.
    # At 64px: red_drift_limit≈3.2px, blue_motion_limit≈4.5px
    # At 256px: red_drift_limit≈12.8px, blue_motion_limit≈18px
    # Base blue_motion set to 4.5 (not 8.0): seeds with short horizontal trajectories
    # (cube starts near box) can have pixel centroids shift only ~11% of frame width.
    frame_w = frames[0].shape[1] if frames else 64
    drift_scale = frame_w / 64.0
    effective_max_red_drift = max_red_drift_px * drift_scale
    effective_min_blue_motion = 4.5 * drift_scale

    warnings: List[str] = []
    if duration < min_duration:
        warnings.append(f"duration {duration:.2f}s shorter than {min_duration:.2f}s")
    if red_drift is None:
        warnings.append("red button centroid could not be tracked")
    elif red_drift > effective_max_red_drift:
        warnings.append(f"red button centroid drift {red_drift:.2f}px > {effective_max_red_drift:.2f}px")
    if blue_motion is None:
        warnings.append("blue cube centroid could not be tracked")
    elif blue_motion < effective_min_blue_motion:
        warnings.append(f"blue cube centroid motion {blue_motion:.2f}px is low")

    tail_delta = None
    if frame_count >= 8:
        tail_delta = float(
            np.mean(np.abs(frames[-1].astype(np.float32) - frames[-8].astype(np.float32)))
        )
        if tail_delta > 8.0:
            warnings.append(f"final frames not visually stable, mean delta {tail_delta:.2f}")

    contact_sheet_path = out_dir / f"contact_sheet_{video_path.stem}.jpg"
    make_contact_sheet(frames, contact_sheet_path, n_frames=contact_sheet_frames)

    return {
        "video": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "red_centroid_drift_px": red_drift,
        "blue_centroid_motion_px": blue_motion,
        "final_tail_mean_abs_delta": tail_delta,
        "warnings": warnings,
        "passed": not warnings,
        "contact_sheet": str(contact_sheet_path),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual QA for button_box rollout videos. "
        "Accepts a single mp4 file or a directory of mp4s."
    )
    parser.add_argument("video", help="Path to a video file or directory of mp4 files")
    parser.add_argument("--output-dir", default="reports/video_qa")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min-duration", type=float, default=8.0)
    parser.add_argument("--max-red-drift-px", type=float, default=4.0)
    parser.add_argument("--contact-sheet-frames", type=int, default=25, help="Number of frames in each contact sheet (20-30)")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.output_dir)

    if video_path.is_dir():
        mp4_files = sorted(video_path.glob("*.mp4"))
        if not mp4_files:
            raise SystemExit(f"No mp4 files found in {video_path}")
        per_video = []
        for mp4 in mp4_files:
            print(f"Analyzing {mp4.name} ...")
            result = analyze_single_video(
                mp4, out_dir, args.min_duration, args.max_red_drift_px,
                contact_sheet_frames=args.contact_sheet_frames,
            )
            per_video.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {status}: {mp4.name} | {result['frame_count']} frames | {result['duration']:.1f}s | warnings: {result['warnings']}")

        all_passed = all(r["passed"] for r in per_video)
        report = {
            "video_dir": str(video_path),
            "num_videos": len(per_video),
            "all_passed": all_passed,
            "per_video": [{k: v for k, v in r.items() if k != "samples"} for r in per_video],
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "video_qa.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in report.items() if k != "per_video"}, indent=2))
        if args.strict and not all_passed:
            raise SystemExit(1)

    else:
        result = analyze_single_video(
            video_path, out_dir, args.min_duration, args.max_red_drift_px,
            contact_sheet_frames=args.contact_sheet_frames,
        )
        # legacy compat: keep "major_warnings" key for single-file mode
        result["major_warnings"] = result["warnings"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "video_qa.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.strict and result["warnings"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
