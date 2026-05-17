"""Lightweight visual QA for rollout mp4 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    return [float(xs.mean()), float(ys.mean())]


def color_centroids(frame: np.ndarray) -> Dict[str, List[float]]:
    arr = frame.astype(np.int16)
    red = (arr[:, :, 0] > 120) & (arr[:, :, 0] > arr[:, :, 1] * 1.5) & (arr[:, :, 0] > arr[:, :, 2] * 1.5)
    blue = (arr[:, :, 2] > 120) & (arr[:, :, 2] > arr[:, :, 0] * 1.25) & (arr[:, :, 2] > arr[:, :, 1] * 1.25)
    out = {}
    red_c = centroid(red)
    blue_c = centroid(blue)
    if red_c is not None:
        out["red"] = red_c
    if blue_c is not None:
        out["blue"] = blue_c
    return out


def make_contact_sheet(frames: List[np.ndarray], output: Path, cols: int = 5) -> None:
    from PIL import Image, ImageDraw

    if not frames:
        return
    idxs = np.linspace(0, len(frames) - 1, min(15, len(frames))).astype(int).tolist()
    thumbs = []
    for idx in idxs:
        img = Image.fromarray(frames[idx]).resize((160, 160))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 52, 18), fill=(0, 0, 0))
        draw.text((4, 3), str(idx), fill=(255, 255, 255))
        thumbs.append(img)
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 160, rows * 160), (30, 30, 30))
    for i, img in enumerate(thumbs):
        sheet.paste(img, ((i % cols) * 160, (i // cols) * 160))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--output-dir", default="reports/video_qa")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min-duration", type=float, default=8.0)
    parser.add_argument("--max-red-drift-px", type=float, default=4.0)
    args = parser.parse_args()

    import imageio.v2 as imageio

    video = Path(args.video)
    reader = imageio.get_reader(video)
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 0.0) or 0.0)
    frames = [frame[:, :, :3] for frame in reader]
    frame_count = len(frames)
    duration = frame_count / fps if fps else 0.0
    samples = []
    for idx in np.linspace(0, frame_count - 1, min(30, frame_count)).astype(int).tolist() if frames else []:
        samples.append({"frame": idx, "centroids": color_centroids(frames[idx])})

    red_points = np.asarray([s["centroids"]["red"] for s in samples if "red" in s["centroids"]], dtype=np.float32)
    blue_points = np.asarray([s["centroids"]["blue"] for s in samples if "blue" in s["centroids"]], dtype=np.float32)
    red_drift = float(np.max(np.linalg.norm(red_points - red_points[0], axis=1))) if len(red_points) > 1 else None
    blue_motion = float(np.max(np.linalg.norm(blue_points - blue_points[0], axis=1))) if len(blue_points) > 1 else None
    warnings = []
    if duration < args.min_duration:
        warnings.append(f"duration {duration:.2f}s is shorter than {args.min_duration:.2f}s")
    if red_drift is None:
        warnings.append("red button centroid could not be tracked")
    elif red_drift > args.max_red_drift_px:
        warnings.append(f"red button centroid drift {red_drift:.2f}px > {args.max_red_drift_px:.2f}px")
    if blue_motion is None:
        warnings.append("blue cube centroid could not be tracked")
    elif blue_motion < 8.0:
        warnings.append(f"blue cube centroid motion {blue_motion:.2f}px is low")
    if frame_count >= 8:
        tail_delta = float(np.mean(np.abs(frames[-1].astype(np.float32) - frames[-8].astype(np.float32))))
        if tail_delta > 8.0:
            warnings.append(f"final frames are not visually stable, mean delta {tail_delta:.2f}")
    else:
        tail_delta = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = out_dir / "contact_sheet.jpg"
    make_contact_sheet(frames, contact_sheet)
    report = {
        "video": str(video),
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "red_centroid_drift_px": red_drift,
        "blue_centroid_motion_px": blue_motion,
        "final_tail_mean_abs_delta": tail_delta,
        "major_warnings": warnings,
        "passed": not warnings,
        "contact_sheet": str(contact_sheet),
        "samples": samples,
    }
    (out_dir / "video_qa.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and warnings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
