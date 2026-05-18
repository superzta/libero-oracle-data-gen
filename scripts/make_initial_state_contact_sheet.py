"""Create a contact sheet of initial frames (frame 0) across all success videos.

Each thumbnail is labelled with:  seed | cube_bin=<id> | box_bin=<id>

Usage:
  python scripts/make_initial_state_contact_sheet.py <dataset_dir> \
      --video-dir videos/<run_name> \
      --output reports/<run_name>_initial_states.jpg \
      [--cols 5] [--thumb-size 256]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_hdf5_metas(dataset_dir: Path) -> Dict[str, dict]:
    """Map success_NNN_seed_SSS.hdf5 stem → metadata dict."""
    import h5py

    out = {}
    for path in sorted(dataset_dir.glob("success_*.hdf5")):
        try:
            with h5py.File(path, "r") as h5:
                out[path.stem] = json.loads(h5.attrs.get("metadata", "{}"))
        except Exception:
            out[path.stem] = {}
    return out


def load_first_frames(
    video_dir: Path,
    dataset_dir: Optional[Path] = None,
    max_videos: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[dict]]:
    """Return (frames, per-frame metadata list) from success_*.mp4."""
    import imageio.v2 as imageio

    hdf5_metas = _load_hdf5_metas(dataset_dir) if dataset_dir else {}

    frames, frame_metas = [], []
    mp4s = sorted(video_dir.glob("success_*.mp4"))
    if max_videos:
        mp4s = mp4s[:max_videos]
    for mp4 in mp4s:
        try:
            reader = imageio.get_reader(str(mp4))
            frame = next(iter(reader))[:, :, :3]
            reader.close()
            frames.append(frame)
        except Exception:
            continue
        # Try to match metadata by stem
        meta = hdf5_metas.get(mp4.stem, {})
        frame_metas.append(meta)
    return frames, frame_metas


def make_grid(
    frames: List[np.ndarray],
    frame_metas: List[dict],
    cols: int,
    thumb_size: int,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    thumbs = []
    for f in frames:
        img = Image.fromarray(f).resize((thumb_size, thumb_size), Image.LANCZOS)
        thumbs.append(np.asarray(img))

    rows = (len(thumbs) + cols - 1) // cols
    h, w = thumb_size, thumb_size
    grid = np.full((rows * h, cols * w, 3), 30, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = thumb

    try:
        img = Image.fromarray(grid)
        draw = ImageDraw.Draw(img)
        for idx, meta in enumerate(frame_metas):
            r, c = divmod(idx, cols)
            x0, y0 = c * w + 4, r * h + 4

            seed = meta.get("seed", "?")
            cb = meta.get("cube_bin_id", "?")
            bb = meta.get("box_bin_id", "?")
            btn = meta.get("button_bin_id", "?")
            line1 = f"s{seed} btn{btn}"
            line2 = f"cb{cb} bb{bb}"

            for line, dy in [(line1, 0), (line2, 14)]:
                draw.text((x0 + 1, y0 + 1 + dy), line, fill=(0, 0, 0))
                draw.text((x0, y0 + dy), line, fill=(255, 255, 80))
        grid = np.asarray(img)
    except Exception:
        pass

    return grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a contact sheet of initial frames from success videos."
    )
    parser.add_argument("dataset_dir", help="Dataset directory (for HDF5 metadata lookup).")
    parser.add_argument("--video-dir", default=None, help="Directory containing success_*.mp4 files.")
    parser.add_argument("--output", default=None, help="Output JPEG path.")
    parser.add_argument("--cols", type=int, default=5, help="Columns in grid (default 5).")
    parser.add_argument("--thumb-size", type=int, default=256, help="Thumbnail size px (default 256).")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    run_name = dataset_dir.name

    # Resolve video dir
    if args.video_dir:
        video_dir = Path(args.video_dir)
    else:
        manifest_path = dataset_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            vd = manifest.get("video_dir")
            if vd:
                video_dir = Path(vd)
                if not video_dir.is_absolute():
                    video_dir = Path(__file__).resolve().parents[1] / video_dir
            else:
                video_dir = Path("videos") / run_name
        else:
            video_dir = Path("videos") / run_name

    if not video_dir.exists():
        raise SystemExit(f"Video directory not found: {video_dir}")

    print(f"Loading first frames from {video_dir} ...")
    frames, frame_metas = load_first_frames(video_dir, dataset_dir=dataset_dir, max_videos=args.max_videos)
    if not frames:
        raise SystemExit(f"No success_*.mp4 found in {video_dir}")
    print(f"  {len(frames)} videos  ({sum(1 for m in frame_metas if m.get('cube_bin_id', -1) >= 0)} with bin metadata)")

    grid = make_grid(frames, frame_metas, cols=args.cols, thumb_size=args.thumb_size)

    if args.output:
        output_path = Path(args.output)
    else:
        reports_dir = Path(__file__).resolve().parents[1] / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / f"{run_name}_initial_states.jpg"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    Image.fromarray(grid).save(str(output_path), quality=92)
    print(f"Saved: {output_path}  ({grid.shape[1]}×{grid.shape[0]} px, {len(frames)} frames, {args.cols} cols)")


if __name__ == "__main__":
    main()
