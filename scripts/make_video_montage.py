"""Create a simple video montage from representative rollout videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-videos", type=int, default=9)
    args = parser.parse_args()

    import imageio.v2 as imageio

    video_dir = Path(args.video_dir)
    paths = sorted(video_dir.glob("*.mp4"))[: args.max_videos]
    if not paths:
        raise SystemExit(f"No mp4 files found in {video_dir}")
    readers = [imageio.get_reader(path) for path in paths]
    try:
        min_len = min(reader.count_frames() for reader in readers)
        sample0 = readers[0].get_data(0)
        h, w = sample0.shape[:2]
        cols = int(np.ceil(np.sqrt(len(readers))))
        rows = int(np.ceil(len(readers) / cols))
        out = Path(args.output) if args.output else video_dir / "montage.mp4"
        writer = imageio.get_writer(out, fps=20)
        for idx in range(min_len):
            canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
            for r_i, reader in enumerate(readers):
                frame = reader.get_data(idx)
                row, col = divmod(r_i, cols)
                canvas[row * h : (row + 1) * h, col * w : (col + 1) * w] = frame[:, :, :3]
            writer.append_data(canvas)
        writer.close()
    finally:
        for reader in readers:
            reader.close()
    print(f"wrote montage to {out}")


if __name__ == "__main__":
    main()

