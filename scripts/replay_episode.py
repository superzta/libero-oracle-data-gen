"""Replay a saved episode's action sequence in the matching LIBERO environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from libero_env_utils import configure_runtime_env

configure_runtime_env()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode")
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--save-video", default=None)
    parser.add_argument("--camera", default="agentview")
    args = parser.parse_args()

    import h5py
    import imageio.v2 as imageio
    import numpy as np
    from libero.libero.envs import OffScreenRenderEnv

    episode = Path(args.episode)
    with h5py.File(episode, "r") as h5:
        metadata = json.loads(h5.attrs["metadata"])
        actions = np.asarray(h5["actions"])
    env = OffScreenRenderEnv(
        bddl_file_name=metadata["bddl_file"],
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
    )
    frames = []
    try:
        env.seed(int(metadata["seed"]))
        env.reset()
        for action in actions:
            env.step(action)
            if args.save_video:
                frames.append(env.env.sim.render(camera_name=args.camera, height=args.camera_size, width=args.camera_size)[::-1])
    finally:
        env.close()
    if args.save_video and frames:
        out = Path(args.save_video)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, frames, fps=20)
    print(f"replayed {len(actions)} actions from {episode}")


if __name__ == "__main__":
    main()
