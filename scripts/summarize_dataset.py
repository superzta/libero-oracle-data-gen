"""Generate a compact JSON and Markdown summary for a collected dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import h5py

    dataset_dir = Path(args.dataset_dir)
    files = sorted(dataset_dir.glob("success_*.hdf5"))
    failure_files = sorted((dataset_dir / "failures").glob("*.hdf5"))
    lengths = []
    seeds = []
    languages = Counter()
    initial_positions = {}
    for path in files:
        with h5py.File(path, "r") as h5:
            metadata = json.loads(h5.attrs.get("metadata", "{}"))
            seeds.append(metadata.get("seed"))
            languages[metadata.get("task_language", "unknown")] += 1
            lengths.append(int(h5["actions"].shape[0]))
            for key, ds in h5["observations"].items():
                if key.endswith("_pos") and ds.shape[0] > 0:
                    initial_positions.setdefault(key, []).append(np.asarray(ds[0]).reshape(-1))

    initial_summary = {}
    for key, values in initial_positions.items():
        arr = np.asarray(values, dtype=np.float32)
        initial_summary[key] = {
            "mean": arr.mean(axis=0).round(4).tolist(),
            "std": arr.std(axis=0).round(4).tolist(),
            "min": arr.min(axis=0).round(4).tolist(),
            "max": arr.max(axis=0).round(4).tolist(),
        }

    collection_summary_path = dataset_dir / "summary.json"
    collection_summary = json.loads(collection_summary_path.read_text()) if collection_summary_path.exists() else {}
    summary = {
        "dataset_dir": str(dataset_dir),
        "successes": len(files),
        "attempts": collection_summary.get("attempts", len(files) + len(failure_files)),
        "success_rate": collection_summary.get("success_rate"),
        "average_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "min_episode_length": int(np.min(lengths)) if lengths else 0,
        "max_episode_length": int(np.max(lengths)) if lengths else 0,
        "failure_reasons": collection_summary.get("failure_reasons", {}),
        "unique_seeds": len(set(seeds)),
        "task_languages": dict(languages),
        "initial_state_distribution": initial_summary,
    }
    output = Path(args.output) if args.output else Path("reports") / f"{dataset_dir.name}_summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("# Dataset Summary\n\n```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

