"""Inspect installed LIBERO paths, task suites, and extension points."""

from __future__ import annotations

from pathlib import Path

from libero_env_utils import configure_runtime_env

configure_runtime_env()


def main() -> None:
    from libero.libero import benchmark
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING
    from libero.libero.utils import get_libero_path

    roots = {
        "bddl_files": Path(get_libero_path("bddl_files")),
        "init_states": Path(get_libero_path("init_states")),
        "datasets": Path(get_libero_path("datasets")),
        "libero_root": Path(get_libero_path("bddl_files")).parents[1],
    }
    lines = ["# LIBERO Structure Inspection", ""]
    for key, value in roots.items():
        lines.append(f"{key}: {value} exists={value.exists()}")

    lines += ["", "## Registered Problem Classes"]
    for name in sorted(TASK_MAPPING):
        lines.append(f"- {name}")

    lines += ["", "## Benchmark Suites"]
    for suite_name, suite_cls in sorted(benchmark.get_benchmark_dict().items()):
        try:
            suite = suite_cls()
            lines.append(f"- {suite_name}: {suite.n_tasks} tasks")
        except Exception as exc:
            lines.append(f"- {suite_name}: unavailable ({type(exc).__name__}: {exc})")

    bddl_root = roots["bddl_files"]
    lines += ["", "## BDDL Folders"]
    for child in sorted(p for p in bddl_root.iterdir() if p.is_dir()):
        lines.append(f"- {child.name}: {len(list(child.glob('*.bddl')))} bddl files")

    text = "\n".join(lines) + "\n"
    print(text)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/libero_structure.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
