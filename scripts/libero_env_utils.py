"""Shared helpers for scripts that import LIBERO / robosuite."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


CUSTOM_TASKS = {
    "button_box": Path(__file__).resolve().parents[1] / "bddl_files" / "button_box.bddl",
    "peg_insertion": Path(__file__).resolve().parents[1] / "bddl_files" / "peg_insertion.bddl",
    "tool_sweep": Path(__file__).resolve().parents[1] / "bddl_files" / "tool_sweep.bddl",
}


def configure_runtime_env() -> None:
    """Set process environment before importing robosuite or LIBERO."""

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/libero_oracle_mplconfig")))
    os.environ.setdefault("MESA_SHADER_CACHE_DIR", str(Path("/tmp/libero_oracle_mesa_cache")))


def register_custom_objects() -> None:
    """Register this repo's custom object classes with LIBERO."""

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import custom_objects.libero_oracle_objects  # noqa: F401


def resolve_bddl_path(
    task: Optional[str] = None,
    suite: str = "libero_10",
    task_id: int = 0,
    bddl_file: Optional[str] = None,
    custom_task: Optional[str] = None,
) -> str:
    configure_runtime_env()
    if custom_task:
        register_custom_objects()
        if custom_task not in CUSTOM_TASKS:
            raise ValueError(f"Unknown custom task {custom_task!r}. Available: {sorted(CUSTOM_TASKS)}")
        return str(CUSTOM_TASKS[custom_task].resolve())
    if bddl_file:
        return str(Path(bddl_file).expanduser().resolve())

    from libero.libero import benchmark
    from libero.libero.utils import get_libero_path

    task_suite = benchmark.get_benchmark_dict()[suite]()
    if task is not None:
        names = task_suite.get_task_names()
        if task in names:
            task_id = names.index(task)
        else:
            matches = [i for i, item in enumerate(task_suite.tasks) if task.lower() in item.language.lower()]
            if not matches:
                raise ValueError(f"Task not found in {suite}: {task}")
            task_id = matches[0]
    task_obj = task_suite.get_task(task_id)
    return str(Path(get_libero_path("bddl_files")) / task_obj.problem_folder / task_obj.bddl_file)


def get_task_language(bddl_file: str) -> str:
    configure_runtime_env()
    import libero.libero.envs.bddl_utils as BDDLUtils

    return BDDLUtils.get_problem_info(bddl_file)["language_instruction"]
