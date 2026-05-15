import os
from pathlib import Path

from libero_env_utils import configure_runtime_env

configure_runtime_env()
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path


def main():
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    task = task_suite.get_task(0)

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    print("Task language:", task.language)
    print("BDDL file:", task_bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=128,
        camera_widths=128,
    )

    env.seed(0)
    obs = env.reset()
    print("Reset OK")
    print("Observation keys:", list(obs.keys()))

    for _ in range(10):
        action = [0, 0, 0, 0, 0, 0, -1]
        obs, reward, done, info = env.step(action)

    print("LIBERO external repo test OK")
    env.close()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/test_builtin_libero.txt").write_text(
        f"Task language: {task.language}\nBDDL file: {task_bddl_file}\nObservation keys: {list(obs.keys())}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
