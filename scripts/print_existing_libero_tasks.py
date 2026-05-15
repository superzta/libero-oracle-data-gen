from pathlib import Path

from libero_env_utils import configure_runtime_env

configure_runtime_env()
from libero.libero import benchmark


def main():
    benchmark_dict = benchmark.get_benchmark_dict()

    suite_names = [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90",
    ]

    lines = []
    for suite_name in suite_names:
        lines.append(f"\n===== {suite_name} =====")
        task_suite = benchmark_dict[suite_name]()
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            lines.append(f"{task_id:03d}: {task.language}")
    text = "\n".join(lines)
    print(text)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/existing_libero_tasks.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
