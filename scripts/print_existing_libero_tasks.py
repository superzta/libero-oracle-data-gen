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

    for suite_name in suite_names:
        print(f"\n===== {suite_name} =====")
        task_suite = benchmark_dict[suite_name]()
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            print(f"{task_id:03d}: {task.language}")


if __name__ == "__main__":
    main()