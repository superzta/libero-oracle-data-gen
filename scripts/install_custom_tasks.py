"""Install external custom BDDL files into a local editable LIBERO checkout.

This script is intentionally conservative: by default it prints the planned
copy operations and does not modify `~/projects/LIBERO`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", default="~/projects/LIBERO")
    parser.add_argument("--suite-name", default="libero_oracle_custom")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--apply", action="store_true", help="Actually copy files.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "bddl_files"
    libero_root = Path(args.libero_root).expanduser().resolve()
    target = libero_root / "libero" / "libero" / "bddl_files" / args.suite_name
    files = sorted(source.glob("*.bddl"))
    if not files:
        print(f"No custom BDDL files found in {source}")
        return
    print(f"Source: {source}")
    print(f"Target: {target}")
    for path in files:
        print(f"copy {path.name} -> {target / path.name}")
    if not args.apply or args.dry_run:
        print("Dry run only. Re-run with --apply to copy files.")
        return
    target.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copy2(path, target / path.name)
    print("Custom BDDL files copied. If new object classes or problem classes are required, add them to LIBERO through a reviewed patch.")


if __name__ == "__main__":
    main()
