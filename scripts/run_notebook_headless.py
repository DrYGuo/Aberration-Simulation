"""Execute a notebook non-interactively and save the executed copy.

This is a thin wrapper around the Jupyter CLI so Colab worker configs can run
notebooks without browser clicks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="Notebook path to execute.")
    parser.add_argument(
        "--output-dir",
        default=Path("colab_worker_logs"),
        type=Path,
        help="Directory for executed notebook and manifest.",
    )
    parser.add_argument(
        "--timeout",
        default="-1",
        help="Notebook execution timeout in seconds. Use -1 for no timeout.",
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="Optional kernel name passed to nbconvert.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebook = args.notebook.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jupyter = shutil.which("jupyter")
    if not jupyter:
        raise RuntimeError("jupyter executable not found")

    run_name = f"{notebook.stem}_executed_{utc_stamp()}"
    output_name = f"{run_name}.ipynb"
    manifest_path = output_dir / f"{run_name}.json"

    command = [
        jupyter,
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook),
        "--output",
        output_name,
        "--output-dir",
        str(output_dir),
        f"--ExecutePreprocessor.timeout={args.timeout}",
    ]
    if args.kernel:
        command.append(f"--ExecutePreprocessor.kernel_name={args.kernel}")

    print("$", " ".join(command), flush=True)
    started = datetime.now(timezone.utc).isoformat()
    output_parts: list[str] = []
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    for line in process.stdout:
        output_parts.append(line)
        print(line, end="" if line.endswith("\n") else "\n", flush=True)
    returncode = int(process.wait())
    finished = datetime.now(timezone.utc).isoformat()

    manifest = {
        "notebook": str(notebook),
        "executed_notebook": str(output_dir / output_name),
        "started_utc": started,
        "finished_utc": finished,
        "returncode": returncode,
        "command": command,
    }
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print("manifest:", manifest_path)

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
