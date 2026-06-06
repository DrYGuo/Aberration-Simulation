"""Small Colab worker loop for regression experiments.

The worker is intentionally generic. It pulls the latest repository state,
runs a configured command, records a cycle manifest, and optionally commits
selected result artifacts back to GitHub.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


TOKEN_ENV_NAMES = ("GITHUB_TOKEN", "GH_TOKEN")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def find_token() -> str | None:
    for name in TOKEN_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    return None


def run_command(
    command: list[str],
    *,
    cwd: Path,
    check: bool = True,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    display = " ".join(shlex.quote(part) for part in command)
    print(f"$ {display}", flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as handle:
            handle.write(f"\n$ {display}\n")
            handle.write(result.stdout or "")
            if result.returncode:
                handle.write(f"\n[exit {result.returncode}]\n")
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result


def configure_token_remote(repo_root: Path, repo_slug: str, branch: str) -> None:
    token = find_token()
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN or GH_TOKEN is required when configure_token_remote is true."
        )
    url = f"https://x-access-token:{token}@github.com/{repo_slug}.git"
    subprocess.run(
        ["git", "remote", "set-url", "origin", url],
        cwd=repo_root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    safe_url = f"https://x-access-token:***@github.com/{repo_slug}.git"
    print(f"origin configured for {safe_url} on branch {branch}")


def git_pull(repo_root: Path, branch: str, log_path: Path) -> None:
    run_command(["git", "fetch", "origin", branch], cwd=repo_root, log_path=log_path)
    run_command(["git", "pull", "--ff-only", "origin", branch], cwd=repo_root, log_path=log_path)


def git_pull_rebase(repo_root: Path, branch: str, log_path: Path) -> None:
    run_command(["git", "pull", "--rebase", "origin", branch], cwd=repo_root, log_path=log_path)


def ensure_git_identity(repo_root: Path, log_path: Path) -> None:
    name = subprocess.run(
        ["git", "config", "--get", "user.name"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()
    email = subprocess.run(
        ["git", "config", "--get", "user.email"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()
    if not name:
        run_command(
            ["git", "config", "user.name", "Colab Regression Worker"],
            cwd=repo_root,
            log_path=log_path,
        )
    if not email:
        run_command(
            ["git", "config", "user.email", "colab-regression-worker@users.noreply.github.com"],
            cwd=repo_root,
            log_path=log_path,
        )


def git_commit_and_push(
    repo_root: Path,
    artifact_paths: list[Path],
    branch: str,
    message: str,
    log_path: Path,
) -> bool:
    existing_paths = [path for path in artifact_paths if path.exists()]
    if not existing_paths:
        print("No artifact paths exist; skipping git commit.")
        return False

    ensure_git_identity(repo_root, log_path)
    run_command(["git", "add", "--force", "--", *[str(path) for path in existing_paths]], cwd=repo_root, log_path=log_path)
    status = run_command(["git", "status", "--short", "--", *[str(path) for path in existing_paths]], cwd=repo_root, check=False)
    if not status.stdout.strip():
        print("No artifact changes to commit.")
        return False

    run_command(["git", "commit", "-m", message], cwd=repo_root, log_path=log_path)
    git_pull_rebase(repo_root, branch, log_path)
    run_command(["git", "push", "origin", branch], cwd=repo_root, log_path=log_path)
    return True


def expand_globs(repo_root: Path, patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for match in glob.glob(str(repo_root / pattern), recursive=True):
            path = Path(match)
            if path.is_file():
                rel = path.relative_to(repo_root)
                if rel not in seen:
                    seen.add(rel)
                    paths.append(rel)
    return paths


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def write_cycle_manifest(
    repo_root: Path,
    manifest_dir: Path,
    *,
    cycle: int,
    cycles: int,
    config_path: Path,
    command: list[str],
    returncode: int | None,
    started_utc: str,
    finished_utc: str,
    artifact_paths: list[Path],
) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"colab_worker_cycle_{cycle:02d}_{utc_stamp()}.json"
    payload = {
        "cycle": cycle,
        "cycles": cycles,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "config_path": display_path(config_path, repo_root),
        "command": command,
        "returncode": returncode,
        "artifact_paths": [str(path) for path in artifact_paths],
        "git_commit_after_cycle": current_commit(repo_root),
    }
    with manifest_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    return manifest_path


def current_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode:
        return None
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/colab_worker_5cycles.json",
        type=Path,
        help="JSON worker config path.",
    )
    parser.add_argument("--cycles", type=int, help="Override cycle count.")
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("--no-push", action="store_true", help="Run cycles without committing/pushing artifacts.")
    parser.add_argument(
        "--skip-token-remote",
        action="store_true",
        help="Do not rewrite origin to include GITHUB_TOKEN. Useful for local dry-runs.",
    )
    parser.add_argument("--skip-git-pull", action="store_true", help="Do not fetch/pull before cycles.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = load_json(config_path)

    cycles = int(args.cycles if args.cycles is not None else config.get("cycles", 1))
    branch = str(config.get("branch", "main"))
    reload_config_each_cycle = bool(config.get("reload_config_each_cycle", True))
    command = [str(part) for part in config.get("command", [])]
    if not command:
        raise RuntimeError("Worker config must define a non-empty command list.")

    manifest_dir = repo_root / str(config.get("artifact_manifest_dir", "colab_worker_logs"))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    worker_log_path = manifest_dir / f"colab_worker_{utc_stamp()}.log"

    if config.get("configure_token_remote", False) and not args.skip_token_remote:
        configure_token_remote(repo_root, str(config["repo_slug"]), branch)

    print(f"repo_root: {repo_root}")
    print(f"config: {config_path}")
    print(f"cycles: {cycles}")
    print(f"branch: {branch}")
    print(f"command: {' '.join(shlex.quote(part) for part in command)}")

    for cycle in range(1, cycles + 1):
        print(f"\n=== Colab worker cycle {cycle}/{cycles} ===", flush=True)
        started = datetime.now(timezone.utc).isoformat()
        returncode: int | None = None

        try:
            if config.get("git_pull_before_cycle", True) and not args.skip_git_pull:
                git_pull(repo_root, branch, worker_log_path)
                if reload_config_each_cycle:
                    config = load_json(config_path)
                    branch = str(config.get("branch", branch))
                    command = [str(part) for part in config.get("command", [])]
                    if not command:
                        raise RuntimeError("Worker config must define a non-empty command list.")
                    print(f"reloaded config after pull: {config_path}")
                    print(f"cycle command: {' '.join(shlex.quote(part) for part in command)}")

            result = run_command(command, cwd=repo_root, check=False, log_path=worker_log_path)
            returncode = result.returncode
            if returncode and config.get("stop_on_command_failure", True):
                print(f"Command failed with exit {returncode}; stopping worker.")

            artifact_paths = expand_globs(repo_root, list(config.get("result_globs", [])))
            finished = datetime.now(timezone.utc).isoformat()
            cycle_manifest = write_cycle_manifest(
                repo_root,
                manifest_dir,
                cycle=cycle,
                cycles=cycles,
                config_path=config_path,
                command=command,
                returncode=returncode,
                started_utc=started,
                finished_utc=finished,
                artifact_paths=artifact_paths,
            )
            artifact_paths.append(cycle_manifest.relative_to(repo_root))

            if config.get("git_push_after_cycle", False) and not args.no_push:
                message = f"{config.get('commit_message_prefix', 'Add Colab worker artifacts')} cycle {cycle}/{cycles}"
                git_commit_and_push(repo_root, artifact_paths, branch, message, worker_log_path)

            if returncode and config.get("stop_on_command_failure", True):
                return returncode

        except Exception as exc:
            finished = datetime.now(timezone.utc).isoformat()
            cycle_manifest = write_cycle_manifest(
                repo_root,
                manifest_dir,
                cycle=cycle,
                cycles=cycles,
                config_path=config_path,
                command=command,
                returncode=returncode,
                started_utc=started,
                finished_utc=finished,
                artifact_paths=[],
            )
            print(f"Cycle {cycle} failed: {exc}", file=sys.stderr)
            print(f"Wrote failure manifest: {cycle_manifest}")
            if config.get("stop_on_command_failure", True):
                return 1

        sleep_seconds = int(config.get("sleep_seconds", 0))
        if cycle < cycles and sleep_seconds > 0:
            print(f"Sleeping {sleep_seconds} seconds before next cycle.")
            time.sleep(sleep_seconds)

    print("Colab worker finished all cycles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
