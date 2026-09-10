#!/usr/bin/env python3
"""Fail-closed W&B guard and detached seed-job supervisor for Runpod."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


EXPECTED_ENTITY = "rundong-liu"
EXPECTED_PROJECT = "JAXUED_TEST"
API_KEY_NAME = "WANDB_API_KEY"


class GuardError(RuntimeError):
    """Raised when a launch or W&B target invariant fails."""


def read_dotenv_value(path: Path, key: str) -> str:
    """Read one literal dotenv value without evaluating shell expressions."""
    if not path.is_file():
        raise GuardError(f"dotenv file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, raw_value = line.partition("=")
        if separator and name.strip() == key:
            try:
                parsed = shlex.split(raw_value, comments=True, posix=True)
            except ValueError as error:
                raise GuardError(f"invalid {key} entry in {path}") from error
            if len(parsed) != 1 or not parsed[0]:
                raise GuardError(f"{key} must have one non-empty literal value in {path}")
            return parsed[0]
    raise GuardError(f"{key} is missing from {path}")


def target_environment(env_file: Path) -> dict[str, str]:
    """Build the only environment permitted for W&B logging."""
    environment = dict(os.environ)
    environment.update(
        {
            API_KEY_NAME: read_dotenv_value(env_file, API_KEY_NAME),
            "WANDB_ENTITY": EXPECTED_ENTITY,
            "WANDB_PROJECT": EXPECTED_PROJECT,
            "WANDB_MODE": "online",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def wandb_api(env_file: Path):
    """Authenticate only with the key read from the requested dotenv file."""
    environment = target_environment(env_file)
    try:
        import wandb
    except ImportError as error:
        raise GuardError("wandb is not installed in this Python environment") from error
    api = wandb.Api(api_key=environment[API_KEY_NAME], timeout=30)
    if api.default_entity != EXPECTED_ENTITY:
        raise GuardError(
            "W&B credential default entity mismatch: "
            f"expected {EXPECTED_ENTITY!r}, got {api.default_entity!r}"
        )
    return api


def preflight(env_file: Path) -> dict[str, Any]:
    """Verify credential identity and access to the canonical project."""
    api = wandb_api(env_file)
    try:
        iterator = iter(api.runs(f"{EXPECTED_ENTITY}/{EXPECTED_PROJECT}", per_page=1))
        next(iterator, None)
    except Exception as error:
        raise GuardError(
            f"cannot access {EXPECTED_ENTITY}/{EXPECTED_PROJECT}: {error}"
        ) from error
    return {"ok": True, "entity": EXPECTED_ENTITY, "project": EXPECTED_PROJECT}


def option_value(command: Sequence[str], option: str) -> str | None:
    """Return a command-line option's value for --option value or --option=value."""
    for index, token in enumerate(command):
        if token == option:
            if index + 1 >= len(command):
                raise GuardError(f"{option} has no value")
            return command[index + 1]
        prefix = f"{option}="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def validate_command(command: Sequence[str], seed: int) -> None:
    """Reject commands that can log a different seed or W&B destination."""
    if not command:
        raise GuardError("training command is empty")
    command_seed = option_value(command, "--seed")
    if command_seed != str(seed):
        raise GuardError(f"command must contain --seed {seed}")
    project = option_value(command, "--project")
    if project != EXPECTED_PROJECT:
        raise GuardError(f"command must contain --project {EXPECTED_PROJECT}")
    entity = option_value(command, "--entity")
    if entity is not None and entity != EXPECTED_ENTITY:
        raise GuardError(f"--entity must be {EXPECTED_ENTITY}")


def atomic_write(path: Path, value: str) -> None:
    """Atomically replace a small state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def pid_is_running(pid: int) -> bool:
    """Check whether a local supervisor PID still exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def launch(
    env_file: Path,
    seed: int,
    state_dir: Path,
    command: Sequence[str],
) -> dict[str, Any]:
    """Preflight, validate, and start one detached supervisor."""
    preflight(env_file)
    validate_command(command, seed)
    state_dir.mkdir(parents=True, exist_ok=True)
    pid_path = state_dir / "supervisor.pid"
    exit_path = state_dir / "exit_code"
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError as error:
            raise GuardError(f"invalid existing PID file: {pid_path}") from error
        if pid_is_running(old_pid):
            raise GuardError(f"seed supervisor is already running with PID {old_pid}")
    if exit_path.exists():
        raise GuardError(f"state directory already contains a completed attempt: {state_dir}")

    manifest = {
        "seed": seed,
        "entity": EXPECTED_ENTITY,
        "project": EXPECTED_PROJECT,
        "command": list(command),
        "launched_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write(state_dir / "job.json", json.dumps(manifest, indent=2) + "\n")
    supervisor_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_supervise",
        "--env-file",
        str(env_file.resolve()),
        "--state-dir",
        str(state_dir.resolve()),
        "--",
        *command,
    ]
    process = subprocess.Popen(
        supervisor_command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    atomic_write(pid_path, f"{process.pid}\n")
    return {
        "ok": True,
        "seed": seed,
        "pid": process.pid,
        "state_dir": str(state_dir),
        "entity": EXPECTED_ENTITY,
        "project": EXPECTED_PROJECT,
    }


def supervise(env_file: Path, state_dir: Path, command: Sequence[str]) -> int:
    """Run the child synchronously and persist its exit code."""
    log_path = state_dir / "train.log"
    exit_path = state_dir / "exit_code"
    with log_path.open("ab", buffering=0) as log_handle:
        completed = subprocess.run(
            list(command),
            env=target_environment(env_file),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    atomic_write(exit_path, f"{completed.returncode}\n")
    return completed.returncode


def status(state_dir: Path) -> dict[str, Any]:
    """Return supervisor state without reading or exposing credentials."""
    pid_path = state_dir / "supervisor.pid"
    exit_path = state_dir / "exit_code"
    pid = int(pid_path.read_text(encoding="utf-8").strip()) if pid_path.exists() else None
    exit_code = (
        int(exit_path.read_text(encoding="utf-8").strip()) if exit_path.exists() else None
    )
    return {
        "state_dir": str(state_dir),
        "pid": pid,
        "running": pid_is_running(pid) if pid is not None and exit_code is None else False,
        "exit_code": exit_code,
        "log": str(state_dir / "train.log"),
        "log_bytes": (state_dir / "train.log").stat().st_size
        if (state_dir / "train.log").exists() else 0,
    }


def history_max_updates(run: Any) -> int | float | None:
    """Read maximum num_updates with a sampled-history fallback."""
    summary_value = dict(run.summary).get("num_updates")
    if isinstance(summary_value, (int, float)):
        return summary_value
    rows = run.history(keys=["num_updates"], samples=10_000, pandas=False)
    values = [row.get("num_updates") for row in rows]
    numeric_values = [value for value in values if isinstance(value, (int, float))]
    return max(numeric_values, default=None)


def verify_run(
    env_file: Path,
    run_id: str,
    expected_seed: int,
    min_updates: int | None,
    require_finished: bool,
) -> dict[str, Any]:
    """Verify an observed run under the one permitted entity/project path."""
    api = wandb_api(env_file)
    path = f"{EXPECTED_ENTITY}/{EXPECTED_PROJECT}/{run_id}"
    try:
        run = api.run(path)
    except Exception as error:
        raise GuardError(f"cannot read expected W&B run {path}: {error}") from error
    actual_seed = run.config.get("seed")
    if actual_seed != expected_seed:
        raise GuardError(
            f"W&B seed mismatch for {path}: expected {expected_seed}, got {actual_seed!r}"
        )
    if require_finished and run.state != "finished":
        raise GuardError(f"W&B run {path} is {run.state!r}, not 'finished'")
    max_updates = history_max_updates(run) if min_updates is not None else None
    if min_updates is not None and (max_updates is None or max_updates < min_updates):
        raise GuardError(
            f"W&B run {path} has max num_updates {max_updates!r}, expected >= {min_updates}"
        )
    return {
        "ok": True,
        "entity": EXPECTED_ENTITY,
        "project": EXPECTED_PROJECT,
        "run_id": run.id,
        "url": run.url,
        "seed": actual_seed,
        "state": run.state,
        "max_num_updates": max_updates,
    }


def parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="action", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--env-file", type=Path, required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--env-file", type=Path, required=True)
    launch_parser.add_argument("--seed", type=int, required=True)
    launch_parser.add_argument("--state-dir", type=Path, required=True)
    launch_parser.add_argument("command", nargs=argparse.REMAINDER)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--state-dir", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify-run")
    verify_parser.add_argument("--env-file", type=Path, required=True)
    verify_parser.add_argument("--run-id", required=True)
    verify_parser.add_argument("--expected-seed", type=int, required=True)
    verify_parser.add_argument("--min-updates", type=int)
    verify_parser.add_argument("--require-finished", action="store_true")

    supervise_parser = subparsers.add_parser("_supervise", help=argparse.SUPPRESS)
    supervise_parser.add_argument("--env-file", type=Path, required=True)
    supervise_parser.add_argument("--state-dir", type=Path, required=True)
    supervise_parser.add_argument("command", nargs=argparse.REMAINDER)
    return root


def stripped_command(command: Sequence[str]) -> list[str]:
    """Remove argparse's optional remainder separator."""
    return list(command[1:] if command and command[0] == "--" else command)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one helper action and emit safe JSON."""
    args = parser().parse_args(argv)
    try:
        if args.action == "preflight":
            result = preflight(args.env_file)
        elif args.action == "launch":
            if args.seed < 0:
                raise GuardError("seed must be non-negative")
            result = launch(
                args.env_file,
                args.seed,
                args.state_dir,
                stripped_command(args.command),
            )
        elif args.action == "status":
            result = status(args.state_dir)
        elif args.action == "verify-run":
            result = verify_run(
                args.env_file,
                args.run_id,
                args.expected_seed,
                args.min_updates,
                args.require_finished,
            )
        elif args.action == "_supervise":
            return supervise(
                args.env_file,
                args.state_dir,
                stripped_command(args.command),
            )
        else:
            raise GuardError(f"unknown action: {args.action}")
    except GuardError as error:
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
