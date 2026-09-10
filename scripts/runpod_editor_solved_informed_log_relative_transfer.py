"""Launch one L40S pod per seed; run from your local checkout, not from a pod.

    .venv/bin/python scripts/runpod_editor_solved_informed_log_relative_transfer.py
    .venv/bin/python scripts/runpod_editor_solved_informed_log_relative_transfer.py --dry-run

Requires runpodctl >= 2.11, registered local SSH key, and WANDB_API_KEY in .env.
Actual execution incurs Runpod charges. No creation is retried automatically.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from typing import Any, Callable
import uuid

from pydantic import BaseModel, ConfigDict, Field

try:
    from scripts import seed_job
    from scripts.editor_solved_informed_log_relative_transfer_three_seeds import (
        LocalSweep, build_seed_command,
    )
except ModuleNotFoundError:
    import seed_job
    from editor_solved_informed_log_relative_transfer_three_seeds import (
        LocalSweep, build_seed_command,
    )


# USER SETTINGS: changing NUM_PODS changes both pod count and seeds (0 .. N-1).
NUM_PODS = 6
NUM_UPDATES = 30_000
STOP_AFTER_HOURS = 168  # Safety deadline, NOT completion detection. Storage still bills.
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
GPU = "NVIDIA L40S"
REPOSITORY = Path(__file__).resolve().parents[1]
REMOTE = "/workspace/jaxued"
UV_VERSION = "0.8.15"


class SweepConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    num_pods: int = Field(default=NUM_PODS, gt=0)
    num_updates: int = Field(default=NUM_UPDATES, gt=0)
    stop_after_hours: int = Field(default=STOP_AFTER_HOURS, gt=0)
    startup_timeout: int = Field(default=1800, gt=0)
    image: str = Field(default=IMAGE, min_length=1)


@dataclass(frozen=True)
class SeedPlan:
    seed: int
    name: str
    run_id: str
    state_dir: str
    command: tuple[str, ...]


class CommandFailure(RuntimeError):
    def __init__(self, program: str, payload: dict[str, Any] | None = None):
        self.payload = payload or {}
        # Do not echo arbitrary command output: it could contain credentials.
        super().__init__(f"{program} failed ({self.payload.get('code', 'check pod setup.log')})")


def command(args: list[str], *, cwd: Path = REPOSITORY,
            data: bytes | None = None, timeout: int = 120) -> bytes:
    """Subprocess boundary: capture output; never print credentials or shell-evaluate argv."""
    result = subprocess.run(args, cwd=cwd, input=data, capture_output=True, timeout=timeout)
    if result.returncode:
        payload = {}
        for stream in (result.stderr, result.stdout):
            try:
                candidate = json.loads(stream)
                if isinstance(candidate, dict):
                    payload.update(candidate)
            except (ValueError, UnicodeDecodeError):
                pass
        raise CommandFailure(args[0], payload)
    return result.stdout


def cli(*args: str, timeout: int = 120) -> Any:
    result = json.loads(command(["runpodctl", *args, "-o", "json"], timeout=timeout))
    if isinstance(result, dict) and "error" in result:
        raise CommandFailure("runpodctl", result)
    return result


def make_plans(config: SweepConfig, group: str) -> tuple[SeedPlan, ...]:
    sweep = LocalSweep(run_name=group, num_updates=config.num_updates,
                       seeds=tuple(range(config.num_pods)))
    return tuple(SeedPlan(
        seed=seed, name=f"{group}-seed-{seed}", run_id=f"{group}-{seed}",
        state_dir=f"runpod-runs/{group}/seed-{seed}",
        command=build_seed_command(sweep, seed, python_executable=".venv/bin/python")
        + ("--wandb_experiment_name", f"solved-logrel-seed-{seed}"),
    ) for seed in sweep.seeds)


def create_args(config: SweepConfig, plan: SeedPlan, stop_at: str) -> tuple[str, ...]:
    return ("pod", "create", "--name", plan.name, "--image", config.image,
            "--gpu-id", GPU, "--gpu-count", "1", "--cloud-type", "SECURE",
            "--container-disk-in-gb", "30", "--volume-in-gb", "100",
            "--volume-mount-path", "/workspace", "--ports", "22/tcp", "--ssh",
            "--stop-after", stop_at, "--wait", "--wait-timeout", "20m")


def build_snapshot(repo: Path, destination: Path) -> str:
    """Include current tracked/untracked runtime files, never .env, caches or results."""
    names = command(["git", "ls-files", "-z", "--cached", "--others",
                     "--exclude-standard", "--", "src", "examples", "scripts",
                     "pyproject.toml", "uv.lock", "README.md", "LICENSE"], cwd=repo)
    allowed_roots = {"src", "examples", "scripts"}
    with tarfile.open(destination, "w:gz") as archive:
        for raw in sorted(set(names.split(b"\0")) - {b""}):
            relative = Path(os.fsdecode(raw))
            source = repo / relative
            if (relative.is_absolute() or ".." in relative.parts
                    or any(part.startswith(".") or part == "__pycache__"
                           for part in relative.parts)):
                continue
            if relative.parts[0] in allowed_roots and relative.suffix not in {".py", ".sh"}:
                continue
            if source.is_symlink() or any(p.is_symlink() for p in source.parents if p != repo):
                raise RuntimeError(f"Refusing snapshot symlink: {relative}")
            if source.is_file():
                contents = source.read_bytes()
                info = tarfile.TarInfo(relative.as_posix())
                info.size, info.mode = len(contents), 0o644
                archive.addfile(info, io.BytesIO(contents))
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def select_ssh_key(override: Path | None) -> Path:
    """Require a local, unencrypted private key matching an already registered key."""
    registered = cli("ssh", "list-keys")["keys"]
    public_keys = {tuple(entry["key"].split()[:2]) for entry in registered}
    candidates = [override] if override else [
        Path.home() / ".runpod/ssh/runpodctl-ssh-key",
        Path.home() / ".ssh/id_ed25519", Path.home() / ".ssh/id_rsa",
    ]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        try:
            public = command(["ssh-keygen", "-y", "-P", "", "-f", str(candidate)])
        except CommandFailure:
            continue
        if tuple(public.decode().split()[:2]) in public_keys:
            return candidate.resolve()
    raise RuntimeError("No matching registered local SSH key. Run runpodctl doctor first, "
                       "or pass --ssh-key PATH (unencrypted key required).")


def local_preflight(config: SweepConfig, ssh_key: Path | None) -> Path:
    for executable in ("runpodctl", "ssh", "ssh-keygen", "git"):
        if not shutil.which(executable):
            raise RuntimeError(f"Missing prerequisite: {executable}")
    help_text = command(["runpodctl", "pod", "create", "--help"]).decode()
    if not all(flag in help_text for flag in ("--wait-timeout", "--stop-after")):
        raise RuntimeError("Update runpodctl to >= 2.11 before launching.")
    seed_job.preflight(REPOSITORY / ".env")
    cli("user")
    gpu = next((g for g in cli("gpu", "list") if g["gpuId"] == GPU), None)
    if not gpu or not gpu.get("available") or not gpu.get("secureCloud"):
        raise RuntimeError("No advertised Secure Cloud L40S capacity. No pods created.")
    # Capacity is not a reservation: failures during creation still trigger cleanup.
    print(f"GPU quote: {config.num_pods} x ${gpu['securePricePerHr']}/hour, plus storage.", flush=True)
    return select_ssh_key(ssh_key)


class Manifest:
    """Serialize concurrent progress, keeping IDs even when setup subsequently fails."""
    def __init__(self, path: Path, config: SweepConfig, plans: tuple[SeedPlan, ...],
                 digest: str, stop_at: str):
        self.path, self.lock = path, threading.Lock()
        self.value: dict[str, Any] = {
            "config": config.model_dump(), "snapshot_sha256": digest,
            "commit": command(["git", "rev-parse", "HEAD"]).decode().strip(),
            "created_at": datetime.now(timezone.utc).isoformat(), "stop_at": stop_at,
            "pods": {str(p.seed): {
                "seed": p.seed, "name": p.name, "run_id": p.run_id,
                "url": f"https://wandb.ai/rundong-liu/JAXUED_TEST/{p.run_id}",
                "command": list(p.command), "state_dir": p.state_dir, "stage": "planned",
            } for p in plans},
        }
        self.save()

    def save(self) -> None:
        seed_job.atomic_write(self.path, json.dumps(self.value, indent=2) + "\n")

    def update(self, seed: int, **fields: Any) -> None:
        with self.lock:
            self.value["pods"][str(seed)].update(fields)
            self.save()


class PodRunner:
    """Runpod/SSH effects for a single seed. No API key is uploaded to pod metadata."""
    def __init__(self, config: SweepConfig, key: Path, manifest: Manifest,
                 snapshot: Path, env_bytes: bytes):
        self.config, self.key, self.manifest = config, key, manifest
        self.snapshot, self.env_bytes = snapshot, env_bytes
        self.connections: dict[int, list[str]] = {}

    def remote(self, plan: SeedPlan, script: str, data: bytes | None = None,
               timeout: int = 120) -> bytes:
        return command([*self.connections[plan.seed], script], data=data, timeout=timeout)

    def prepare(self, plan: SeedPlan) -> None:
        self.manifest.update(plan.seed, stage="creating")
        try:
            pod = cli(*create_args(self.config, plan, self.manifest.value["stop_at"]), timeout=1260)
        except CommandFailure as error:
            if error.payload.get("id"):
                self.manifest.update(plan.seed, pod_id=error.payload["id"], stage="create_failed")
            else:
                self.manifest.update(plan.seed, stage="creation_uncertain")
            raise
        pod_id = pod["id"]
        self.manifest.update(plan.seed, pod_id=pod_id, stage="setting_up")
        connection = cli("ssh", "info", pod_id)
        ip, port = connection["ip"], int(connection["port"])
        ipaddress.ip_address(ip)
        if not 0 < port < 65536:
            raise RuntimeError("Invalid SSH port from Runpod")
        self.connections[plan.seed] = [
            "ssh", "-i", str(self.key), "-p", str(port), "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={self.manifest.path.parent / 'known_hosts'}",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", f"root@{ip}",
        ]
        self.remote(plan, f"mkdir -p {REMOTE} && cd {REMOTE} && tar xzf -",
                    self.snapshot.read_bytes(), timeout=300)
        # stdin avoids placing secrets in argv, shell history, archive or manifest.
        self.remote(plan, f"umask 077; cat > {REMOTE}/.env; chmod 600 {REMOTE}/.env", self.env_bytes)
        setup = f"""set -eu
cd {REMOTE}
exec > setup.log 2>&1
export UV_CACHE_DIR=/workspace/.uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/.uv-python
python3 -m venv /workspace/bootstrap
/workspace/bootstrap/bin/python -m pip install uv=={UV_VERSION}
/workspace/bootstrap/bin/uv sync --frozen --extra cuda --python 3.11
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda .venv/bin/python -c 'import jax; d=jax.devices(); assert len(d)==1 and d[0].platform=="gpu", d; print(d)'
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | tr -d '\\r')" = 'NVIDIA L40S'
.venv/bin/python scripts/seed_job.py preflight --env-file .env
"""
        self.remote(plan, "bash -s", setup.encode(), timeout=1800)
        self.manifest.update(plan.seed, stage="ready")
        print(f"Seed {plan.seed}: pod {pod_id} ready.", flush=True)

    def launch(self, plan: SeedPlan) -> None:
        args = ["env", "-u", "LD_LIBRARY_PATH", "JAX_PLATFORMS=cuda",
                f"WANDB_RUN_ID={plan.run_id}", "WANDB_RESUME=never",
                ".venv/bin/python", "scripts/seed_job.py", "launch", "--env-file", ".env",
                "--seed", str(plan.seed), "--state-dir", plan.state_dir, "--", *plan.command]
        self.manifest.update(plan.seed, stage="launching")
        result = json.loads(self.remote(plan, f"cd {REMOTE} && {shlex.join(args)}"))
        if result.get("ok") is not True:
            raise RuntimeError(f"Seed {plan.seed}: supervisor refused launch")
        self.manifest.update(plan.seed, stage="started", supervisor_pid=result["pid"],
                             launched_at=datetime.now(timezone.utc).isoformat())

    def verify(self, plan: SeedPlan) -> None:
        deadline, previous_size = time.monotonic() + self.config.startup_timeout, 0
        while time.monotonic() < deadline:
            state_command = shlex.join([".venv/bin/python", "scripts/seed_job.py", "status",
                                        "--state-dir", plan.state_dir])
            state = json.loads(self.remote(plan, f"cd {REMOTE} && {state_command}"))
            if not state["running"]:
                raise RuntimeError(f"Seed {plan.seed}: supervisor stopped during startup")
            size = state["log_bytes"]
            if size > previous_size:
                # Query the exact preassigned run ID; mismatched seed is fatal.
                api = seed_job.wandb_api(REPOSITORY / ".env")
                try:
                    run = api.run(f"rundong-liu/JAXUED_TEST/{plan.run_id}")
                except Exception:
                    time.sleep(10)
                    continue
                if run.config.get("seed") != plan.seed:
                    raise RuntimeError(f"Seed {plan.seed}: W&B seed mismatch")
                if run.state != "running":
                    raise RuntimeError(f"Seed {plan.seed}: W&B is {run.state}, not running")
                if previous_size > 0:
                    self.manifest.update(plan.seed, stage="verified", log_bytes=size)
                    print(f"Seed {plan.seed}: running at {run.url}", flush=True)
                    return
                previous_size = size
            time.sleep(10)
        raise RuntimeError(f"Seed {plan.seed}: startup verification timed out")


def parallel_stage(plans: tuple[SeedPlan, ...], action: Callable[[SeedPlan], None]) -> None:
    """Join every worker before cleanup, so late-created pods cannot escape the manifest."""
    with ThreadPoolExecutor(max_workers=len(plans)) as pool:
        futures = [pool.submit(action, plan) for plan in plans]
        for future in futures:
            future.result()


def stop_created_pods(manifest: Manifest) -> None:
    """Recover uncertain creates by exact unique name; stop only this invocation's pods."""
    # Read once after all create calls have joined; never retry a create.
    try:
        pods = cli("pod", "list")
    except Exception:
        pods = []
    for record in list(manifest.value["pods"].values()):
        pod_id = record.get("pod_id")
        matches = [p["id"] for p in pods if p.get("name") == record["name"]]
        ids = [pod_id] if pod_id else matches
        if not ids and record["stage"] in {"creating", "creation_uncertain"}:
            manifest.update(record["seed"], stage="creation_uncertain_check_console")
        for found in ids:
            try:
                current = cli("pod", "get", found)
                if current.get("desiredStatus") in {"STOPPED", "EXITED"}:
                    manifest.update(record["seed"], pod_id=found, stage="already_stopped")
                    continue
                cli("pod", "stop", found)
                manifest.update(record["seed"], pod_id=found, stage="stop_requested")
            except Exception:
                manifest.update(record["seed"], pod_id=found, stage="stop_failed_check_console")


def execute_pipeline(plans: tuple[SeedPlan, ...], runner: PodRunner) -> None:
    """Start each seed as soon as its own pod is ready, without a sweep barrier."""
    failed = threading.Event()

    def run_seed(plan: SeedPlan) -> None:
        try:
            for action in (runner.prepare, runner.launch, runner.verify):
                if failed.is_set():
                    return
                action(plan)
        except BaseException:
            # Let in-flight operations settle, but do not start subsequent stages.
            failed.set()
            raise

    try:
        parallel_stage(plans, run_seed)
    except BaseException:
        stop_created_pods(runner.manifest)
        raise


def wait_for_completion(plans: tuple[SeedPlan, ...], runner: PodRunner, *, poll_interval: int) -> None:
    """Wait until every supervised seed has exited, then return.

    Exit criterion is `seed_job.status` reporting a terminal `exit_code`.
    If a seed transitions to non-running without an exit code, we treat it
    as terminal and fail-fast by stopping the sweep.
    """
    terminal_seeds: set[int] = set()
    while True:
        all_terminal = True
        for plan in plans:
            if plan.seed in terminal_seeds:
                continue
            state_command = shlex.join([".venv/bin/python", "scripts/seed_job.py", "status",
                                        "--state-dir", plan.state_dir])
            try:
                state = json.loads(runner.remote(plan, f"cd {REMOTE} && {state_command}"))
            except CommandFailure:
                all_terminal = False
                continue
            exit_code = state.get("exit_code")
            if exit_code is not None:
                terminal_seeds.add(plan.seed)
                runner.manifest.update(plan.seed, stage="completed",
                                      exit_code=exit_code, running=False)
                print(f"Seed {plan.seed}: completed with exit_code={exit_code}", flush=True)
                continue
            if not state.get("running", False):
                # No exit file but not running: training process already terminal/unavailable.
                terminal_seeds.add(plan.seed)
                runner.manifest.update(plan.seed, stage="terminal_without_exit")
                print(f"Seed {plan.seed}: terminal without exit_code (likely crashed/setup mismatch).", flush=True)
                continue
            all_terminal = False
        if all_terminal and len(terminal_seeds) == len(plans):
            return
        time.sleep(poll_interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print plan; no network or writes")
    parser.add_argument("--num-pods", type=int, default=NUM_PODS)
    parser.add_argument("--ssh-key", type=Path)
    parser.add_argument("--stop-after-hours", type=int, default=STOP_AFTER_HOURS)
    parser.add_argument("--stop-on-completion", action="store_true",
                        help="stop pods automatically once all detached jobs finish")
    parser.add_argument("--completion-poll-interval", type=int, default=30,
                        help="seconds between completion checks when --stop-on-completion is used")
    args = parser.parse_args(argv)
    config = SweepConfig(num_pods=args.num_pods, stop_after_hours=args.stop_after_hours)
    group = "solved-logrel-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    plans = make_plans(config, group)
    stop_at = (datetime.now(timezone.utc) + timedelta(hours=config.stop_after_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.dry_run:
        for plan in plans:
            print(shlex.join(("runpodctl", *create_args(config, plan, stop_at))))
            print(shlex.join(plan.command))
        return 0
    work = REPOSITORY / "runpod-runs" / group
    try:
        key = local_preflight(config, args.ssh_key)
        work.mkdir(parents=True, mode=0o700)
        snapshot = work / "source.tar.gz"
        digest = build_snapshot(REPOSITORY, snapshot)
        manifest = Manifest(work / "manifest.json", config, plans, digest, stop_at)
        print(f"Manifest: {manifest.path}\nSafety STOP deadline: {stop_at}; storage still bills.", flush=True)
        runner = PodRunner(config, key, manifest, snapshot, (REPOSITORY / ".env").read_bytes())
        execute_pipeline(plans, runner)
    except (Exception, KeyboardInterrupt) as error:
        print(f"Launch failed: {type(error).__name__}: {error}\nInspect {work}/manifest.json and Runpod console before retrying.", file=sys.stderr)
        return 1
    print(f"All {config.num_pods} seed jobs verified running. Manifest: {manifest.path}")
    if args.stop_on_completion:
        if args.completion_poll_interval <= 0:
            print("--completion-poll-interval must be positive when --stop-on-completion is used.")
            return 1
        print("Watching all runs; pods will stop automatically after completion.", flush=True)
        wait_for_completion(plans, runner, poll_interval=args.completion_poll_interval)
        stop_created_pods(runner.manifest)
        print("All seeds reached terminal state and stop requests were issued.", flush=True)
        return 0
    print("Launcher may now exit. Pods are NOT automatically stopped at training completion;")
    print("stop them manually when finished, or they stop at the safety deadline above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
