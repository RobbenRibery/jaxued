"""Offline launch safety tests: never provision real resources from tests."""

import json
from pathlib import Path
import subprocess
import tarfile
import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from scripts import runpod_editor_solved_informed_log_relative_transfer as launcher
from scripts import seed_job

REAL_CLI = launcher.cli


@pytest.fixture(autouse=True)
def no_external_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected external operation in offline test")
    monkeypatch.setattr(launcher, "command", forbidden)
    monkeypatch.setattr(launcher, "cli", forbidden)
    monkeypatch.setattr(seed_job, "wandb_api", forbidden)


@pytest.mark.parametrize("count", [1, 2, 6, 9])
def test_count_controls_unique_seeds_and_one_gpu_per_pod(count):
    config = launcher.SweepConfig(num_pods=count)
    plans = launcher.make_plans(config, "test-sweep")
    assert [p.seed for p in plans] == list(range(count))
    assert len({p.name for p in plans}) == len({p.run_id for p in plans}) == count
    for plan in plans:
        args = launcher.create_args(config, plan, "2030-01-01T00:00:00Z")
        assert seed_job.option_value(args, "--gpu-count") == "1"
        assert seed_job.option_value(args, "--gpu-id") == "NVIDIA L40S"
        assert "--stop-after" in args and "--terminate-after" not in args
        seed_job.validate_command(plan.command, plan.seed)
        assert seed_job.option_value(plan.command, "--transfer_solved_confidence") == "0.8"
        assert seed_job.option_value(plan.command, "--transfer_log_relative_tau") == "0.1"
        assert seed_job.option_value(plan.command, "--transfer_solved_prior_alpha") == "1.0"
        assert seed_job.option_value(plan.command, "--score_function") == "editor_solved_informed_log_relative_transfer"
        assert "--no-exploratory_grad_updates" in plan.command
        assert "--no-use_accel" in plan.command


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "6"])
def test_invalid_count_rejected(count):
    with pytest.raises(ValidationError):
        launcher.SweepConfig(num_pods=count)


def test_editable_constant_is_cli_default(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "NUM_PODS", 3)
    assert launcher.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert output.count("runpodctl pod create") == 3


def test_override_and_dry_run_need_no_credentials_or_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launcher, "REPOSITORY", tmp_path)
    assert launcher.main(["--dry-run", "--num-pods", "2"]) == 0
    assert capsys.readouterr().out.count("runpodctl pod create") == 2
    assert list(tmp_path.iterdir()) == []


def test_snapshot_includes_dirty_runtime_not_credentials(tmp_path, monkeypatch):
    names = ["src/pkg/new.py", "examples/maze_plr.py", "scripts/seed_job.py",
             "pyproject.toml", "uv.lock", ".env", "src/pkg/.env", "src/pkg/test.pem",
             "src/pkg/__pycache__/thing.py", "README.md"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SECRET" if ".env" in name else "dirty working tree")
    monkeypatch.setattr(launcher, "command", lambda *a, **k: b"\0".join(n.encode() for n in names))
    archive = tmp_path / "source.tar.gz"
    digest = launcher.build_snapshot(tmp_path, archive)
    assert len(digest) == 64
    with tarfile.open(archive) as bundle:
        assert bundle.getnames() == sorted([
            "src/pkg/new.py", "examples/maze_plr.py", "scripts/seed_job.py",
            "pyproject.toml", "uv.lock", "README.md",
        ])
        assert bundle.extractfile("src/pkg/new.py").read() == b"dirty working tree"


def test_snapshot_rejects_symlink(tmp_path, monkeypatch):
    (tmp_path / "outside").write_text("secret")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/link.py").symlink_to(tmp_path / "outside")
    monkeypatch.setattr(launcher, "command", lambda *a, **k: b"scripts/link.py\0")
    with pytest.raises(RuntimeError, match="symlink"):
        launcher.build_snapshot(tmp_path, tmp_path / "snapshot.tgz")


def test_pipeline_launches_and_verifies_ready_pod_while_others_prepare():
    plans = launcher.make_plans(launcher.SweepConfig(num_pods=3), "test")
    barrier = threading.Barrier(3, timeout=5)
    first_verified = threading.Event()
    events, lock = [], threading.Lock()

    def prepare(plan):
        barrier.wait()  # Would fail if provisioning became sequential.
        if plan.seed != 0:
            assert first_verified.wait(timeout=5)
        with lock:
            events.append(("ready", plan.seed))

    def launch(plan):
        with lock:
            assert ("ready", plan.seed) in events
            if plan.seed == 0:
                assert events == [("ready", 0)]
            events.append(("started", plan.seed))

    def verify(plan):
        with lock:
            assert ("started", plan.seed) in events
            events.append(("verified", plan.seed))
        if plan.seed == 0:
            first_verified.set()

    runner = SimpleNamespace(prepare=prepare, launch=launch, verify=verify)
    launcher.execute_pipeline(plans, runner)
    assert len(events) == 9
    assert events[:3] == [("ready", 0), ("started", 0), ("verified", 0)]


@pytest.mark.parametrize("failure_stage", ["prepare", "launch", "verify"])
def test_failure_joins_workers_then_cleans_up(failure_stage, monkeypatch):
    plans = launcher.make_plans(launcher.SweepConfig(num_pods=3), "test")
    events, completed = [], set()
    barrier = threading.Barrier(3, timeout=5)

    def stage(name):
        def run(plan):
            events.append((name, plan.seed))
            if name == failure_stage:
                barrier.wait()
                completed.add(plan.seed)
                if plan.seed == 0:
                    raise RuntimeError("deliberate failure")
        return run

    def cleanup(manifest):
        assert completed == {0, 1, 2}
        events.append("cleanup")

    monkeypatch.setattr(launcher, "stop_created_pods", cleanup)
    runner = SimpleNamespace(**{s: stage(s) for s in ("prepare", "launch", "verify")}, manifest=None)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        launcher.execute_pipeline(plans, runner)
    assert events[-1] == "cleanup"
    stages = ["prepare", "launch", "verify"]
    for later in stages[stages.index(failure_stage) + 1:]:
        assert (later, 0) not in events


def fake_manifest(records=None):
    records = records or {"0": {"seed": 0, "name": "unique-seed-0", "stage": "planned"}}
    return SimpleNamespace(
        value={"pods": records, "stop_at": "2030-01-01T00:00:00Z"},
        update=lambda seed, **fields: records[str(seed)].update(fields),
        path=Path("/tmp/test-runpod/manifest.json"),
    )


def test_completion_option_still_waits_for_all_jobs_then_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "REPOSITORY", tmp_path)
    (tmp_path / ".env").write_text("test-only")
    monkeypatch.setattr(launcher, "local_preflight", lambda *a: Path("key"))
    monkeypatch.setattr(launcher, "build_snapshot", lambda *a: "digest")
    manifest = fake_manifest({str(seed): {
        "seed": seed, "name": f"seed-{seed}", "stage": "planned",
    } for seed in range(2)})
    monkeypatch.setattr(launcher, "Manifest", lambda *a: manifest)
    events = []
    states = {0: iter([{"running": False, "exit_code": 0}]),
              1: iter([{"running": True}, {"running": False, "exit_code": 0}])}

    def remote(plan, script):
        assert ("verified", plan.seed) in events
        return json.dumps(next(states[plan.seed])).encode()

    runner = SimpleNamespace(
        manifest=manifest, remote=remote,
        prepare=lambda p: events.append(("ready", p.seed)),
        launch=lambda p: events.append(("started", p.seed)),
        verify=lambda p: events.append(("verified", p.seed)),
    )
    monkeypatch.setattr(launcher, "PodRunner", lambda *a: runner)
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: events.append(("poll", seconds)))

    def stop(actual_manifest):
        assert actual_manifest is manifest
        assert all(record["exit_code"] == 0 for record in manifest.value["pods"].values())
        events.append(("stop", None))

    monkeypatch.setattr(launcher, "stop_created_pods", stop)
    assert launcher.main(["--num-pods", "2", "--stop-on-completion"]) == 0
    assert events[-2:] == [("poll", 30), ("stop", None)]


def test_wait_timeout_records_id_without_retry(monkeypatch):
    manifest, calls = fake_manifest(), []
    runner = launcher.PodRunner(launcher.SweepConfig(), Path("key"), manifest, Path("snapshot"), b"")
    def failed_create(*args, **kwargs):
        calls.append(args)
        raise launcher.CommandFailure("runpodctl", {"id": "created-pod", "code": "timeout"})
    monkeypatch.setattr(launcher, "cli", failed_create)
    plan = launcher.make_plans(launcher.SweepConfig(num_pods=1), "unique")[0]
    with pytest.raises(launcher.CommandFailure):
        runner.prepare(plan)
    assert len(calls) == 1
    assert manifest.value["pods"]["0"]["pod_id"] == "created-pod"


def test_cleanup_recovers_only_exact_names_never_deletes(monkeypatch):
    manifest, stopped = fake_manifest(), []
    manifest.update(0, stage="creation_uncertain")
    def fake_cli(*args, **kwargs):
        if args == ("pod", "list"):
            return [{"id": "owned", "name": "unique-seed-0"},
                    {"id": "other", "name": "other-seed-0"}]
        if args == ("pod", "get", "owned"):
            return {"desiredStatus": "RUNNING"}
        assert args[:2] == ("pod", "stop")
        stopped.append(args[2])
    monkeypatch.setattr(launcher, "cli", fake_cli)
    launcher.stop_created_pods(manifest)
    assert stopped == ["owned"]
    assert manifest.value["pods"]["0"]["pod_id"] == "owned"


def test_cleanup_does_not_stop_an_already_stopped_pod(monkeypatch):
    manifest = fake_manifest()
    manifest.update(0, pod_id="owned")
    def fake_cli(*args, **kwargs):
        if args == ("pod", "list"):
            return []
        assert args == ("pod", "get", "owned")
        return {"desiredStatus": "EXITED"}
    monkeypatch.setattr(launcher, "cli", fake_cli)
    launcher.stop_created_pods(manifest)
    assert manifest.value["pods"]["0"]["stage"] == "already_stopped"


def test_unrecoverable_create_is_reported(monkeypatch):
    manifest = fake_manifest()
    manifest.update(0, stage="creating")
    monkeypatch.setattr(launcher, "cli", lambda *a, **k: [])
    launcher.stop_created_pods(manifest)
    assert manifest.value["pods"]["0"]["stage"] == "creation_uncertain_check_console"


def test_cli_zero_exit_error_is_not_success(monkeypatch):
    monkeypatch.setattr(launcher, "command", lambda *a, **k: b'{"error":"pod not ready"}')
    with pytest.raises(launcher.CommandFailure):
        REAL_CLI("ssh", "info", "pod")


def test_wandb_preflight_failure_prevents_all_creates(monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: name)
    monkeypatch.setattr(launcher, "command", lambda *a, **k: b"--wait-timeout --stop-after")
    monkeypatch.setattr(seed_job, "preflight", lambda *a: (_ for _ in ()).throw(RuntimeError("wrong account")))
    with pytest.raises(RuntimeError, match="wrong account"):
        launcher.local_preflight(launcher.SweepConfig(), None)


def test_detached_launch_command_has_seed_cuda_and_preassigned_wandb_id(monkeypatch):
    plan = launcher.make_plans(launcher.SweepConfig(num_pods=1), "unique")[0]
    runner = launcher.PodRunner(launcher.SweepConfig(), Path("key"), fake_manifest(), Path("source"), b"SECRET")
    commands = []
    def remote(plan, script, *args, **kwargs):
        commands.append(script)
        return b'{"ok":true,"pid":123}'
    monkeypatch.setattr(runner, "remote", remote)
    runner.launch(plan)
    assert "WANDB_RUN_ID=unique-0" in commands[0]
    assert "env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda" in commands[0]
    assert "scripts/seed_job.py launch" in commands[0]
    assert "SECRET" not in commands[0]


def test_helper_dotenv_is_literal_and_does_not_fall_back(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text('export WANDB_API_KEY="$(do-not-execute)"\n')
    monkeypatch.setenv("WANDB_API_KEY", "wrong-key")
    env = seed_job.target_environment(dotenv)
    assert env["WANDB_API_KEY"] == "$(do-not-execute)"
    assert env["WANDB_ENTITY"] == "rundong-liu"
    assert env["WANDB_PROJECT"] == "JAXUED_TEST"
    with pytest.raises(seed_job.GuardError):
        seed_job.target_environment(tmp_path / "missing.env")


@pytest.mark.parametrize("args", [[], ["--seed", "0", "--project", "wrong"],
                                 ["--seed", "1", "--project", "JAXUED_TEST"]])
def test_helper_rejects_wrong_target(args):
    with pytest.raises(seed_job.GuardError):
        seed_job.validate_command(args, 0)


def test_helper_launch_detaches_and_uses_only_safe_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(seed_job, "preflight", lambda p: {"ok": True})
    launches = []
    def popen(args, **kwargs):
        launches.append((args, kwargs))
        return SimpleNamespace(pid=123456)
    monkeypatch.setattr(subprocess, "Popen", popen)
    command = ("python", "train.py", "--seed", "0", "--project", "JAXUED_TEST")
    result = seed_job.launch(tmp_path / ".env", 0, tmp_path / "state", command)
    assert result["pid"] == 123456
    assert launches[0][1]["start_new_session"] is True
    assert launches[0][1]["stdin"] == subprocess.DEVNULL
    manifest = json.loads((tmp_path / "state/job.json").read_text())
    assert manifest["command"] == list(command)
    assert "WANDB_API_KEY" not in json.dumps(manifest)


def test_prepare_pins_cuda_setup_and_secure_secret_transfer(tmp_path, monkeypatch):
    manifest = fake_manifest()
    source = tmp_path / "snapshot"
    source.write_bytes(b"test-archive")
    runner = launcher.PodRunner(launcher.SweepConfig(), Path("key"), manifest, source, b"SECRET")
    plan = launcher.make_plans(launcher.SweepConfig(num_pods=1), "unique")[0]
    calls, transfers = [], []
    def fake_cli(*args, **kwargs):
        calls.append(args)
        if args[:2] == ("pod", "create"):
            return {"id": "pod0"}
        assert args == ("ssh", "info", "pod0")
        return {"ip": "192.0.2.1", "port": 12345}
    def remote(plan, script, data=None, timeout=120):
        transfers.append((script, data))
        return b""
    monkeypatch.setattr(launcher, "cli", fake_cli)
    monkeypatch.setattr(runner, "remote", remote)
    runner.prepare(plan)
    assert len(calls) == 2
    assert manifest.value["pods"]["0"]["stage"] == "ready"
    assert transfers[0][1] == b"test-archive"
    assert "umask 077" in transfers[1][0] and "chmod 600" in transfers[1][0]
    assert transfers[1][1] == b"SECRET"
    assert all("SECRET" not in script for script, _ in transfers)
    setup = transfers[2][1].decode()
    assert "uv sync --frozen --extra cuda --python 3.11" in setup
    assert "JAX_PLATFORMS=cuda" in setup
    assert "scripts/seed_job.py preflight --env-file .env" in setup
    assert "StrictHostKeyChecking=accept-new" in runner.connections[0]
    assert "SECRET" not in json.dumps(manifest.value)
    # Parse the generated shell without executing installs or starting any pods.
    assert subprocess.run(["bash", "-n"], input=setup, text=True, capture_output=True).returncode == 0


@pytest.mark.parametrize("state", ["crashed", "finished"])
def test_startup_rejects_nonrunning_wandb(state, monkeypatch):
    plan = launcher.make_plans(launcher.SweepConfig(num_pods=1), "unique")[0]
    runner = launcher.PodRunner(launcher.SweepConfig(), Path("key"), fake_manifest(), Path("source"), b"")
    monkeypatch.setattr(runner, "remote", lambda *a, **k: b'{"running":true,"log_bytes":10}')
    monkeypatch.setattr(seed_job, "wandb_api", lambda p: SimpleNamespace(
        run=lambda path: SimpleNamespace(config={"seed": 0}, state=state)))
    with pytest.raises(RuntimeError, match="not running"):
        runner.verify(plan)


def test_startup_rejects_wrong_seed(monkeypatch):
    plan = launcher.make_plans(launcher.SweepConfig(num_pods=1), "unique")[0]
    runner = launcher.PodRunner(launcher.SweepConfig(), Path("key"), fake_manifest(), Path("source"), b"")
    monkeypatch.setattr(runner, "remote", lambda *a, **k: b'{"running":true,"log_bytes":10}')
    monkeypatch.setattr(seed_job, "wandb_api", lambda p: SimpleNamespace(
        run=lambda path: SimpleNamespace(config={"seed": 99}, state="running")))
    with pytest.raises(RuntimeError, match="seed mismatch"):
        runner.verify(plan)


def test_startup_requires_growing_log_and_expected_wandb_path(monkeypatch):
    plan = launcher.make_plans(launcher.SweepConfig(num_pods=1), "unique")[0]
    manifest = fake_manifest()
    runner = launcher.PodRunner(launcher.SweepConfig(), Path("key"), manifest, Path("source"), b"")
    sizes = iter([0, 10, 10, 20])
    monkeypatch.setattr(runner, "remote", lambda *a, **k: json.dumps({"running": True, "log_bytes": next(sizes)}).encode())
    monkeypatch.setattr(launcher.time, "sleep", lambda s: None)
    paths = []
    def run(path):
        paths.append(path)
        return SimpleNamespace(config={"seed": 0}, state="running", url="expected-url")
    monkeypatch.setattr(seed_job, "wandb_api", lambda p: SimpleNamespace(run=run))
    runner.verify(plan)
    assert paths == ["rundong-liu/JAXUED_TEST/unique-0"] * 2
    assert manifest.value["pods"]["0"]["stage"] == "verified"
    assert manifest.value["pods"]["0"]["log_bytes"] == 20


def test_supervisor_status_handles_log_not_created_yet(tmp_path):
    status = seed_job.status(tmp_path)
    assert status["log_bytes"] == 0
    assert status["running"] is False


def test_ssh_key_selection_matches_registered_public_key(tmp_path, monkeypatch):
    key = tmp_path / "ssh-key"
    key.write_text("test-fixture-not-a-real-private-key")
    monkeypatch.setattr(launcher, "cli", lambda *a: {"keys": [{"key": "ssh-ed25519 public registered-comment"}]})
    monkeypatch.setattr(launcher, "command", lambda *a, **k: b"ssh-ed25519 public different-comment")
    assert launcher.select_ssh_key(key) == key.resolve()


def test_ssh_key_selection_rejects_unregistered_key(tmp_path, monkeypatch):
    key = tmp_path / "ssh-key"
    key.write_text("test-fixture-not-a-real-private-key")
    monkeypatch.setattr(launcher, "cli", lambda *a: {"keys": [{"key": "ssh-ed25519 other"}]})
    monkeypatch.setattr(launcher, "command", lambda *a, **k: b"ssh-ed25519 public")
    with pytest.raises(RuntimeError, match="No matching"):
        launcher.select_ssh_key(key)
