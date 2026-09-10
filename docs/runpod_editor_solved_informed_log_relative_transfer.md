# One-command Runpod seed sweep

From your local JaxUED checkout:

```bash
.venv/bin/python scripts/runpod_editor_solved_informed_log_relative_transfer.py
```

Edit `NUM_PODS = 6` near the top of `scripts/runpod_editor_solved_informed_log_relative_transfer.py` to set the
default count. N pods run seeds `0 .. N-1`, concurrently, with one L40S each.
An optional override is `.venv/bin/python scripts/runpod_editor_solved_informed_log_relative_transfer.py --num-pods 4`.

Preview without network access, credentials or file writes:

```bash
.venv/bin/python scripts/runpod_editor_solved_informed_log_relative_transfer.py --dry-run
```

## Prerequisites

- Local repository environment with dependencies installed (`uv sync --frozen`).
- `runpodctl` 2.11 or newer, authenticated to your Runpod account.
- Registered local SSH key. The launcher checks the Runpod CLI key and then
  `~/.ssh/id_ed25519` and `~/.ssh/id_rsa`; `--ssh-key PATH` overrides discovery.
  Unencrypted keys are required for unattended startup. `runpodctl doctor`
  can configure a key interactively before running the launcher.
- Repository `.env` containing the W&B key for `rundong-liu/JAXUED_TEST`.
  No W&B key is read from the shell or `.netrc`.

## What runs

The launcher snapshots the **current working-tree runtime files**, including
uncommitted scorer/history changes, and copies the same snapshot to all pods.
It excludes `.env`, virtual environments, caches, checkpoints and analysis output
from that archive. The same `.env` is transferred separately over SSH with mode 0600.

All pods use the same pinned Runpod PyTorch image, 30 GB container disk, 100 GB
persistent pod volume, Python 3.11 and `uv sync --frozen --extra cuda`.
Each pod independently creates, sets up, passes its GPU and W&B preflights,
then immediately launches and verifies its seed job. These sequences run
concurrently: a ready pod does not wait for slower pods to finish starting.

Each detached training job uses:

- `editor_solved_informed_log_relative_transfer`, 30,000 updates;
- confidence **0.8**, tau **0.1**, alpha/beta priors **1/1**;
- 128 transfer targets, 16 edits, no exploratory gradient updates, no ACCEL;
- checkpoint interval 10, and the training script's existing evaluation defaults.

The launcher verifies each supervisor, growing log and W&B identity before
reporting successful startup. It then exits; training survives SSH disconnection.
This is startup verification, **not** proof of completed training.

Use `--stop-on-completion` to wait for all jobs to finish and stop pods automatically:

```bash
.venv/bin/python scripts/runpod_editor_solved_informed_log_relative_transfer.py --stop-on-completion
```

## Costs, failures and evidence

Execution creates paid resources. The live GPU quote is printed before creation;
advertised stock does not guarantee N available machines or reserve capacity.

`STOP_AFTER_HOURS = 168` is a configurable safety deadline, **not a runtime
estimate**. Use `--stop-after-hours HOURS` to override it. The deadline stops the
pod even if training has not finished. It never deletes the pod. Storage keeps
billing after a stop. `--stop-on-completion` is independent: if enabled, pods are
stopped automatically after all jobs report terminal state, and they still stop at
the safety deadline when time runs out.

Local evidence: `runpod-runs/<unique-sweep>/manifest.json` and `source.tar.gz`.
The manifest records snapshot SHA-256, base commit, per-seed command, pod name/ID,
W&B run ID/URL, launch time and state. Runtime files are under `/workspace/jaxued`;
remote setup output is `setup.log`, and per-seed output is
`runpod-runs/<unique-sweep>/seed-N/train.log`.

On setup/startup failure, workers skip subsequent stages after observing the failure.
On setup/startup failure or Ctrl-C, the launcher joins outstanding operations,
then requests stops for pods created by this invocation; it never deletes pods.
Ambiguous creates are looked up by their unique names and **not retried**.
Check the manifest and Runpod console after a failure: API/network failures can
prevent cleanup, and `stop_requested` is not confirmation of stopped state.
Do not rerun blindly; each invocation creates a fresh sweep, not a resume.

Offline tests:

```bash
.venv/bin/python -m pytest tests/test_runpod_editor_solved_informed_log_relative_transfer.py -q
```
