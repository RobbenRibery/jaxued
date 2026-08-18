"""Benchmark virtual level batch sizes on Modal L40S and RTX PRO 6000 GPUs.

Run with::

    modal run modal-run/maze/benchmark_ensemble_epistemic.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import modal

from _common import REMOTE_REPOSITORY, image


app = modal.App("jaxued-maze-ensemble-epistemic-benchmark", image=image)

# Modal's public per-second prices converted to hourly prices. The local
# entrypoint exposes both values because cloud pricing can change.
DEFAULT_L40S_HOURLY_PRICE = 0.000542 * 3600
DEFAULT_RTX_PRO_6000_HOURLY_PRICE = 0.000842 * 3600


def _run_batch_size(batch_size: int, hourly_gpu_price: float) -> dict[str, Any]:
    command = (
        sys.executable,
        "benchmarks/maze_ensemble_virtual.py",
        "--batch_size",
        str(batch_size),
        "--hourly_gpu_price",
        str(hourly_gpu_price),
    )
    completed = subprocess.run(
        command,
        cwd=REMOTE_REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "batch_size": batch_size,
            "returncode": completed.returncode,
            "stderr": completed.stderr[-4000:],
        }
    return json.loads(completed.stdout.splitlines()[-1])


@app.function(gpu="L40S", timeout=2 * 60 * 60)
def benchmark_l40s(request: tuple[int, float]) -> dict[str, Any]:
    """Benchmark one batch size on a fresh L40S subprocess."""
    batch_size, hourly_gpu_price = request
    return _run_batch_size(batch_size, hourly_gpu_price)


@app.function(gpu="RTX-PRO-6000", timeout=2 * 60 * 60)
def benchmark_rtx_pro_6000(request: tuple[int, float]) -> dict[str, Any]:
    """Benchmark one batch size on a fresh RTX PRO 6000 subprocess."""
    batch_size, hourly_gpu_price = request
    return _run_batch_size(batch_size, hourly_gpu_price)


def _normalize_results(
    batch_sizes: list[int],
    results: list[dict[str, Any] | BaseException],
) -> list[dict[str, Any]]:
    """Convert remote failures, including OOM terminations, into records."""
    normalized = []
    for batch_size, result in zip(batch_sizes, results):
        if isinstance(result, BaseException):
            normalized.append(
                {
                    "status": "remote_failed",
                    "batch_size": batch_size,
                    "error": f"{type(result).__name__}: {result}",
                }
            )
        else:
            normalized.append(result)
    return normalized


def _fastest_success(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [result for result in results if result.get("status") == "ok"]
    if not successful:
        return None
    return min(successful, key=lambda result: result["median_seconds"])


@app.local_entrypoint()
def main(
    gpu: str = "both",
    l40s_hourly_price: float = DEFAULT_L40S_HOURLY_PRICE,
    rtx_pro_6000_hourly_price: float = DEFAULT_RTX_PRO_6000_HOURLY_PRICE,
) -> None:
    """Run the full batch-size sweep and print machine-readable results."""
    batch_sizes = [1, 2, 4, 8, 16, 32]
    output: dict[str, Any] = {}
    if gpu in ("both", "l40s"):
        l40s_results = _normalize_results(
            batch_sizes,
            list(
                benchmark_l40s.map(
                    [(size, l40s_hourly_price) for size in batch_sizes],
                    return_exceptions=True,
                )
            ),
        )
        output["L40S"] = {
            "results": l40s_results,
            "fastest": _fastest_success(l40s_results),
        }
    if gpu in ("both", "rtx-pro-6000"):
        rtx_results = _normalize_results(
            batch_sizes,
            list(
                benchmark_rtx_pro_6000.map(
                    [(size, rtx_pro_6000_hourly_price) for size in batch_sizes],
                    return_exceptions=True,
                )
            ),
        )
        output["RTX-PRO-6000"] = {
            "results": rtx_results,
            "fastest": _fastest_success(rtx_results),
        }
    if not output:
        raise ValueError("gpu must be one of: both, l40s, rtx-pro-6000")
    print(json.dumps(output, indent=2, sort_keys=True))
