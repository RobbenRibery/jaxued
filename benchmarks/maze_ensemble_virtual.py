"""Benchmark multi-phase Maze ensemble scoring on one JAX device."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

from examples.maze_ensemble_plr import (  # noqa: E402
    ActorCritic,
    PPOParameters,
    VirtualLearningParameters,
    collect_ensemble_trajectories,
    score_multi_phase_ensemble_rollout,
)
from jaxued.environments import Maze  # noqa: E402
from jaxued.environments.maze import make_level_generator  # noqa: E402
from jaxued.wrappers import AutoReplayWrapper  # noqa: E402


@dataclass(frozen=True)
class BenchmarkResult:
    """Serializable steady-state benchmark measurements."""

    status: str
    device: str
    batch_size: int
    num_agents: int
    num_levels: int
    num_steps: int
    num_phases: int
    virtual_epochs: int
    iterations: int
    compile_seconds: float
    median_seconds: float
    p95_seconds: float
    score_updates_per_second: float
    score_updates_per_dollar: float | None
    peak_gpu_memory_mib: int | None
    compiled_argument_bytes: int | None
    compiled_output_bytes: int | None
    compiled_temporary_bytes: int | None
    score_checksum: float


class GpuMemorySampler:
    """Poll ``nvidia-smi`` while a benchmark runs."""

    def __init__(self, interval_seconds: float = 0.05):
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mib: int | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                values = [
                    int(value.strip())
                    for value in completed.stdout.splitlines()
                    if value.strip()
                ]
                if values:
                    current = max(values)
                    self.peak_mib = (
                        current
                        if self.peak_mib is None
                        else max(self.peak_mib, current)
                    )
            except (FileNotFoundError, subprocess.SubprocessError, ValueError):
                self._stop.set()
                return
            self._stop.wait(self._interval_seconds)

    def __enter__(self) -> GpuMemorySampler:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


def _memory_analysis(compiled: Any, name: str) -> int | None:
    if not hasattr(compiled, "memory_analysis"):
        return None
    analysis = compiled.memory_analysis()
    value = getattr(analysis, name, None)
    return None if value is None else int(value)


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    """Build a realistic fixture and time only the virtual scoring kernel."""
    if not 1 <= args.batch_size <= args.num_levels:
        raise ValueError("batch_size must be between 1 and num_levels")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")

    base_env = Maze(
        max_height=13,
        max_width=13,
        agent_view_size=5,
        normalize_obs=True,
    )
    env = AutoReplayWrapper(base_env)
    env_params = env.default_params
    level_generator = make_level_generator(13, 13, 25)
    levels = jax.vmap(level_generator)(
        jax.random.split(jax.random.PRNGKey(args.seed), args.num_levels)
    )
    init_obs, init_env_state = jax.vmap(
        env.reset_to_level,
        in_axes=(0, 0, None),
    )(
        jax.random.split(jax.random.PRNGKey(args.seed + 1), args.num_levels),
        levels,
        env_params,
    )

    network = ActorCritic(env.action_space(env_params).n)
    init_inputs = (
        jax.tree_util.tree_map(lambda value: value[None, ...], init_obs),
        jnp.zeros((1, args.num_levels), dtype=bool),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(1e-4, eps=1e-5),
    )

    def _create_agent(rng):
        params = network.init(
            rng,
            init_inputs,
            ActorCritic.initialize_carry((args.num_levels,)),
        )
        return TrainState.create(apply_fn=network.apply, params=params, tx=optimizer)

    agents = jax.vmap(_create_agent)(
        jax.random.split(jax.random.PRNGKey(args.seed + 2), args.num_agents)
    )
    phase_zero = collect_ensemble_trajectories(
        jax.random.split(jax.random.PRNGKey(args.seed + 3), args.num_agents),
        env,
        env_params,
        agents,
        init_obs,
        init_env_state,
        args.num_levels,
        args.num_steps,
        base_env.max_width,
    )
    ppo = PPOParameters(
        gamma=0.995,
        gae_lambda=0.98,
        num_minibatches=1,
        num_epochs=args.virtual_epochs,
        clip_eps=0.2,
        entropy_coeff=1e-3,
        critic_coeff=0.5,
    )
    virtual = VirtualLearningParameters(
        ppo=ppo,
        num_phases=args.num_phases,
        level_batch_size=args.batch_size,
    )
    num_states = base_env.max_height * base_env.max_width * 4

    def _score(rng, member_states, trajectory, last_values, observations, env_state):
        return score_multi_phase_ensemble_rollout(
            rng,
            env,
            env_params,
            member_states,
            trajectory,
            last_values,
            observations,
            env_state,
            args.num_steps,
            base_env.max_width,
            num_states,
            virtual,
        )

    scorer = jax.jit(_score)
    sample_args = (
        jax.random.PRNGKey(args.seed + 4),
        agents,
        phase_zero.trajectory,
        phase_zero.last_values,
        init_obs,
        init_env_state,
    )

    with GpuMemorySampler() as memory_sampler:
        compile_start = time.perf_counter()
        compiled = scorer.lower(*sample_args).compile()
        compile_seconds = time.perf_counter() - compile_start

        warmup = compiled(*sample_args)
        jax.block_until_ready(warmup.disagreement.scores)

        timings = []
        result = warmup
        for iteration in range(args.iterations):
            call_args = (
                jax.random.fold_in(sample_args[0], iteration + 1),
                *sample_args[1:],
            )
            start = time.perf_counter()
            result = compiled(*call_args)
            jax.block_until_ready(result.disagreement.scores)
            timings.append(time.perf_counter() - start)

    median_seconds = float(np.median(timings))
    updates_per_second = 1.0 / median_seconds
    updates_per_dollar = None
    if args.hourly_gpu_price is not None:
        updates_per_dollar = updates_per_second * 3600.0 / args.hourly_gpu_price

    return BenchmarkResult(
        status="ok",
        device=str(jax.devices()[0].device_kind),
        batch_size=args.batch_size,
        num_agents=args.num_agents,
        num_levels=args.num_levels,
        num_steps=args.num_steps,
        num_phases=args.num_phases,
        virtual_epochs=args.virtual_epochs,
        iterations=args.iterations,
        compile_seconds=compile_seconds,
        median_seconds=median_seconds,
        p95_seconds=float(np.percentile(timings, 95)),
        score_updates_per_second=updates_per_second,
        score_updates_per_dollar=updates_per_dollar,
        peak_gpu_memory_mib=memory_sampler.peak_mib,
        compiled_argument_bytes=_memory_analysis(compiled, "argument_size_in_bytes"),
        compiled_output_bytes=_memory_analysis(compiled, "output_size_in_bytes"),
        compiled_temporary_bytes=_memory_analysis(compiled, "temp_size_in_bytes"),
        score_checksum=float(jnp.sum(result.disagreement.scores)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_agents", type=int, default=8)
    parser.add_argument("--num_levels", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--num_phases", type=int, default=3)
    parser.add_argument("--virtual_epochs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hourly_gpu_price", type=float, default=None)
    return parser


if __name__ == "__main__":
    result = run_benchmark(build_parser().parse_args())
    print(json.dumps(asdict(result), sort_keys=True))
