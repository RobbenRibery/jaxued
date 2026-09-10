#!/usr/bin/env python3
"""Analyze PLR-prioritized Maze batches logged to Weights & Biases.

The script deliberately separates three objects that are easy to conflate:

1. the full PLR bank (not available from W&B image logs),
2. the rank-weighted replay batch sampled from that bank, and
3. the fixed evaluation suite used for solve-rate reporting.

It uses W&B's ``images/replay_levels`` payload to measure exact replay-batch
concentration and, at a configurable cadence, decodes the rendered mazes to
measure wall-map diversity and structural difficulty. No credential is written
to disk or included in an output artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import netrc
import os
import tempfile
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import wandb
from PIL import Image
from scipy.stats import spearmanr, t, ttest_rel


PRIMARY_GROUPS = {
    "maze_robust_plr_three_seeds": "MaxMC",
    "maze_mean_absolute_advantage_three_seeds": "Mean absolute advantage",
    "maze_mean_positive_delight_three_seeds": "Mean positive delight",
    "maze_editor_transfer_three_seeds": "Editor transfer",
    "maze_editor_log_relative_transfer_three_seeds": "Log-relative transfer",
}

ROBUSTNESS_GROUPS = {
    "maze_editor_transfer_robust_three_seeds": "Editor transfer (earlier replication)",
}

EXPECTED_PROTOCOL = {
    "num_updates": 30_000,
    "eval_freq": 200,
    "num_train_envs": 32,
    "num_steps": 256,
    "num_minibatches": 1,
    "epoch_ppo": 5,
    "lr": 1e-4,
    "gamma": 0.995,
    "gae_lambda": 0.98,
    "replay_prob": 0.8,
    "level_buffer_capacity": 4_000,
    "temperature": 0.3,
    "staleness_coeff": 0.3,
    "prioritization": "rank",
    "num_edits": 5,
    "use_accel": False,
    "exploratory_grad_updates": False,
    "buffer_duplicate_check": True,
}

HISTORY_FIELDS = {
    "solve_rate": "solve_rate/mean",
    "return_mean": "return/mean",
    "eval_episode_length": "eval_ep_lengths/mean",
    "bank_size": "level_sampler/size",
    "bank_mean_score": "level_sampler/mean_score",
    "bank_weighted_score": "level_sampler/weighted_score",
    "bank_max_score": "level_sampler/max_score",
    "bank_score_q10": "level_sampler/score_q10",
    "bank_score_q50": "level_sampler/score_q50",
    "bank_score_q90": "level_sampler/score_q90",
    "bank_top_score_gap": "level_sampler/top_score_gap",
    "mean_wall_count_logged": "mean_num_blocks",
    "transfer_pre_return": "transfer/pre_return_mean",
    "transfer_post_return": "transfer/post_return_mean",
    "transfer_gain_mean": "transfer/gain_mean",
    "transfer_log_relative_gain_mean": "transfer/log_relative_gain_mean",
    "transfer_standard_error": "transfer/standard_error_mean",
    "agent_entropy": "agent/entropy",
    "agent_policy_loss": "agent/policy_loss",
    "agent_value_loss": "agent/value_loss",
    "agent_total_loss": "agent/loss",
}

STRUCTURAL_METRICS = (
    "wall_count",
    "wall_fraction",
    "solvable",
    "shortest_path",
    "manhattan_distance",
    "path_stretch",
    "reachable_fraction",
    "dead_end_fraction",
    "junction_fraction",
)

DIVERSITY_METRICS = (
    "exact_unique_ratio",
    "duplicate_fraction",
    "dominant_share",
    "shannon_effective_size",
    "simpson_effective_size",
    "topology_unique_ratio",
    "mean_pairwise_wall_distance_d4",
    "mean_nearest_wall_distance_d4_unique",
)


@dataclass(frozen=True)
class SourceSpec:
    credential: str
    entity: str
    project: str


@dataclass(frozen=True)
class RunSpec:
    credential: str
    entity: str
    project: str
    run_id: str
    run_name: str
    group: str
    algorithm: str
    seed: int | None
    state: str
    role: str
    created_at: str
    url: str
    final_logged_update: int | None
    config_matches_protocol: bool
    protocol_mismatches: str


@dataclass(frozen=True)
class DecodedMaze:
    wall_map: np.ndarray
    agent: tuple[int, int] | None
    goal: tuple[int, int] | None
    metrics: Mapping[str, float]


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"").strip("'")
    return values


def credentials(repo: Path, include_secondary: bool) -> dict[str, str]:
    env_values = parse_env_file(repo / ".env")
    env_key = os.environ.get("WANDB_API_KEY") or env_values.get("WANDB_API_KEY")
    if not env_key:
        raise RuntimeError("WANDB_API_KEY is missing from the environment and .env")
    result = {"env": env_key}
    if include_secondary:
        auth = netrc.netrc().authenticators("api.wandb.ai")
        if auth is None or not auth[2]:
            raise RuntimeError("Secondary W&B credential is unavailable in .netrc")
        result["secondary"] = auth[2]
    return result


def scalar(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return math.nan
    return converted if math.isfinite(converted) else math.nan


def stable_hash_from_filename(filename: str) -> str:
    return filename.rsplit("_", 1)[-1].split(".", 1)[0]


def stage_for_update(update: int) -> str:
    if update <= 10_000:
        return "Early (0-10k)"
    if update <= 20_000:
        return "Middle (10-20k)"
    return "Late (20-30k)"


def protocol_mismatches(config: Mapping[str, Any]) -> list[str]:
    mismatches = []
    for key, expected in EXPECTED_PROTOCOL.items():
        actual = config.get(key)
        if isinstance(expected, float) and isinstance(actual, (float, int)):
            matches = math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12)
        else:
            matches = actual == expected
        if not matches:
            mismatches.append(f"{key}={actual!r} (expected {expected!r})")
    return mismatches


def discover_runs(
    apis: Mapping[str, wandb.Api], sources: Sequence[SourceSpec]
) -> tuple[list[RunSpec], list[dict[str, Any]]]:
    selected: list[RunSpec] = []
    inventory: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for source in sources:
        api = apis[source.credential]
        for run in api.runs(f"{source.entity}/{source.project}", per_page=200):
            config = dict(run.config or {})
            summary = dict(run.summary or {})
            group = run.group or str(config.get("run_name") or "")
            algorithm = PRIMARY_GROUPS.get(group) or ROBUSTNESS_GROUPS.get(group)
            role = (
                "primary"
                if group in PRIMARY_GROUPS
                else "robustness"
                if group in ROBUSTNESS_GROUPS
                else "excluded"
            )
            mismatch_list = protocol_mismatches(config) if algorithm else []
            final_update = summary.get("num_updates")
            row = {
                "credential_source": source.credential,
                "entity": source.entity,
                "project": source.project,
                "run_id": run.id,
                "run_name": run.name,
                "group": group,
                "algorithm": algorithm or str(config.get("score_function") or "Unclassified"),
                "seed": config.get("seed"),
                "state": run.state,
                "role": role,
                "created_at": run.created_at,
                "final_logged_update": final_update,
                "has_replay_images": "images/replay_levels" in summary,
                "config_matches_protocol": not mismatch_list if algorithm else False,
                "protocol_mismatches": "; ".join(mismatch_list),
                "url": f"https://wandb.ai/{source.entity}/{source.project}/runs/{run.id}",
            }
            inventory.append(row)
            if role == "excluded":
                continue
            identity = (source.entity, source.project, run.id)
            if identity in seen:
                continue
            seen.add(identity)
            selected.append(
                RunSpec(
                    credential=source.credential,
                    entity=source.entity,
                    project=source.project,
                    run_id=run.id,
                    run_name=run.name,
                    group=group,
                    algorithm=algorithm or "Unclassified",
                    seed=int(config["seed"]) if config.get("seed") is not None else None,
                    state=run.state,
                    role=role,
                    created_at=run.created_at,
                    url=row["url"],
                    final_logged_update=int(final_update) if final_update is not None else None,
                    config_matches_protocol=not mismatch_list,
                    protocol_mismatches="; ".join(mismatch_list),
                )
            )

    selected.sort(key=lambda x: (x.role, x.algorithm, x.seed if x.seed is not None else -1))
    inventory.sort(key=lambda x: (x["role"], x["algorithm"], x.get("seed") or -1))
    return selected, inventory


def history_for_run(api: wandb.Api, spec: RunSpec) -> pd.DataFrame:
    run = api.run(f"{spec.entity}/{spec.project}/{spec.run_id}")
    history = run.history(samples=10_000, pandas=True)
    if history.empty:
        raise RuntimeError(f"No history returned for {spec.run_id}")
    if "num_updates" not in history:
        raise RuntimeError(f"num_updates missing from history for {spec.run_id}")
    history = history.dropna(subset=["num_updates"]).copy()
    history["num_updates"] = history["num_updates"].astype(int)
    history = history.sort_values("num_updates").drop_duplicates("num_updates", keep="last")
    return history


def exact_batch_metrics(filenames: Sequence[str]) -> dict[str, Any]:
    hashes = [stable_hash_from_filename(name) for name in filenames]
    counts = Counter(hashes)
    n = len(hashes)
    if n == 0:
        return {key: math.nan for key in DIVERSITY_METRICS[:5]}
    probabilities = np.asarray(list(counts.values()), dtype=float) / n
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return {
        "replay_batch_size": float(n),
        "exact_unique_count": float(len(counts)),
        "exact_unique_ratio": len(counts) / n,
        "duplicate_fraction": 1.0 - len(counts) / n,
        "dominant_count": float(max(counts.values())),
        "dominant_share": max(counts.values()) / n,
        "dominant_exact_hash": counts.most_common(1)[0][0],
        "shannon_effective_size": math.exp(entropy),
        "simpson_effective_size": 1.0 / float(np.sum(probabilities**2)),
    }


def _tile_scores(tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    red = (tiles[..., 0] > 180) & (tiles[..., 1] < 180) & (tiles[..., 2] < 180)
    green = (tiles[..., 1] > 180) & (tiles[..., 0] < 130) & (tiles[..., 2] < 130)
    return red.sum(axis=(-1, -2)), green.sum(axis=(-1, -2))


def _neighbors(row: int, col: int, height: int, width: int) -> Iterable[tuple[int, int]]:
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = row + dr, col + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _graph_metrics(
    wall_map: np.ndarray,
    agent: tuple[int, int] | None,
    goal: tuple[int, int] | None,
) -> dict[str, float]:
    height, width = wall_map.shape
    open_map = ~wall_map
    open_count = int(open_map.sum())
    metrics = {
        "wall_count": float(wall_map.sum()),
        "wall_fraction": float(wall_map.mean()),
        "solvable": math.nan,
        "shortest_path": math.nan,
        "manhattan_distance": math.nan,
        "path_stretch": math.nan,
        "reachable_fraction": math.nan,
        "dead_end_fraction": math.nan,
        "junction_fraction": math.nan,
    }
    if agent is None or goal is None or not open_map[agent] or not open_map[goal]:
        return metrics

    distance = {agent: 0}
    queue: deque[tuple[int, int]] = deque([agent])
    while queue:
        current = queue.popleft()
        for nxt in _neighbors(*current, height, width):
            if open_map[nxt] and nxt not in distance:
                distance[nxt] = distance[current] + 1
                queue.append(nxt)

    reachable = list(distance)
    degrees = []
    for row, col in reachable:
        degree = sum(open_map[nxt] for nxt in _neighbors(row, col, height, width))
        degrees.append(int(degree))

    shortest = distance.get(goal)
    manhattan = abs(agent[0] - goal[0]) + abs(agent[1] - goal[1])
    metrics.update(
        {
            "solvable": float(shortest is not None),
            "shortest_path": float(shortest) if shortest is not None else math.nan,
            "manhattan_distance": float(manhattan),
            "path_stretch": (
                float(shortest / manhattan)
                if shortest is not None and manhattan > 0
                else math.nan
            ),
            "reachable_fraction": len(reachable) / max(open_count, 1),
            "dead_end_fraction": float(np.mean(np.asarray(degrees) <= 1)),
            "junction_fraction": float(np.mean(np.asarray(degrees) >= 3)),
        }
    )
    return metrics


def decode_maze(content: bytes) -> DecodedMaze:
    image = np.asarray(Image.open(BytesIO(content)).convert("RGB"))
    if image.shape[:2] != (120, 120):
        raise ValueError(f"Unexpected replay image shape {image.shape}")

    tiles = image.reshape(15, 8, 15, 8, 3).transpose(0, 2, 1, 3, 4)
    centers = tiles[:, :, 4, 4, :]
    grayscale = (centers[..., 0] == centers[..., 1]) & (
        centers[..., 1] == centers[..., 2]
    )
    wall = grayscale & np.isin(centers[..., 0], (100, 146))
    red_score, green_score = _tile_scores(tiles)

    inner_wall = wall[1:-1, 1:-1].copy()
    inner_red = red_score[1:-1, 1:-1]
    inner_green = green_score[1:-1, 1:-1]

    agent_index = np.unravel_index(np.argmax(inner_red), inner_red.shape)
    goal_index = np.unravel_index(np.argmax(inner_green), inner_green.shape)
    agent = tuple(int(x) for x in agent_index) if inner_red[agent_index] > 0 else None
    goal = tuple(int(x) for x in goal_index) if inner_green[goal_index] > 0 else None
    if agent is not None:
        inner_wall[agent] = False
    if goal is not None:
        inner_wall[goal] = False
    return DecodedMaze(
        wall_map=inner_wall,
        agent=agent,
        goal=goal,
        metrics=_graph_metrics(inner_wall, agent, goal),
    )


def d4_variants(wall_map: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for turns in range(4):
        rotated = np.rot90(wall_map, turns)
        variants.extend((rotated, np.fliplr(rotated)))
    return variants


def d4_wall_distance(left: np.ndarray, right: np.ndarray) -> float:
    return min(float(np.mean(variant != right)) for variant in d4_variants(left))


def structural_batch_metrics(
    filenames: Sequence[str], decoded: Mapping[str, DecodedMaze]
) -> dict[str, float]:
    hashes = [stable_hash_from_filename(name) for name in filenames]
    mazes = [decoded[value] for value in hashes]
    result: dict[str, float] = {}
    for metric in STRUCTURAL_METRICS:
        values = np.asarray([maze.metrics[metric] for maze in mazes], dtype=float)
        finite = values[np.isfinite(values)]
        result[f"mean_{metric}"] = float(np.mean(finite)) if finite.size else math.nan
        result[f"median_{metric}"] = float(np.median(finite)) if finite.size else math.nan

    unique_by_topology: dict[bytes, np.ndarray] = {}
    for maze in mazes:
        unique_by_topology.setdefault(maze.wall_map.tobytes(), maze.wall_map)
    unique_maps = list(unique_by_topology.values())
    result["topology_unique_count"] = float(len(unique_maps))
    result["topology_unique_ratio"] = len(unique_maps) / max(len(mazes), 1)

    pairwise = [
        d4_wall_distance(mazes[i].wall_map, mazes[j].wall_map)
        for i in range(len(mazes))
        for j in range(i + 1, len(mazes))
    ]
    result["mean_pairwise_wall_distance_d4"] = (
        float(np.mean(pairwise)) if pairwise else math.nan
    )

    if len(unique_maps) >= 2:
        nearest = [
            min(
                d4_wall_distance(wall_map, candidate)
                for index, candidate in enumerate(unique_maps)
                if index != wall_index
            )
            for wall_index, wall_map in enumerate(unique_maps)
        ]
        result["mean_nearest_wall_distance_d4_unique"] = float(np.mean(nearest))
    else:
        result["mean_nearest_wall_distance_d4_unique"] = math.nan

    dominant_hash = Counter(hashes).most_common(1)[0][0]
    dominant_maze = decoded[dominant_hash]
    for metric in STRUCTURAL_METRICS:
        result[f"dominant_level_{metric}"] = float(dominant_maze.metrics[metric])
    return result


def image_url(spec: RunSpec, filename: str) -> str:
    safe_path = "/".join(quote(part) for part in filename.split("/"))
    return (
        f"https://api.wandb.ai/files/{spec.entity}/{spec.project}/"
        f"{spec.run_id}/{safe_path}"
    )


def download_one(
    spec: RunSpec,
    key: str,
    filename: str,
    content_hash: str,
    cache_dir: Path,
) -> tuple[str, bytes]:
    cache_path = cache_dir / f"{content_hash}.png"
    if cache_path.exists():
        return content_hash, cache_path.read_bytes()
    url = image_url(spec, filename)
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, auth=("api", key), timeout=45)
            response.raise_for_status()
            content = response.content
            Image.open(BytesIO(content)).verify()
            cache_path.write_bytes(content)
            return content_hash, content
        except Exception as exc:  # noqa: BLE001 - retried with bounded attempts
            last_error = exc
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Failed to download {filename}: {last_error}")


def decoded_images_for_rows(
    spec: RunSpec,
    key: str,
    rows: Sequence[tuple[int, Sequence[str]]],
    cache_dir: Path,
    workers: int,
) -> dict[str, DecodedMaze]:
    needed: dict[str, str] = {}
    for _, filenames in rows:
        for filename in filenames:
            needed.setdefault(stable_hash_from_filename(filename), filename)
    decoded: dict[str, DecodedMaze] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, spec, key, filename, content_hash, cache_dir): content_hash
            for content_hash, filename in needed.items()
        }
        for future in as_completed(futures):
            content_hash, content = future.result()
            decoded[content_hash] = decode_maze(content)
    return decoded


def extract_run_rows(
    spec: RunSpec,
    api: wandb.Api,
    key: str,
    structural_every: int,
    cache_dir: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    history = history_for_run(api, spec)
    output: list[dict[str, Any]] = []
    structural_rows: list[tuple[int, Sequence[str]]] = []
    image_rows = 0

    for _, source_row in history.iterrows():
        update = int(source_row["num_updates"])
        image_object = source_row.get("images/replay_levels")
        filenames: Sequence[str] = []
        if isinstance(image_object, dict):
            filenames = image_object.get("filenames") or []
        if filenames:
            image_rows += 1
        row: dict[str, Any] = {
            "entity": spec.entity,
            "project": spec.project,
            "run_id": spec.run_id,
            "run_name": spec.run_name,
            "run_url": spec.url,
            "group": spec.group,
            "algorithm": spec.algorithm,
            "seed": spec.seed,
            "role": spec.role,
            "run_state": spec.state,
            "num_updates": update,
            "progress_fraction": update / 30_000,
            "stage": stage_for_update(update),
        }
        for output_name, wandb_name in HISTORY_FIELDS.items():
            row[output_name] = scalar(source_row.get(wandb_name))
        if filenames:
            row.update(exact_batch_metrics(filenames))
        else:
            for metric in (
                "replay_batch_size",
                "exact_unique_count",
                "exact_unique_ratio",
                "duplicate_fraction",
                "dominant_count",
                "dominant_share",
                "dominant_exact_hash",
                "shannon_effective_size",
                "simpson_effective_size",
            ):
                row[metric] = None if metric == "dominant_exact_hash" else math.nan

        score_range = row["bank_score_q90"] - row["bank_score_q10"]
        priority_lift = row["bank_weighted_score"] - row["bank_mean_score"]
        row["score_priority_lift"] = priority_lift
        row["score_priority_lift_q80_normalized"] = (
            priority_lift / score_range
            if math.isfinite(priority_lift)
            and math.isfinite(score_range)
            and abs(score_range) > 1e-12
            else math.nan
        )
        output.append(row)
        if filenames and update % structural_every == 0:
            structural_rows.append((update, filenames))

    decode_failures = 0
    if structural_rows:
        decoded = decoded_images_for_rows(
            spec, key, structural_rows, cache_dir, workers
        )
        by_update = {row["num_updates"]: row for row in output}
        for update, filenames in structural_rows:
            try:
                by_update[update].update(structural_batch_metrics(filenames, decoded))
            except Exception:  # retain exact metrics and count the failed structural row
                decode_failures += 1

    quality = {
        **asdict(spec),
        "history_rows": int(len(history)),
        "expected_history_rows": 150,
        "image_rows": int(image_rows),
        "expected_image_rows": 150,
        "structural_rows_expected": int(
            sum(int(update % structural_every == 0) for update in history["num_updates"])
        ),
        "structural_decode_failures": int(decode_failures),
        "min_update": int(history["num_updates"].min()),
        "max_update": int(history["num_updates"].max()),
        "duplicate_update_rows": int(history["num_updates"].duplicated().sum()),
    }
    return output, quality


def mean_sem(values: pd.Series) -> tuple[float, float, int]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return math.nan, math.nan, 0
    mean = float(clean.mean())
    sem = float(clean.std(ddof=1) / math.sqrt(len(clean))) if len(clean) > 1 else math.nan
    return mean, sem, int(len(clean))


def aggregate_algorithm_timeseries(data: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "solve_rate",
        "return_mean",
        "bank_size",
        "score_priority_lift",
        "score_priority_lift_q80_normalized",
        *DIVERSITY_METRICS,
        *(f"mean_{metric}" for metric in STRUCTURAL_METRICS),
    ]
    records: list[dict[str, Any]] = []
    primary = data[data["role"] == "primary"]
    for (algorithm, update), group in primary.groupby(["algorithm", "num_updates"]):
        record: dict[str, Any] = {
            "algorithm": algorithm,
            "num_updates": int(update),
            "progress_fraction": int(update) / 30_000,
            "stage": stage_for_update(int(update)),
            "n_runs": int(group["run_id"].nunique()),
        }
        for metric in metrics:
            if metric not in group:
                continue
            mean, sem, count = mean_sem(group[metric])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sem"] = sem
            record[f"{metric}_n"] = count
        records.append(record)
    return pd.DataFrame(records).sort_values(["algorithm", "num_updates"])


def aggregate_stage_summary(data: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "solve_rate",
        "score_priority_lift",
        "score_priority_lift_q80_normalized",
        *DIVERSITY_METRICS,
        *(f"mean_{metric}" for metric in STRUCTURAL_METRICS),
    ]
    primary = data[data["role"] == "primary"]
    run_stage = (
        primary.groupby(["algorithm", "run_id", "seed", "stage"], dropna=False)[metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    records: list[dict[str, Any]] = []
    stage_order = ["Early (0-10k)", "Middle (10-20k)", "Late (20-30k)"]
    for (algorithm, stage), group in run_stage.groupby(["algorithm", "stage"]):
        record: dict[str, Any] = {
            "algorithm": algorithm,
            "stage": stage,
            "stage_order": stage_order.index(stage),
            "n_runs": int(group["run_id"].nunique()),
        }
        for metric in metrics:
            mean, sem, count = mean_sem(group[metric])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sem"] = sem
            record[f"{metric}_n"] = count
        records.append(record)
    return pd.DataFrame(records).sort_values(["algorithm", "stage_order"])


def summarize_runs(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for run_id, group in data.groupby("run_id"):
        group = group.sort_values("num_updates")
        solve = group.dropna(subset=["solve_rate"])
        final = solve.iloc[-1] if not solve.empty else None
        peak = solve.loc[solve["solve_rate"].idxmax()] if not solve.empty else None
        record: dict[str, Any] = {
            "run_id": run_id,
            "run_name": group.iloc[0]["run_name"],
            "run_url": group.iloc[0]["run_url"],
            "algorithm": group.iloc[0]["algorithm"],
            "seed": group.iloc[0]["seed"],
            "role": group.iloc[0]["role"],
            "run_state": group.iloc[0]["run_state"],
            "logged_updates": int(group["num_updates"].max()),
            "history_rows": int(len(group)),
            "journey_mean_solve_rate": float(solve["solve_rate"].mean()) if not solve.empty else math.nan,
            "final_solve_rate": float(final["solve_rate"]) if final is not None else math.nan,
            "peak_solve_rate": float(peak["solve_rate"]) if peak is not None else math.nan,
            "peak_solve_update": int(peak["num_updates"]) if peak is not None else math.nan,
            "peak_to_final_drop": (
                float(peak["solve_rate"] - final["solve_rate"])
                if peak is not None and final is not None
                else math.nan
            ),
        }
        for metric in (
            *DIVERSITY_METRICS,
            "score_priority_lift",
            "score_priority_lift_q80_normalized",
            *(f"mean_{value}" for value in STRUCTURAL_METRICS),
        ):
            record[f"journey_{metric}"] = (
                float(group[metric].mean()) if metric in group else math.nan
            )
        records.append(record)
    return pd.DataFrame(records).sort_values(["role", "algorithm", "seed"])


def correlation_summary(data: pd.DataFrame, lead_updates: int = 1_000) -> pd.DataFrame:
    metrics = [
        "exact_unique_ratio",
        "dominant_share",
        "simpson_effective_size",
        "mean_pairwise_wall_distance_d4",
        "mean_shortest_path",
        "mean_solvable",
        "score_priority_lift_q80_normalized",
    ]
    records: list[dict[str, Any]] = []
    for run_id, group in data[data["role"] == "primary"].groupby("run_id"):
        group = group.sort_values("num_updates").copy()
        future = group[["num_updates", "solve_rate"]].rename(
            columns={"num_updates": "future_update", "solve_rate": "future_solve_rate"}
        )
        group["future_update"] = group["num_updates"] + lead_updates
        aligned = group.merge(future, on="future_update", how="left")
        for metric in metrics:
            for outcome, label in (
                ("solve_rate", "contemporaneous"),
                ("future_solve_rate", f"lead_{lead_updates}"),
            ):
                valid = aligned[[metric, outcome]].dropna()
                if len(valid) < 8 or valid[metric].nunique() < 2 or valid[outcome].nunique() < 2:
                    rho, p_value = math.nan, math.nan
                else:
                    result = spearmanr(valid[metric], valid[outcome])
                    rho, p_value = float(result.statistic), float(result.pvalue)
                records.append(
                    {
                        "run_id": run_id,
                        "algorithm": group.iloc[0]["algorithm"],
                        "seed": group.iloc[0]["seed"],
                        "metric": metric,
                        "outcome": label,
                        "n_points": int(len(valid)),
                        "spearman_rho": rho,
                        "p_value": p_value,
                    }
                )
    return pd.DataFrame(records)


def matched_maxmc_log_relative(run_summary: pd.DataFrame) -> pd.DataFrame:
    primary = run_summary[
        (run_summary["role"] == "primary")
        & run_summary["algorithm"].isin(("MaxMC", "Log-relative transfer"))
    ]
    metrics = [
        "final_solve_rate",
        "journey_mean_solve_rate",
        "peak_to_final_drop",
        "journey_exact_unique_ratio",
        "journey_simpson_effective_size",
        "journey_mean_pairwise_wall_distance_d4",
        "journey_mean_shortest_path",
        "journey_mean_solvable",
    ]
    records: list[dict[str, Any]] = []
    for seed, group in primary.groupby("seed"):
        indexed = group.set_index("algorithm")
        if not {"MaxMC", "Log-relative transfer"}.issubset(indexed.index):
            continue
        record: dict[str, Any] = {"seed": int(seed)}
        for metric in metrics:
            maxmc = scalar(indexed.loc["MaxMC", metric])
            log_relative = scalar(indexed.loc["Log-relative transfer", metric])
            record[f"maxmc_{metric}"] = maxmc
            record[f"log_relative_{metric}"] = log_relative
            record[f"delta_{metric}"] = log_relative - maxmc
        records.append(record)
    return pd.DataFrame(records).sort_values("seed")


def paired_comparison_stats(matched: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for metric in ("final_solve_rate", "journey_mean_solve_rate"):
        maxmc = matched[f"maxmc_{metric}"].dropna()
        log_relative = matched.loc[maxmc.index, f"log_relative_{metric}"]
        delta = log_relative - maxmc
        n = len(delta)
        sem = float(delta.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
        if n > 1:
            confidence_low, confidence_high = t.interval(
                0.95, n - 1, loc=float(delta.mean()), scale=sem
            )
            test = ttest_rel(log_relative, maxmc)
            statistic, p_value = float(test.statistic), float(test.pvalue)
        else:
            confidence_low = confidence_high = statistic = p_value = math.nan
        records.append(
            {
                "metric": metric,
                "n_matched_seeds": n,
                "mean_log_relative_minus_maxmc": float(delta.mean()),
                "standard_error": sem,
                "confidence_95_low": float(confidence_low),
                "confidence_95_high": float(confidence_high),
                "paired_t_statistic": statistic,
                "paired_t_p_value": p_value,
            }
        )
    return pd.DataFrame(records)


def aggregate_correlations(correlations: pd.DataFrame) -> pd.DataFrame:
    return (
        correlations.groupby(["metric", "outcome"])["spearman_rho"]
        .agg(
            median_rho="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            n_runs="count",
        )
        .reset_index()
    )


def crash_window_summary(data: pd.DataFrame, window_updates: int = 2_000) -> pd.DataFrame:
    metrics = [
        "solve_rate",
        "exact_unique_ratio",
        "dominant_share",
        "simpson_effective_size",
        "score_priority_lift_q80_normalized",
        "mean_pairwise_wall_distance_d4",
        "mean_shortest_path",
        "mean_solvable",
        "mean_dead_end_fraction",
    ]
    records: list[dict[str, Any]] = []
    for run_id, group in data[data["role"] == "primary"].groupby("run_id"):
        group = group.sort_values("num_updates")
        valid_solve = group.dropna(subset=["solve_rate"])
        if valid_solve.empty:
            continue
        peak_row = valid_solve.loc[valid_solve["solve_rate"].idxmax()]
        peak_update = int(peak_row["num_updates"])
        final_update = int(group["num_updates"].max())
        peak_window = group[
            group["num_updates"].between(max(0, peak_update - window_updates), peak_update)
        ]
        final_window = group[
            group["num_updates"].between(final_update - window_updates, final_update)
        ]
        record: dict[str, Any] = {
            "run_id": run_id,
            "algorithm": group.iloc[0]["algorithm"],
            "seed": group.iloc[0]["seed"],
            "peak_update": peak_update,
            "final_update": final_update,
            "peak_solve_rate": float(peak_row["solve_rate"]),
            "final_solve_rate": float(valid_solve.iloc[-1]["solve_rate"]),
        }
        for metric in metrics:
            before = float(peak_window[metric].mean()) if metric in peak_window else math.nan
            after = float(final_window[metric].mean()) if metric in final_window else math.nan
            record[f"peak_window_{metric}"] = before
            record[f"final_window_{metric}"] = after
            record[f"delta_{metric}"] = after - before
        records.append(record)
    return pd.DataFrame(records).sort_values(["algorithm", "seed"])


def unsolvable_priority_events(data: pd.DataFrame) -> pd.DataFrame:
    structural = data[data["mean_solvable"].notna()].copy()
    future = structural[["run_id", "num_updates", "solve_rate"]].rename(
        columns={"num_updates": "future_update", "solve_rate": "solve_rate_plus_1000"}
    )
    structural["future_update"] = structural["num_updates"] + 1_000
    structural = structural.merge(future, on=["run_id", "future_update"], how="left")
    columns = [
        "run_id",
        "run_url",
        "algorithm",
        "seed",
        "num_updates",
        "solve_rate",
        "solve_rate_plus_1000",
        "mean_solvable",
        "dominant_share",
        "dominant_exact_hash",
        "dominant_level_solvable",
        "dominant_level_shortest_path",
        "exact_unique_ratio",
        "simpson_effective_size",
    ]
    return structural[structural["mean_solvable"] < 0.9][columns].sort_values(
        ["mean_solvable", "algorithm", "seed", "num_updates"]
    )


def algorithm_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    primary = run_summary[run_summary["role"] == "primary"]
    metrics = [
        "final_solve_rate",
        "journey_mean_solve_rate",
        "peak_to_final_drop",
        "journey_exact_unique_ratio",
        "journey_duplicate_fraction",
        "journey_dominant_share",
        "journey_simpson_effective_size",
        "journey_mean_pairwise_wall_distance_d4",
        "journey_mean_nearest_wall_distance_d4_unique",
        "journey_mean_solvable",
        "journey_mean_shortest_path",
        "journey_mean_reachable_fraction",
        "journey_mean_dead_end_fraction",
    ]
    records: list[dict[str, Any]] = []
    for algorithm, group in primary.groupby("algorithm"):
        record: dict[str, Any] = {
            "algorithm": algorithm,
            "n_runs": int(group["run_id"].nunique()),
            "n_complete_30k": int((group["logged_updates"] >= 30_000).sum()),
        }
        for metric in metrics:
            mean, sem, count = mean_sem(group[metric])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sem"] = sem
            record[f"{metric}_n"] = count
            record[f"{metric}_min"] = float(group[metric].min())
            record[f"{metric}_max"] = float(group[metric].max())
        records.append(record)
    return pd.DataFrame(records).sort_values("algorithm")


def quality_summary(
    quality_rows: Sequence[Mapping[str, Any]], inventory: pd.DataFrame
) -> dict[str, Any]:
    selected = pd.DataFrame(quality_rows)
    primary = selected[selected["role"] == "primary"]
    return {
        "selected_runs": int(len(selected)),
        "primary_runs": int(len(primary)),
        "robustness_runs": int((selected["role"] == "robustness").sum()),
        "complete_primary_runs": int((primary["max_update"] >= 30_000).sum()),
        "partial_primary_runs": int((primary["max_update"] < 30_000).sum()),
        "primary_history_rows": int(primary["history_rows"].sum()),
        "primary_image_rows": int(primary["image_rows"].sum()),
        "missing_primary_image_rows": int(
            (primary["expected_image_rows"] - primary["image_rows"]).clip(lower=0).sum()
        ),
        "structural_decode_failures": int(primary["structural_decode_failures"].sum()),
        "protocol_match_primary_runs": int(primary["config_matches_protocol"].sum()),
        "inventory_runs": int(len(inventory)),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return clean_for_json(value.item())
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--structural-every", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--include-secondary", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.structural_every <= 0 or 30_000 % args.structural_every != 0:
        raise ValueError("--structural-every must be a positive divisor of 30000")

    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(tempfile.gettempdir()) / "jaxued-plr-replay-image-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    keys = credentials(repo, args.include_secondary)
    apis = {name: wandb.Api(api_key=key, timeout=90) for name, key in keys.items()}
    sources = [SourceSpec("env", "rundong-liu", "JAXUED_TEST")]
    if args.include_secondary:
        sources.append(SourceSpec("secondary", "autocurriculum", "JAXUED_TEST"))

    specs, inventory_rows = discover_runs(apis, sources)
    if not specs:
        raise RuntimeError("No selected PLR runs were discovered")

    all_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] {spec.algorithm} seed={spec.seed} "
            f"run={spec.run_id} role={spec.role}",
            flush=True,
        )
        rows, quality = extract_run_rows(
            spec,
            apis[spec.credential],
            keys[spec.credential],
            args.structural_every,
            cache_dir,
            args.workers,
        )
        all_rows.extend(rows)
        quality_rows.append(quality)

    data = pd.DataFrame(all_rows).sort_values(["role", "algorithm", "seed", "num_updates"])
    inventory = pd.DataFrame(inventory_rows)
    quality = pd.DataFrame(quality_rows)
    run_summary = summarize_runs(data)
    algorithm_timeseries = aggregate_algorithm_timeseries(data)
    stage_summary = aggregate_stage_summary(data)
    correlations = correlation_summary(data)
    matched = matched_maxmc_log_relative(run_summary)
    paired_stats = paired_comparison_stats(matched)
    crash = crash_window_summary(data)
    unsolvable_events = unsolvable_priority_events(data)
    algorithms = algorithm_summary(run_summary)
    correlation_aggregate = aggregate_correlations(correlations)

    outputs = {
        "run_inventory.csv": inventory,
        "data_quality.csv": quality,
        "replay_metrics.csv": data,
        "algorithm_timeseries.csv": algorithm_timeseries,
        "stage_summary.csv": stage_summary,
        "run_summary.csv": run_summary,
        "algorithm_summary.csv": algorithms,
        "correlation_summary.csv": correlations,
        "matched_maxmc_log_relative.csv": matched,
        "paired_comparison_stats.csv": paired_stats,
        "crash_window_summary.csv": crash,
        "unsolvable_priority_events.csv": unsolvable_events,
        "correlation_aggregate.csv": correlation_aggregate,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    generated_at = datetime.now(timezone.utc).isoformat()
    ranks = np.arange(1, EXPECTED_PROTOCOL["level_buffer_capacity"] + 1, dtype=float)
    pure_rank_weights = ranks ** (-1.0 / EXPECTED_PROTOCOL["temperature"])
    pure_rank_weights /= pure_rank_weights.sum()
    score_component_top_share = float(
        (1.0 - EXPECTED_PROTOCOL["staleness_coeff"]) * pure_rank_weights[0]
    )
    summary = {
        "generated_at": generated_at,
        "source": "Weights & Biases run histories and replay-level image files",
        "time_window_updates": [200, 30_000],
        "structural_cadence_updates": args.structural_every,
        "stage_definition": {
            "early": "updates 200-10000",
            "middle": "updates 10200-20000",
            "late": "updates 20200-30000",
        },
        "quality": quality_summary(quality_rows, inventory),
        "sampler_theory": {
            "pure_rank_top_probability": float(pure_rank_weights[0]),
            "pure_rank_effective_bank_size": float(1.0 / np.sum(pure_rank_weights**2)),
            "score_component_top_share_after_staleness_mix": score_component_top_share,
            "expected_top_copies_in_batch_from_score_component": float(
                EXPECTED_PROTOCOL["num_train_envs"] * score_component_top_share
            ),
            "formula": "rank_weight proportional to rank^(-1/temperature), mixed as 0.7 score weight + 0.3 staleness weight",
        },
        "algorithm_summary": clean_for_json(algorithms.to_dict(orient="records")),
        "matched_maxmc_log_relative": clean_for_json(matched.to_dict(orient="records")),
        "metric_definitions": {
            "exact_unique_ratio": "Distinct rendered maze hashes divided by 32 sampled replay slots.",
            "duplicate_fraction": "One minus exact_unique_ratio; repeated exposure beyond one copy per identity.",
            "dominant_share": "Largest exact-identity multiplicity divided by 32.",
            "shannon_effective_size": "exp(-sum p_i log p_i)) over exact identities in a replay batch.",
            "simpson_effective_size": "1/sum p_i^2 over exact identities in a replay batch.",
            "mean_pairwise_wall_distance_d4": "Mean pairwise normalized wall Hamming distance across all 32 slots after best rotation/reflection alignment.",
            "mean_nearest_wall_distance_d4_unique": "For each unique wall map, distance to its nearest distinct topology after best rotation/reflection alignment.",
            "solvable": "Whether an orthogonal open-cell path exists from rendered agent start to goal.",
            "shortest_path": "Breadth-first shortest open-cell path length, conditional on solvability.",
            "reachable_fraction": "Fraction of non-wall cells reachable from the agent start.",
            "dead_end_fraction": "Share of reachable cells with at most one open orthogonal neighbor.",
            "score_priority_lift": "W&B level_sampler/weighted_score minus level_sampler/mean_score.",
            "score_priority_lift_q80_normalized": "Priority lift divided by bank score q90-q10; unavailable for older runs without quantile logs.",
            "solve_rate": "Mean solve rate on the fixed eight-level evaluation suite, not replay-batch success.",
        },
        "limitations": [
            "W&B replay images are the last sampled 32-level replay batch at each evaluation, not the full 4000-level PLR bank.",
            "Raw score magnitudes have different meanings and scales across scoring functions; cross-function comparisons use structural metrics and normalized priority lift where available.",
            "Structural difficulty proxies do not measure policy-conditioned success on each replay level.",
            "Repeated evaluation time points are not independent experimental replicates; seed is the replication unit.",
        ],
    }
    write_json(output_dir / "analysis_summary.json", clean_for_json(summary))
    print(json.dumps(summary["quality"], indent=2), flush=True)


if __name__ == "__main__":
    main()
