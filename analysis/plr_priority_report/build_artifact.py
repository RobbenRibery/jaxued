#!/usr/bin/env python3
"""Build the canonical Data Analytics report artifact from reviewed outputs."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TITLE = "What PLR Actually Prioritized in Maze Training"
SOURCE_ID = "wandb_plr_priority_analysis"


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    value = frame.replace({np.nan: None}).to_dict(orient="records")
    return [clean(row) for row in value]


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def materialize_widget_datasets(
    datasets: dict[str, pd.DataFrame | list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Materialize embedded widget rows through the SQL recorded as provenance."""
    rendered: dict[str, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    with sqlite3.connect(":memory:") as connection:
        for dataset_name, value in datasets.items():
            frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
            table_name = f"widget_{dataset_name}"
            frame.to_sql(table_name, connection, index=False, if_exists="replace")
            sql = f'SELECT * FROM "{table_name}"'
            rendered[dataset_name] = records(pd.read_sql_query(sql, connection))
            sources.append(
                {
                    "id": f"source_{dataset_name}",
                    "label": f"Materialized {dataset_name} widget dataset",
                    "path": "analysis/plr_priority_report/artifact.json",
                    "href": "https://wandb.ai/rundong-liu/JAXUED_TEST",
                    "query": {
                        "engine": "sqlite",
                        "sql": sql,
                        "description": (
                            "Executed against the in-memory widget table populated from the "
                            "reviewed PLR analysis CSV outputs."
                        ),
                    },
                }
            )
    return rendered, sources


def main() -> None:
    analysis = json.loads((DATA / "analysis_summary.json").read_text())
    run_summary = pd.read_csv(DATA / "run_summary.csv")
    algorithm_summary = pd.read_csv(DATA / "algorithm_summary.csv")
    algorithm_timeseries = pd.read_csv(DATA / "algorithm_timeseries.csv")
    stage_summary = pd.read_csv(DATA / "stage_summary.csv")
    replay = pd.read_csv(DATA / "replay_metrics.csv")
    matched = pd.read_csv(DATA / "matched_maxmc_log_relative.csv")
    paired_stats = pd.read_csv(DATA / "paired_comparison_stats.csv")
    correlations = pd.read_csv(DATA / "correlation_aggregate.csv")
    anomalies = pd.read_csv(DATA / "unsolvable_priority_events.csv")
    quality = pd.read_csv(DATA / "data_quality.csv")

    primary_runs = run_summary[run_summary["role"] == "primary"].copy()
    primary_replay = replay[replay["role"] == "primary"].copy()
    run_count = int(primary_runs["run_id"].nunique())
    dominant_share = float(primary_runs["journey_dominant_share"].mean())
    exact_unique_ratio = float(primary_runs["journey_exact_unique_ratio"].mean())
    effective_size = float(primary_runs["journey_simpson_effective_size"].mean())
    theoretical_share = float(
        analysis["sampler_theory"]["score_component_top_share_after_staleness_mix"]
    )
    final_delta = float(matched["delta_final_solve_rate"].mean())
    journey_delta = float(matched["delta_journey_mean_solve_rate"].mean())
    primary_quality = quality[quality["role"] == "primary"]
    image_batches = int(primary_quality["image_rows"].sum())
    replay_slots = image_batches * 32
    structural_batches = int(primary_quality["structural_rows_expected"].sum())
    structural_slots = structural_batches * 32

    headline = pd.DataFrame(
        [
            {
                "dominant_share": dominant_share,
                "theoretical_dominant_share": theoretical_share,
                "exact_unique_ratio": exact_unique_ratio,
                "mean_exact_unique_count": 32 * exact_unique_ratio,
                "effective_batch_size": effective_size,
                "nominal_batch_size": 32,
                "matched_final_delta": final_delta,
                "matched_journey_delta": journey_delta,
                "matched_seed_count": int(len(matched)),
                "replay_slots": replay_slots,
                "structural_slots": structural_slots,
                "decode_failures": int(primary_quality["structural_decode_failures"].sum()),
            }
        ]
    )

    concentration = algorithm_timeseries[
        algorithm_timeseries["num_updates"] % 1_000 == 0
    ][
        [
            "algorithm",
            "num_updates",
            "stage",
            "n_runs",
            "simpson_effective_size_mean",
            "simpson_effective_size_sem",
            "exact_unique_ratio_mean",
            "dominant_share_mean",
            "solve_rate_mean",
        ]
    ].copy()

    difficulty = algorithm_summary[
        [
            "algorithm",
            "n_runs",
            "n_complete_30k",
            "journey_mean_shortest_path_mean",
            "journey_mean_shortest_path_sem",
            "journey_mean_solvable_mean",
            "journey_mean_reachable_fraction_mean",
            "journey_mean_dead_end_fraction_mean",
            "journey_exact_unique_ratio_mean",
            "journey_mean_nearest_wall_distance_d4_unique_mean",
            "final_solve_rate_mean",
            "journey_mean_solve_rate_mean",
        ]
    ].copy()

    performance_rows: list[dict[str, Any]] = []
    for row in records(algorithm_summary):
        for measure, field in (
            ("Journey mean", "journey_mean_solve_rate_mean"),
            ("Final checkpoint", "final_solve_rate_mean"),
        ):
            performance_rows.append(
                {
                    "algorithm": row["algorithm"],
                    "performance_measure": measure,
                    "solve_rate": row[field],
                    "n_runs": row["n_runs"],
                    "n_complete_30k": row["n_complete_30k"],
                    "peak_to_final_drop_mean": row["peak_to_final_drop_mean"],
                    "mean_effective_batch_size": row[
                        "journey_simpson_effective_size_mean"
                    ],
                }
            )

    topology_stage = stage_summary[
        [
            "algorithm",
            "stage",
            "stage_order",
            "n_runs",
            "mean_nearest_wall_distance_d4_unique_mean",
            "mean_pairwise_wall_distance_d4_mean",
            "exact_unique_ratio_mean",
            "dominant_share_mean",
            "mean_shortest_path_mean",
            "solve_rate_mean",
        ]
    ].copy()

    relationship = primary_runs[
        [
            "run_id",
            "run_name",
            "run_url",
            "algorithm",
            "seed",
            "journey_mean_solve_rate",
            "final_solve_rate",
            "peak_to_final_drop",
            "journey_simpson_effective_size",
            "journey_exact_unique_ratio",
            "journey_dominant_share",
            "journey_mean_pairwise_wall_distance_d4",
            "journey_mean_shortest_path",
        ]
    ].copy()
    relationship["run_label"] = relationship.apply(
        lambda row: f"{row['algorithm']} · seed {int(row['seed'])}", axis=1
    )

    log_seed_trajectory = primary_replay[
        (primary_replay["algorithm"] == "Log-relative transfer")
        & (primary_replay["seed"].isin([3, 4, 5]))
        & (primary_replay["num_updates"] % 1_000 == 0)
    ][
        [
            "run_id",
            "run_url",
            "seed",
            "num_updates",
            "stage",
            "solve_rate",
            "exact_unique_ratio",
            "dominant_share",
            "simpson_effective_size",
            "score_priority_lift_q80_normalized",
            "mean_pairwise_wall_distance_d4",
            "mean_shortest_path",
            "mean_solvable",
            "agent_entropy",
            "agent_policy_loss",
            "agent_value_loss",
        ]
    ].copy()
    seed_labels = {
        3: "Seed 3 · 51.25% final",
        4: "Seed 4 · 12.5% final",
        5: "Seed 5 · 0% final",
    }
    seed_styles = {3: "solid", 4: "dashed", 5: "dotted"}
    log_seed_trajectory["seed_label"] = log_seed_trajectory["seed"].map(seed_labels)
    log_seed_trajectory["line_style"] = log_seed_trajectory["seed"].map(seed_styles)

    seed_stage = (
        log_seed_trajectory.groupby(["seed", "seed_label", "stage"], dropna=False)[
            [
                "solve_rate",
                "exact_unique_ratio",
                "dominant_share",
                "simpson_effective_size",
                "score_priority_lift_q80_normalized",
                "mean_pairwise_wall_distance_d4",
                "mean_shortest_path",
                "mean_solvable",
                "agent_entropy",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    stage_order = {"Early (0-10k)": 0, "Middle (10-20k)": 1, "Late (20-30k)": 2}
    seed_stage["stage_order"] = seed_stage["stage"].map(stage_order)
    seed_stage = seed_stage.sort_values(["seed", "stage_order"])

    matched_display = matched[
        [
            "seed",
            "maxmc_final_solve_rate",
            "log_relative_final_solve_rate",
            "delta_final_solve_rate",
            "delta_journey_mean_solve_rate",
            "delta_journey_exact_unique_ratio",
            "delta_journey_simpson_effective_size",
            "delta_journey_mean_shortest_path",
        ]
    ].copy()

    anomalies_display = anomalies[
        [
            "algorithm",
            "seed",
            "num_updates",
            "solve_rate",
            "solve_rate_plus_1000",
            "mean_solvable",
            "dominant_share",
            "dominant_level_solvable",
            "exact_unique_ratio",
            "simpson_effective_size",
            "run_url",
        ]
    ].copy()

    correlation_display = correlations.copy()
    correlation_display["metric_label"] = correlation_display["metric"].map(
        {
            "dominant_share": "Dominant-maze share",
            "exact_unique_ratio": "Exact unique ratio",
            "mean_pairwise_wall_distance_d4": "Pairwise wall distance",
            "mean_shortest_path": "Shortest path",
            "mean_solvable": "Solvable exposure",
            "score_priority_lift_q80_normalized": "Normalized priority lift",
            "simpson_effective_size": "Effective batch size",
        }
    )
    correlation_display["outcome_label"] = correlation_display["outcome"].map(
        {"contemporaneous": "Current solve rate", "lead_1000": "Solve rate +1k updates"}
    )

    metric_rows = [
        {"metric": key, "definition": value}
        for key, value in analysis["metric_definitions"].items()
    ]

    quality_by_algorithm = (
        quality[quality["role"] == "primary"]
        .groupby("algorithm")
        .agg(
            runs=("run_id", "nunique"),
            complete_30k=("max_update", lambda values: int((values >= 30_000).sum())),
            history_rows=("history_rows", "sum"),
            replay_image_rows=("image_rows", "sum"),
            structural_snapshots=("structural_rows_expected", "sum"),
            decode_failures=("structural_decode_failures", "sum"),
            protocol_matches=("config_matches_protocol", "sum"),
        )
        .reset_index()
    )

    editor_sensitivity = run_summary[
        run_summary["algorithm"].isin(
            ["Editor transfer", "Editor transfer (earlier replication)"]
        )
    ][
        [
            "algorithm",
            "seed",
            "final_solve_rate",
            "journey_mean_solve_rate",
            "journey_exact_unique_ratio",
            "journey_simpson_effective_size",
            "peak_to_final_drop",
            "run_url",
        ]
    ].copy()
    editor_sensitivity["cohort"] = editor_sensitivity["algorithm"].map(
        {
            "Editor transfer": "21 Aug cohort",
            "Editor transfer (earlier replication)": "19 Aug cohort",
        }
    )

    source = {
        "id": SOURCE_ID,
        "label": "replay_metrics.csv generated from W&B JAXUED_TEST histories and replay images",
        "path": "analysis/plr_priority_report/data/replay_metrics.csv",
        "href": "https://wandb.ai/rundong-liu/JAXUED_TEST",
        "query": {
            "engine": "sqlite",
            "sql": 'SELECT * FROM "widget_headline"',
            "description": "Executed against the materialized headline widget table.",
        },
    }

    cards = [
        {
            "id": "card_concentration",
            "dataset": "headline",
            "sourceId": SOURCE_ID,
            "description": "Observed exposure concentration versus the sampler-derived expectation.",
            "metrics": [
                {"label": "Dominant maze share", "field": "dominant_share", "format": "percent"},
                {"label": "Sampler expectation", "field": "theoretical_dominant_share", "format": "percent"},
                {"label": "Unique slots", "field": "exact_unique_ratio", "format": "percent"},
            ],
        },
        {
            "id": "card_effective_batch",
            "dataset": "headline",
            "sourceId": SOURCE_ID,
            "description": "Inverse-Simpson effective number of distinct exposures in each 32-slot replay batch.",
            "metrics": [
                {"label": "Effective batch size", "field": "effective_batch_size", "format": "number"},
                {"label": "Nominal slots", "field": "nominal_batch_size", "format": "number"},
                {"label": "Exact unique count", "field": "mean_exact_unique_count", "format": "number"},
            ],
        },
        {
            "id": "card_matched_delta",
            "dataset": "headline",
            "sourceId": SOURCE_ID,
            "description": "Paired log-relative minus MaxMC performance across seeds 0-5.",
            "metrics": [
                {"label": "Final solve-rate delta", "field": "matched_final_delta", "format": "percent", "signed": True},
                {"label": "Journey delta", "field": "matched_journey_delta", "format": "percent", "signed": True},
                {"label": "Matched seeds", "field": "matched_seed_count", "format": "number"},
            ],
        },
        {
            "id": "card_evidence",
            "dataset": "headline",
            "sourceId": SOURCE_ID,
            "description": "Reviewed replay slots and structural decoding coverage in the primary cohort.",
            "metrics": [
                {"label": "Replay slots", "field": "replay_slots", "format": "compact"},
                {"label": "Structural slots", "field": "structural_slots", "format": "compact"},
                {"label": "Decode failures", "field": "decode_failures", "format": "number"},
            ],
        },
    ]

    charts = [
        {
            "id": "effective_size_trend",
            "title": "Effective replay-batch size across training",
            "subtitle": "Inverse-Simpson effective count in each 32-slot sampled replay batch; 1k-update snapshots and algorithm means across available seeds",
            "showDescription": True,
            "intent": "trend",
            "question": "Does sampled replay exposure become more or less concentrated during 30k updates?",
            "rationale": "A multi-series line chart uses 30 evenly spaced snapshots and reveals whether concentration changes continuously or only at isolated steps; all 150 evaluations remain in the supporting dataset.",
            "comparisonContext": {
                "grain": "200-update evaluation step",
                "denominator": "32 sampled replay slots",
                "normalization": "inverse Simpson effective count",
                "unit": "effective mazes",
            },
            "type": "line",
            "dataset": "concentration_timeseries",
            "sourceId": SOURCE_ID,
            "encodings": {
                "x": {"field": "num_updates", "type": "quantitative", "label": "Training updates"},
                "y": {"field": "simpson_effective_size_mean", "type": "quantitative", "label": "Effective mazes"},
                "color": {"field": "algorithm", "type": "nominal", "label": "Scoring function"},
                "tooltip": [
                    {"field": "exact_unique_ratio_mean", "format": "percent", "label": "Unique slots"},
                    {"field": "dominant_share_mean", "format": "percent", "label": "Dominant share"},
                    {"field": "n_runs", "format": "number", "label": "Runs"},
                ],
            },
            "layout": "full",
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "sort": "labelAsc", "title": "Scoring function"},
            "labels": {"values": "endpoints"},
            "settings": {"showPoints": "never"},
        },
        {
            "id": "difficulty_algorithm",
            "title": "Shortest-path difficulty by scoring function",
            "subtitle": "Mean BFS path length among solvable prioritized replay slots; 30 structural snapshots per complete run",
            "showDescription": True,
            "intent": "comparison",
            "question": "Which scoring functions prioritize structurally longer solution paths?",
            "rationale": "A sorted horizontal bar chart makes the five algorithm-level path-length differences directly comparable without implying a temporal trend.",
            "comparisonContext": {
                "grain": "run-level journey average",
                "denominator": "solvable prioritized replay slots",
                "unit": "grid steps",
            },
            "type": "horizontalBar",
            "dataset": "difficulty_by_algorithm",
            "sourceId": SOURCE_ID,
            "encodings": {
                "x": {"field": "algorithm", "type": "nominal", "label": "Scoring function"},
                "y": {"field": "journey_mean_shortest_path_mean", "type": "quantitative", "label": "Shortest path"},
                "tooltip": [
                    {"field": "journey_mean_solvable_mean", "format": "percent", "label": "Solvable exposure"},
                    {"field": "journey_mean_dead_end_fraction_mean", "format": "percent", "label": "Dead-end cells"},
                    {"field": "n_runs", "format": "number", "label": "Runs"},
                ],
            },
            "layout": "full",
            "palette": {"kind": "sequential"},
            "labels": {"values": "all"},
            "settings": {"orientation": "horizontal", "sort": "descending", "showValues": True},
        },
        {
            "id": "topology_stage",
            "title": "Nearest-topology distance across training stages",
            "subtitle": "Rotation/reflection-invariant wall Hamming distance after exact duplicates are removed",
            "showDescription": True,
            "intent": "comparison",
            "question": "Does the repertoire of distinct sampled wall maps collapse from early to late training?",
            "rationale": "Grouped stage bars expose early/middle/late changes while avoiding another dense time-series chart.",
            "comparisonContext": {
                "grain": "run-stage average",
                "normalization": "fraction of 169 wall cells after best D4 alignment",
                "unit": "normalized wall distance",
            },
            "type": "bar",
            "dataset": "topology_stage",
            "sourceId": SOURCE_ID,
            "encodings": {
                "x": {"field": "algorithm", "type": "nominal", "label": "Scoring function"},
                "y": {"field": "mean_nearest_wall_distance_d4_unique_mean", "type": "quantitative", "label": "Nearest wall distance"},
                "color": {"field": "stage", "type": "ordinal", "label": "Training stage"},
                "tooltip": [
                    {"field": "exact_unique_ratio_mean", "format": "percent", "label": "Exact unique slots"},
                    {"field": "dominant_share_mean", "format": "percent", "label": "Dominant share"},
                    {"field": "n_runs", "format": "number", "label": "Runs"},
                ],
            },
            "layout": "full",
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "sort": "spec", "title": "Training stage"},
            "settings": {"groupMode": "grouped", "sort": "none"},
        },
        {
            "id": "performance_algorithm",
            "title": "Evaluation solve rate by scoring function",
            "subtitle": "Journey mean and final checkpoint; seed is the replication unit and one absolute-advantage run ends at 29.6k",
            "showDescription": True,
            "intent": "comparison",
            "question": "How do evaluation outcomes differ across the five scoring families?",
            "rationale": "Grouped bars compare two decision-relevant summaries without overplotting 150 noisy evaluations per run.",
            "comparisonContext": {
                "grain": "run-level summary",
                "denominator": "fixed eight-level evaluation suite",
                "unit": "solve rate",
            },
            "type": "bar",
            "dataset": "performance_by_algorithm",
            "sourceId": SOURCE_ID,
            "encodings": {
                "x": {"field": "algorithm", "type": "nominal", "label": "Scoring function"},
                "y": {"field": "solve_rate", "type": "quantitative", "format": "percent", "label": "Solve rate"},
                "color": {"field": "performance_measure", "type": "nominal", "label": "Summary"},
                "tooltip": [
                    {"field": "n_runs", "format": "number", "label": "Runs"},
                    {"field": "peak_to_final_drop_mean", "format": "percent", "label": "Mean peak-to-final drop"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
            "palette": {"kind": "semantic", "name": "actual-vs-comparison"},
            "legend": {"position": "bottom", "sort": "spec", "title": "Summary"},
            "settings": {"groupMode": "grouped", "sort": "none"},
        },
        {
            "id": "diversity_relationship",
            "title": "Effective replay size and journey solve rate",
            "subtitle": "One point per primary run; 21 runs across five scoring families",
            "showDescription": True,
            "intent": "relationship",
            "question": "Do runs with less concentrated replay exposure achieve higher solve rates?",
            "rationale": "A run-level scatter respects seed as the replication unit and shows clusters, counterexamples, and algorithm context.",
            "comparisonContext": {
                "grain": "one completed or near-complete training run",
                "denominator": "32 sampled replay slots and fixed evaluation suite",
            },
            "type": "scatter",
            "dataset": "run_relationship",
            "sourceId": SOURCE_ID,
            "encodings": {
                "x": {"field": "journey_simpson_effective_size", "type": "quantitative", "label": "Effective replay size"},
                "y": {"field": "journey_mean_solve_rate", "type": "quantitative", "format": "percent", "label": "Journey solve rate"},
                "color": {"field": "algorithm", "type": "nominal", "label": "Scoring function"},
                "label": {"field": "run_label", "type": "text", "label": "Run"},
                "tooltip": [
                    {"field": "final_solve_rate", "format": "percent", "label": "Final solve rate"},
                    {"field": "journey_mean_shortest_path", "format": "number", "label": "Shortest path"},
                    {"field": "journey_dominant_share", "format": "percent", "label": "Dominant share"},
                ],
            },
            "layout": "full",
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "sort": "labelAsc", "title": "Scoring function"},
        },
        {
            "id": "log_seed_trajectory",
            "title": "Log-relative solve rate for seeds 3-5",
            "subtitle": "The best final run, intermediate run, and 0% final run at 1k-update intervals; all 150 evaluations support the analysis",
            "showDescription": True,
            "intent": "trend",
            "question": "When do the log-relative seeds diverge, peak, and lose performance?",
            "rationale": "A three-series line chart preserves the full learning journey and makes the late seed-5 collapse visible without aggregating it away.",
            "comparisonContext": {
                "grain": "200-update evaluation step",
                "denominator": "fixed eight-level evaluation suite",
                "unit": "solve rate",
            },
            "type": "line",
            "dataset": "log_seed_trajectory",
            "sourceId": SOURCE_ID,
            "encodings": {
                "x": {"field": "num_updates", "type": "quantitative", "label": "Training updates"},
                "y": {"field": "solve_rate", "type": "quantitative", "format": "percent", "label": "Solve rate"},
                "color": {"field": "seed_label", "type": "nominal", "label": "Run"},
                "lineStyle": {"field": "line_style", "type": "nominal", "label": "Line style"},
                "tooltip": [
                    {"field": "simpson_effective_size", "format": "number", "label": "Effective replay size"},
                    {"field": "dominant_share", "format": "percent", "label": "Dominant share"},
                    {"field": "score_priority_lift_q80_normalized", "format": "number", "label": "Normalized priority lift"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "sort": "spec", "title": "Run"},
            "labels": {"values": "endpoints"},
            "settings": {"showPoints": "never"},
        },
    ]

    tables = [
        {
            "id": "matched_table",
            "title": "Matched MaxMC and log-relative seeds",
            "subtitle": "Log-relative minus MaxMC deltas; same seed and protocol controls",
            "showDescription": True,
            "dataset": "matched_comparison",
            "defaultSort": {"field": "seed", "direction": "asc"},
            "density": "spacious",
            "sourceId": SOURCE_ID,
            "layout": "full",
            "columns": [
                {"field": "seed", "label": "Seed", "format": "number"},
                {"field": "maxmc_final_solve_rate", "label": "MaxMC final", "format": "percent"},
                {"field": "log_relative_final_solve_rate", "label": "Log-relative final", "format": "percent"},
                {"field": "delta_final_solve_rate", "label": "Final delta", "format": "percent", "movement": True},
                {"field": "delta_journey_mean_solve_rate", "label": "Journey delta", "format": "percent", "movement": True},
                {"field": "delta_journey_exact_unique_ratio", "label": "Unique-slot delta", "format": "percent", "movement": True},
                {"field": "delta_journey_simpson_effective_size", "label": "Effective-size delta", "format": "number", "movement": True},
                {"field": "delta_journey_mean_shortest_path", "label": "Path-length delta", "format": "number", "movement": True},
            ],
        },
        {
            "id": "seed_stage_table",
            "title": "Seeds 3-5 by training stage",
            "subtitle": "Stage averages separate evaluation outcome, replay concentration, priority pressure, and structural path difficulty",
            "showDescription": True,
            "dataset": "seed_stage",
            "defaultSort": {"field": "seed", "direction": "asc"},
            "density": "dense",
            "sourceId": SOURCE_ID,
            "layout": "full",
            "columns": [
                {"field": "seed", "label": "Seed", "format": "number"},
                {"field": "stage", "label": "Stage", "type": "text"},
                {"field": "solve_rate", "label": "Solve rate", "format": "percent"},
                {"field": "simpson_effective_size", "label": "Effective size", "format": "number"},
                {"field": "dominant_share", "label": "Dominant share", "format": "percent"},
                {"field": "mean_shortest_path", "label": "Shortest path", "format": "number"},
                {"field": "mean_solvable", "label": "Solvable exposure", "format": "percent"},
                {"field": "score_priority_lift_q80_normalized", "label": "Priority lift / q80", "format": "number"},
                {"field": "agent_entropy", "label": "Policy entropy", "format": "number"},
            ],
        },
        {
            "id": "unsolvable_events",
            "title": "Dominant unsolvable-maze events",
            "subtitle": "Structural snapshots where fewer than 90% of the 32 sampled slots were solvable",
            "showDescription": True,
            "dataset": "unsolvable_events",
            "defaultSort": {"field": "mean_solvable", "direction": "asc"},
            "density": "spacious",
            "sourceId": SOURCE_ID,
            "layout": "full",
            "columns": [
                {"field": "algorithm", "label": "Scoring function", "type": "text"},
                {"field": "seed", "label": "Seed", "format": "number"},
                {"field": "num_updates", "label": "Update", "format": "number"},
                {"field": "mean_solvable", "label": "Solvable slots", "format": "percent"},
                {"field": "dominant_share", "label": "Dominant share", "format": "percent"},
                {"field": "solve_rate", "label": "Solve rate", "format": "percent"},
                {"field": "solve_rate_plus_1000", "label": "Solve rate +1k", "format": "percent", "movement": True},
                {"field": "simpson_effective_size", "label": "Effective size", "format": "number"},
            ],
        },
        {
            "id": "correlation_table",
            "title": "Run-level diagnostic correlations",
            "subtitle": "Median within-run Spearman correlation across seeds; time points are diagnostic observations, not independent replicates",
            "showDescription": True,
            "dataset": "correlation_summary",
            "defaultSort": {"field": "median_rho", "direction": "desc"},
            "density": "dense",
            "sourceId": SOURCE_ID,
            "layout": "full",
            "columns": [
                {"field": "metric_label", "label": "Replay metric", "type": "text"},
                {"field": "outcome_label", "label": "Outcome", "type": "text"},
                {"field": "median_rho", "label": "Median rho", "format": "number"},
                {"field": "q25", "label": "25th percentile", "format": "number"},
                {"field": "q75", "label": "75th percentile", "format": "number"},
                {"field": "n_runs", "label": "Runs", "format": "number"},
            ],
        },
        {
            "id": "metric_dictionary",
            "title": "Metric definitions",
            "subtitle": "Definitions distinguish sampled exposure, topology, structural difficulty, score pressure, and evaluation outcome",
            "showDescription": True,
            "dataset": "metric_dictionary",
            "defaultSort": {"field": "metric", "direction": "asc"},
            "density": "spacious",
            "sourceId": SOURCE_ID,
            "layout": "full",
            "columns": [
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "definition", "label": "Definition", "type": "text"},
            ],
        },
        {
            "id": "quality_table",
            "title": "Evidence coverage by scoring function",
            "subtitle": "One near-complete run ends at 29.6k; every available replay image decoded successfully",
            "showDescription": True,
            "dataset": "quality_by_algorithm",
            "defaultSort": {"field": "replay_image_rows", "direction": "desc"},
            "density": "spacious",
            "sourceId": SOURCE_ID,
            "layout": "full",
            "columns": [
                {"field": "algorithm", "label": "Scoring function", "type": "text"},
                {"field": "runs", "label": "Runs", "format": "number"},
                {"field": "complete_30k", "label": "Complete 30k", "format": "number"},
                {"field": "history_rows", "label": "Evaluation rows", "format": "number"},
                {"field": "replay_image_rows", "label": "Replay batches", "format": "number"},
                {"field": "structural_snapshots", "label": "Structural snapshots", "format": "number"},
                {"field": "decode_failures", "label": "Decode failures", "format": "number"},
                {"field": "protocol_matches", "label": "Protocol matches", "format": "number"},
            ],
        },
        {
            "id": "editor_sensitivity",
            "title": "Raw editor-transfer cohort sensitivity",
            "subtitle": "Two separately launched three-seed cohorts with matched visible controls; treated as sensitivity evidence, not pooled replication",
            "showDescription": True,
            "dataset": "editor_sensitivity",
            "defaultSort": {"field": "cohort", "direction": "asc"},
            "density": "dense",
            "sourceId": SOURCE_ID,
            "layout": "full",
            "columns": [
                {"field": "cohort", "label": "Cohort", "type": "text"},
                {"field": "seed", "label": "Seed", "format": "number"},
                {"field": "final_solve_rate", "label": "Final solve", "format": "percent"},
                {"field": "journey_mean_solve_rate", "label": "Journey solve", "format": "percent"},
                {"field": "journey_exact_unique_ratio", "label": "Unique slots", "format": "percent"},
                {"field": "journey_simpson_effective_size", "label": "Effective size", "format": "number"},
                {"field": "peak_to_final_drop", "label": "Peak-to-final drop", "format": "percent"},
            ],
        },
    ]

    technical_summary = f"""## Technical summary

**Replay exposure was concentrated by construction, not because W&B shows a duplicate-filled bank.** Across {run_count} primary runs and {replay_slots:,} sampled replay slots, the average 32-slot batch contained **{32 * exact_unique_ratio:.2f} exact maze identities**, had an inverse-Simpson effective size of **{effective_size:.2f}**, and devoted **{dominant_share:.1%}** of slots to its most-sampled maze. The configured rank temperature and staleness mixture predict **{theoretical_share:.1%}** from the score component alone—almost exactly the observed concentration.

**The score function changed structural difficulty more than diversity.** Journey-average solvable shortest paths range from roughly **8.1 to 10.9 grid steps** across score families, while exact unique-slot share stays within a range below one percentage point and nearest-topology wall distance remains near **0.188**.

**Maze similarity does not explain the log-relative seed divergence.** Matched across seeds 0-5, log-relative transfer trails MaxMC by **{abs(final_delta):.1%}** at the final checkpoint on average, but the paired 95% interval includes zero. Seed 3 reaches **51.25%** final solve rate while seed 5 finishes at **0%**, despite nearly indistinguishable replay concentration and topology metrics. This is descriptive evidence against a general structural-diversity-collapse explanation, not proof of an alternative cause.

**A narrower failure mode is real:** two structural snapshots were dominated by a single unsolvable maze, consuming **75-78%** of replay slots. One occurred in the best log-relative seed and recovered; the other preceded the near-complete absolute-advantage run's crash. The event is operationally wasteful but not sufficient as a universal crash explanation."""

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}", "layout": "full"},
        {"id": "technical_summary", "type": "markdown", "body": technical_summary, "layout": "full", "sourceId": SOURCE_ID},
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards], "layout": "full"},
        {
            "id": "concentration_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": """## Rank sampling—not bank duplication—collapses the effective batch

The sampler draws each of the 32 replay slots independently with replacement. At temperature `0.3`, pure rank weighting assigns about 87.2% probability to rank one; mixing 70% score weight with 30% staleness yields a 61.0% score-component share, or 19.5 expected copies in a 32-slot batch. The observed dominant share is 61.1% across score families. The chart shows that the effective exposure count stays near 2.7 rather than progressively collapsing late in training.

**Implication:** duplicate replay is an efficiency and gradient-correlation problem shared by every scoring function here. It is not evidence that the underlying 4,000-entry bank itself contains exact duplicates.""",
        },
        {"id": "effective_size_chart", "type": "chart", "chartId": "effective_size_trend", "layout": "full"},
        {
            "id": "difficulty_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": """## Score functions select different path difficulty, not different topology diversity

Mean absolute advantage prioritizes the longest solvable paths, followed by MaxMC. Both editor-transfer variants and log-relative transfer select shorter paths, while mean positive delight selects the shortest. This is a structural proxy—not policy-conditioned level success—but the 2.7-step spread is materially larger than the between-score differences in exact or nearest-neighbour diversity.""",
        },
        {"id": "difficulty_chart", "type": "chart", "chartId": "difficulty_algorithm", "layout": "full"},
        {
            "id": "topology_explanation",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": """### Distinct wall maps do not converge toward one another late

After removing exact repeats and aligning rotations/reflections, the average nearest distinct topology remains approximately 18.8% of wall cells away across early, middle, and late training. The absence of a late drop is the key negative result: sampled morphology remains stable even when evaluation performance falls.""",
        },
        {"id": "topology_chart", "type": "chart", "chartId": "topology_stage", "layout": "full"},
        {
            "id": "performance_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": """## Performance varies far more than sampled diversity

MaxMC has the highest mean journey and final solve rate in this snapshot. Log-relative transfer is the most seed-sensitive family: its final solve rates span 0% to 51.25%. Because the algorithm-level exposure metrics occupy a narrow band, the run-level scatter tests the more appropriate question—whether the small remaining diversity differences align with learning outcomes.""",
        },
        {"id": "performance_chart", "type": "chart", "chartId": "performance_algorithm", "layout": "full"},
        {
            "id": "relationship_explanation",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": """### Effective batch size has essentially no diagnostic relationship with solve rate

Across 21 runs, effective replay size clusters tightly around 2.7 while journey solve rate spans roughly 9% to 45%. Median within-run Spearman correlations for unique ratio, dominant share, effective size, and pairwise topology distance are all close to zero both contemporaneously and at a 1,000-update lead. The evidence supports monitoring concentration as wasted exposure, but not naming it as the observed cause of the crashes.""",
        },
        {"id": "relationship_chart", "type": "chart", "chartId": "diversity_relationship", "layout": "full"},
        {
            "id": "matched_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": """## Matched seeds do not support a superiority claim for log-relative transfer

Log-relative transfer beats MaxMC at the final checkpoint in seeds 1-3 and loses in seeds 0, 4, and 5. The paired mean final difference is -10.2 percentage points with a 95% interval from about -37.6 to +17.2 points (`p=0.38`, six seeds). Journey-average performance is 12.2 points lower and narrowly misses a conventional paired threshold (`p=0.053`). These results are descriptive and underpowered; they do not establish algorithmic inferiority or superiority.""",
        },
        {"id": "matched_table_block", "type": "table", "tableId": "matched_table", "layout": "full"},
        {
            "id": "seed_crash_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": """## Seeds 3 and 5 diverge without a diversity-collapse signature

Seed 3's evaluation curve improves into late training and finishes at 51.25%; seed 5 peaks at 38.75% around update 20.8k and finishes at 0%. Yet seed 5's late effective replay size and exact unique share are slightly higher—not lower—than near its peak, and its distinct-topology distance does not collapse. Seed 4 provides an intermediate low-performing trajectory.""",
        },
        {"id": "seed_trajectory_chart", "type": "chart", "chartId": "log_seed_trajectory", "layout": "full"},
        {
            "id": "seed_stage_explanation",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": """### Stage averages separate learner outcome from what the buffer sampled

The table shows that concentration, normalized priority pressure, and structural path length remain broadly comparable while solve rates separate. Policy entropy is included as learner-state context; it should not be read as a causal mediator without matched checkpoint interventions.""",
        },
        {"id": "seed_stage_table_block", "type": "table", "tableId": "seed_stage_table", "layout": "full"},
        {
            "id": "unsolvable_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": """## Concentration becomes harmful when the top-ranked maze is invalid

Two replay snapshots had fewer than 90% solvable exposures. In each, the dominant exact maze was unsolvable and occupied almost the entire batch. The event in log-relative seed 3 did not prevent recovery; the absolute-advantage event was followed by a solve-rate decline and a run crash at 29.6k. This motivates a solvability gate and sampler-level exposure cap, but two events cannot establish a general causal mechanism.""",
        },
        {"id": "unsolvable_table_block", "type": "table", "tableId": "unsolvable_events", "layout": "full"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "layout": "full",
            "body": """## Scope, data, and metric definitions

The primary cohort contains 21 protocol-matched runs across five scoring families. Twenty reach 30k updates; mean-absolute-advantage seed 2 ends at 29.6k and remains included as near-complete, visibly caveated evidence. A separate three-run raw editor-transfer cohort is used only for sensitivity.

The unit observed from W&B is the **last sampled replay batch**, not the entire bank. Exact rendered identity includes walls, goal, agent position, and agent direction. Wall-topology metrics ignore agent/goal and canonicalize the eight rotation/reflection transforms. Difficulty is decomposed into transparent structural proxies rather than an opaque composite score.""",
        },
        {"id": "metric_dictionary_block", "type": "table", "tableId": "metric_dictionary", "layout": "full"},
        {
            "id": "methodology",
            "type": "markdown",
            "layout": "full",
            "body": """## Methodology

1. Discover all runs in the two authorized W&B projects and select named score-function cohorts with matched visible controls.
2. Retrieve every evaluation history row from updates 200-30k and validate the 32-file replay image payload.
3. Measure exact identity concentration at every 200-update evaluation.
4. Every 1,000 updates, decode the 15×15 rendered grid, remove the fixed border, locate agent and goal, and compute BFS reachability, shortest path, graph-degree features, and D4-invariant wall distances.
5. Aggregate time series within run first and use seed as the replication unit. Use matched seed deltas for MaxMC versus log-relative transfer.
6. Treat correlations as diagnostic descriptions only; no temporal association is promoted to causality.""",
        },
        {
            "id": "correlation_explanation",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": """### Correlation robustness check

The median correlations below remain near zero at both current and +1k-update outcomes. Variation across runs is retained through the interquartile range. This test directly challenges the hypothesis that replay similarity reliably precedes evaluation deterioration.""",
        },
        {"id": "correlation_table_block", "type": "table", "tableId": "correlation_table", "layout": "full"},
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": """## Limitations, uncertainty, and robustness

- W&B images cannot measure exact or near-duplicate prevalence across all 4,000 bank entries; checkpoint tensors are required for a bank census.
- Structural path length is not policy-conditioned difficulty. The missing gold-standard metric is replay-level pre-update success or return under the current learner.
- Raw PLR score magnitudes are not comparable across score functions because they encode different objects. Normalized priority lift is available only for newer runs with bank quantile logs.
- Six matched seeds leave wide uncertainty on final performance differences. Repeated evaluations within a seed are not independent replicates.
- The two raw editor-transfer cohorts have matched visible controls but different launch dates/code snapshots; their 11.25-point mean final difference is sensitivity evidence, not clean replication.
- The site is a fixed W&B snapshot, not a live dashboard.""",
        },
        {"id": "editor_sensitivity_block", "type": "table", "tableId": "editor_sensitivity", "layout": "full"},
        {
            "id": "quality_explanation",
            "type": "markdown",
            "layout": "full",
            "sourceId": SOURCE_ID,
            "body": f"""### Evidence coverage

The primary analysis covers {image_batches:,} logged replay batches ({replay_slots:,} slots) and {structural_batches:,} structural snapshots ({structural_slots:,} slots), with zero structural decode failures. The sole incomplete run contributes 148 of 150 expected evaluations.""",
        },
        {"id": "quality_table_block", "type": "table", "tableId": "quality_table", "layout": "full"},
        {
            "id": "recommendations",
            "type": "markdown",
            "layout": "full",
            "body": """## Recommended next steps

1. **Cap within-batch multiplicity or sample without replacement.** Preserve rank prioritization while preventing one level from consuming most of a PPO batch; compare equal-environment-step counterfactuals.
2. **Reject unsolvable levels before insertion and after mutation.** Structural formatting alone is insufficient; run reachability from agent to goal.
3. **Log replay-level pre-update return and sampled indices.** This supplies policy-conditioned difficulty and removes reliance on image hashes for exact exposure accounting.
4. **Audit checkpoint banks at early, middle, peak, and post-crash states.** Measure exact bank uniqueness, D4 clusters, score-stratified morphology, and sampling effective size separately.
5. **Run a matched ablation.** Cross scoring function `{MaxMC, log-relative}` with sampler `{current replacement, capped/without replacement}` using shared seeds and a frozen audit suite. This is the experiment that can tell whether concentration harms transfer-powered prioritization.""",
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "layout": "full",
            "body": """## Further questions

- Does full-bank topology remain diverse while rank mass concentrates on a small score frontier?
- Are high log-relative scores stable under rescoring, or does noise repeatedly rotate the dominant maze?
- Does seed 5's policy state become less plastic before the solve-rate collapse even though sampled topology remains stable?
- How much performance is recovered by changing only the sampling distribution while holding score computation fixed?""",
        },
    ]

    raw_datasets: dict[str, pd.DataFrame | list[dict[str, Any]]] = {
        "headline": headline,
        "concentration_timeseries": concentration,
        "difficulty_by_algorithm": difficulty,
        "performance_by_algorithm": performance_rows,
        "topology_stage": topology_stage,
        "run_relationship": relationship,
        "log_seed_trajectory": log_seed_trajectory,
        "seed_stage": seed_stage,
        "matched_comparison": matched_display,
        "paired_stats": paired_stats,
        "unsolvable_events": anomalies_display,
        "correlation_summary": correlation_display,
        "metric_dictionary": metric_rows,
        "quality_by_algorithm": quality_by_algorithm,
        "editor_sensitivity": editor_sensitivity,
    }
    materialized_datasets, dataset_sources = materialize_widget_datasets(raw_datasets)
    for item in [*cards, *charts, *tables]:
        item["sourceId"] = f"source_{item['dataset']}"

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "Technical analysis of replay diversity, structural difficulty, sampler concentration, and learning outcomes across PLR scoring functions.",
            "generatedAt": analysis["generated_at"],
            "sources": [source, *dataset_sources],
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": analysis["generated_at"],
            "status": "ready",
            "datasets": materialized_datasets,
        },
        "sources": [source, *dataset_sources],
        "package_info": {
            "analysis_id": "jaxued-plr-priority-2026-08-26",
            "analysis_script": "analysis/plr_priority_report/analyze_wandb.py",
            "artifact_builder": "analysis/plr_priority_report/build_artifact.py",
        },
    }
    output = ROOT / "artifact.json"
    output.write_text(
        json.dumps(clean(artifact), allow_nan=False, separators=(",", ":")) + "\n"
    )
    print(
        json.dumps(
            {
                "artifact": str(output),
                "datasets": len(artifact["snapshot"]["datasets"]),
                "rows": {
                    key: len(value)
                    for key, value in artifact["snapshot"]["datasets"].items()
                },
                "bytes": output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
