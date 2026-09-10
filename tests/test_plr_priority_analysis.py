import importlib.util
import math
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "plr_priority_report"
    / "analyze_wandb.py"
)
SPEC = importlib.util.spec_from_file_location("plr_priority_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exact_batch_metrics_report_effective_exposure() -> None:
    filenames = [f"replay_0_{'a' * 20}.png"] * 20 + [
        f"replay_0_{'b' * 20}.png"
    ] * 12
    metrics = MODULE.exact_batch_metrics(filenames)

    assert metrics["exact_unique_count"] == 2
    assert metrics["exact_unique_ratio"] == 2 / 32
    assert metrics["dominant_share"] == 20 / 32
    assert math.isclose(
        metrics["simpson_effective_size"],
        1 / ((20 / 32) ** 2 + (12 / 32) ** 2),
    )


def test_d4_distance_is_rotation_and_reflection_invariant() -> None:
    wall_map = np.zeros((13, 13), dtype=bool)
    wall_map[1, 2] = True
    wall_map[4, 8] = True

    assert MODULE.d4_wall_distance(wall_map, np.rot90(wall_map)) == 0
    assert MODULE.d4_wall_distance(wall_map, np.fliplr(wall_map)) == 0


def test_decode_maze_recovers_walls_positions_and_path_metrics() -> None:
    tiles = np.zeros((15, 15, 8, 8, 3), dtype=np.uint8)
    tiles[0, :, :, :, :] = 100
    tiles[-1, :, :, :, :] = 100
    tiles[:, 0, :, :, :] = 100
    tiles[:, -1, :, :, :] = 100
    tiles[2, 2, :, :, :] = 100
    tiles[1, 1, :, :, :] = np.array([255, 76, 76], dtype=np.uint8)
    tiles[3, 3, :, :, :] = np.array([0, 255, 0], dtype=np.uint8)
    image = tiles.transpose(0, 2, 1, 3, 4).reshape(120, 120, 3)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")

    decoded = MODULE.decode_maze(buffer.getvalue())

    assert decoded.agent == (0, 0)
    assert decoded.goal == (2, 2)
    assert decoded.wall_map.sum() == 1
    assert decoded.metrics["solvable"] == 1
    assert decoded.metrics["shortest_path"] == 4
