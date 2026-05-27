from dataclasses import dataclass
from typing import Any, Callable

import chex
import jax

from jaxued.environments.underspecified_env import EnvParams, Level, UnderspecifiedEnv
from jaxued.wrappers import AutoReplayWrapper

from .mutators import make_craftax_symbolic_mutator
from .wrappers import CraftaxLogWrapper, CraftaxUEDWrapper

SUPPORTED_CRAFTAX_ENVS = {"Craftax-Symbolic-v1"}
CRAFTAX_INSTALL_MESSAGE = (
    "Craftax support is optional. Install it with `uv sync --extra craftax`."
)


@dataclass(frozen=True)
class CraftaxSymbolicBundle:
    env: UnderspecifiedEnv
    eval_env: UnderspecifiedEnv
    env_params: EnvParams
    sample_level: Callable[[chex.PRNGKey], Level]
    mutate_level: Callable[[chex.PRNGKey, Level, int], Level]
    render_level: Callable[[Level], chex.Array]
    base_env: Any
    static_env_params: Any
    generate_world: Callable[..., Level]


@dataclass(frozen=True)
class _CraftaxSymbols:
    make_env_from_name: Callable[..., Any] | None
    env_class: Any
    generate_world: Callable[..., Level]
    render_craftax_pixels: Callable[..., chex.Array]
    block_pixel_size_img: int


def _raise_missing_craftax(exc: ImportError) -> None:
    raise ImportError(CRAFTAX_INSTALL_MESSAGE) from exc


def _load_craftax_symbols() -> _CraftaxSymbols:
    try:
        from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG
        from craftax.craftax.renderer import render_craftax_pixels
        from craftax.craftax.world_gen.world_gen import generate_world
    except ImportError:
        try:
            from craftax.constants import BLOCK_PIXEL_SIZE_IMG
            from craftax.renderer import render_craftax_pixels
            from craftax.world_gen.world_gen import generate_world
        except ImportError as exc:
            _raise_missing_craftax(exc)

    try:
        from craftax.craftax_env import make_craftax_env_from_name

        return _CraftaxSymbols(
            make_env_from_name=make_craftax_env_from_name,
            env_class=None,
            generate_world=generate_world,
            render_craftax_pixels=render_craftax_pixels,
            block_pixel_size_img=BLOCK_PIXEL_SIZE_IMG,
        )
    except ImportError:
        try:
            from craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
        except ImportError as exc:
            _raise_missing_craftax(exc)

        return _CraftaxSymbols(
            make_env_from_name=None,
            env_class=CraftaxSymbolicEnv,
            generate_world=generate_world,
            render_craftax_pixels=render_craftax_pixels,
            block_pixel_size_img=BLOCK_PIXEL_SIZE_IMG,
        )


def _make_base_env(symbols: _CraftaxSymbols, env_name: str) -> Any:
    if env_name not in SUPPORTED_CRAFTAX_ENVS:
        raise ValueError("v1 supports Craftax-Symbolic-v1 only.")

    if symbols.make_env_from_name is not None:
        return symbols.make_env_from_name(env_name, auto_reset=False)

    static_env_params = symbols.env_class.default_static_params()
    return symbols.env_class(static_env_params)


def _get_static_env_params(base_env: Any) -> Any:
    if hasattr(base_env, "static_env_params"):
        return base_env.static_env_params
    if hasattr(base_env, "default_static_params"):
        return base_env.default_static_params()
    raise AttributeError("Could not find Craftax static environment parameters.")


def make_craftax_symbolic_level_generator(
    generate_world: Callable[..., Level],
    env_params: EnvParams,
    static_env_params: Any,
    *,
    randomize_fractal_noise_angles: bool = False,
) -> Callable[[chex.PRNGKey], Level]:
    def sample_level(rng: chex.PRNGKey) -> Level:
        if not randomize_fractal_noise_angles:
            return generate_world(rng, env_params, static_env_params)

        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
        larger_res = (
            static_env_params.map_size[0] // 4,
            static_env_params.map_size[1] // 4,
        )
        small_res = (
            static_env_params.map_size[0] // 16,
            static_env_params.map_size[1] // 16,
        )
        x_res = (
            static_env_params.map_size[0] // 8,
            static_env_params.map_size[1] // 2,
        )
        fractal_noise_angles = (
            jax.random.uniform(rng1, (small_res[0] + 1, small_res[1] + 1)),
            jax.random.uniform(rng2, (small_res[0] + 1, small_res[1] + 1)),
            jax.random.uniform(rng3, (x_res[0] + 1, x_res[1] + 1)),
            jax.random.uniform(rng4, (larger_res[0] + 1, larger_res[1] + 1)),
        )
        params_to_use = env_params.replace(fractal_noise_angles=fractal_noise_angles)
        return generate_world(rng, params_to_use, static_env_params).replace(
            fractal_noise_angles=fractal_noise_angles
        )

    return sample_level


def make_craftax_symbolic_renderer(
    render_craftax_pixels: Callable[..., chex.Array],
    block_pixel_size_img: int,
) -> Callable[[Level], chex.Array]:
    def render_level(level: Level) -> chex.Array:
        return render_craftax_pixels(level, block_pixel_size_img)

    return render_level


def make_craftax_symbolic(
    *,
    env_name: str = "Craftax-Symbolic-v1",
    accel_mutation: str = "swap",
    auto_replay: bool = True,
) -> CraftaxSymbolicBundle:
    symbols = _load_craftax_symbols()
    base_env = _make_base_env(symbols, env_name)
    static_env_params = _get_static_env_params(base_env)

    env: UnderspecifiedEnv = CraftaxLogWrapper(CraftaxUEDWrapper(base_env))
    if auto_replay:
        env = AutoReplayWrapper(env)

    env_params = env.default_params
    sample_level = make_craftax_symbolic_level_generator(
        symbols.generate_world,
        env_params,
        static_env_params,
        randomize_fractal_noise_angles=accel_mutation == "noise",
    )
    mutate_level = make_craftax_symbolic_mutator(
        accel_mutation,
        generate_world=symbols.generate_world,
        static_env_params=static_env_params,
        env_params=env_params,
    )
    render_level = make_craftax_symbolic_renderer(
        symbols.render_craftax_pixels,
        symbols.block_pixel_size_img,
    )

    return CraftaxSymbolicBundle(
        env=env,
        eval_env=env,
        env_params=env_params,
        sample_level=sample_level,
        mutate_level=mutate_level,
        render_level=render_level,
        base_env=base_env,
        static_env_params=static_env_params,
        generate_world=symbols.generate_world,
    )
