import importlib.util

import jax
import pytest


def _assert_compatible_leaf(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype


def test_craftax_module_imports_without_optional_dependency() -> None:
    from jaxued.environments.craftax import (
        CraftaxLogWrapper,
        CraftaxUEDWrapper,
        make_craftax_symbolic,
        make_craftax_symbolic_mutator,
    )

    assert CraftaxLogWrapper is not None
    assert CraftaxUEDWrapper is not None
    assert callable(make_craftax_symbolic)
    assert callable(make_craftax_symbolic_mutator)


def test_make_craftax_symbolic_missing_dependency_message() -> None:
    if importlib.util.find_spec("craftax") is not None:
        pytest.skip("Craftax is installed.")

    from jaxued.environments.craftax import make_craftax_symbolic

    with pytest.raises(ImportError, match="uv sync --extra craftax"):
        make_craftax_symbolic()


def test_make_craftax_symbolic_smoke() -> None:
    pytest.importorskip("craftax")

    from jaxued.environments.craftax import make_craftax_symbolic

    bundle = make_craftax_symbolic(accel_mutation="swap")
    level = bundle.sample_level(jax.random.PRNGKey(0))
    obs, state = bundle.env.reset_to_level(
        jax.random.PRNGKey(1), level, bundle.env_params
    )
    action = bundle.env.action_space(bundle.env_params).sample(jax.random.PRNGKey(2))
    obs2, state2, reward, done, info = bundle.env.step(
        jax.random.PRNGKey(3),
        state,
        action,
        bundle.env_params,
    )

    assert obs is not None
    assert obs2 is not None
    assert state2 is not None
    assert reward.shape == ()
    assert done.shape == ()
    assert "returned_episode_returns" in info


@pytest.mark.parametrize("mutation", ["swap", "swap_restricted", "noise"])
def test_make_craftax_symbolic_mutators(mutation: str) -> None:
    pytest.importorskip("craftax")

    from jaxued.environments.craftax import make_craftax_symbolic

    bundle = make_craftax_symbolic(accel_mutation=mutation)
    level = bundle.sample_level(jax.random.PRNGKey(0))
    child = bundle.mutate_level(jax.random.PRNGKey(1), level, 1)

    jax.tree_util.tree_map(_assert_compatible_leaf, level, child)
