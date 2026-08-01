"""Flax neural-network layers used by JaxUED examples."""

from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


Carry = Any
Output = Any


class ResetRNN(nn.Module):
    """Apply an RNN while resetting carry at episode boundaries.

    The wrapped cell is scanned over the leading time dimension. Before each
    cell application, batch elements whose reset flag is true receive
    ``reset_carry`` instead of their previous hidden state. This matches
    vectorized RL rollouts in which different environments terminate at
    different timesteps.

    Attributes:
        cell: Flax recurrent cell to scan over the input sequence.
    """

    cell: nn.RNNCellBase

    @nn.compact
    def __call__(
        self,
        inputs: Tuple[jax.Array, jax.Array],
        *,
        initial_carry: Optional[Carry] = None,
        reset_carry: Optional[Carry] = None,
    ) -> Tuple[Carry, Output]:
        """Run the recurrent cell over inputs and reset selected batch carries.

        Args:
            inputs: Tuple ``(observations, resets)``. Observations have leading
                dimensions ``(time, batch, ...)`` and resets have shape
                ``(time, batch)``.
            initial_carry: Carry used before the first timestep. When omitted,
                ``reset_carry`` is used.
            reset_carry: Carry substituted for batch elements at reset
                timesteps. When omitted, it is initialized by the wrapped cell.

        Returns:
            A tuple containing the final carry and all per-timestep cell
            outputs.
        """
        # On episode completion, model resets to this
        if reset_carry is None:
            reset_carry = self.cell.initialize_carry(
                jax.random.PRNGKey(0), inputs[0].shape[1:]
            )
        carry = initial_carry if initial_carry is not None else reset_carry

        def scan_fn(cell, carry, inputs):
            x, resets = inputs
            carry = jax.tree_util.tree_map(
                lambda a, b: jnp.where(resets[:, None], a, b), reset_carry, carry
            )
            return cell(carry, x)

        scan = nn.scan(
            scan_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )

        return scan(self.cell, carry, inputs)
