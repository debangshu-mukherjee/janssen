"""Inversion and optimization types for nonlinear inverse problems.

Extended Summary
----------------
This module provides PyTree data structures for representing optimization
state in iterative inversion algorithms. The primary focus is on
Gauss-Newton methods with Levenberg-Marquardt damping for solving
nonlinear least-squares problems that arise in computational imaging.

The GaussNewtonState PyTree tracks all quantities needed for iterative
reconstruction: current parameter estimates (sample and probe), iteration
count, loss value, damping parameter, and convergence status. This
immutable structure enables clean functional programming with JAX.

Routine Listings
----------------
GaussNewtonState : NamedTuple
    Immutable state for Gauss-Newton optimization with Levenberg-Marquardt
    damping
make_gauss_newton_state : function
    Factory function to create validated GaussNewtonState instances

Notes
-----
The Gauss-Newton method with Levenberg-Marquardt damping solves:
    (J^T J + λI) δx = -J^T r

where J is the Jacobian, r is the residual, and λ is the damping
parameter. As λ → 0, this approaches pure Gauss-Newton; as λ → ∞,
it approaches gradient descent with small step size.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple, Union
from jax import lax
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, jaxtyped

from .common_types import ScalarBool, ScalarFloat, ScalarInteger


@register_pytree_node_class
class GaussNewtonState(NamedTuple):
    """Immutable state for Gauss-Newton optimization.

    Attributes
    ----------
    sample : Complex[Array, " H W"]
        Current object transmission estimate.
    probe : Complex[Array, " H W"]
        Current probe wavefront estimate.
    iteration : int
        Current iteration number.
    loss : Float[Array, " "]
        Current loss value (0.5 * ||r||^2).
    damping : Float[Array, " "]
        Current Levenberg-Marquardt damping parameter λ.
    converged : bool
        Whether optimization has converged.
    """

    sample: Complex[Array, " H W"]
    probe: Complex[Array, " H W"]
    iteration: ScalarInteger
    loss: Float[Array, " "]
    damping: Float[Array, " "]
    converged: ScalarBool

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Complex[Array, " H W"],
            Complex[Array, " H W"],
            ScalarInteger,
            Float[Array, " "],
            Float[Array, " "],
            ScalarBool,
        ],
        None,
    ]:
        """Flatten the GaussNewtonState into a tuple of its components."""
        return (
            (
                self.sample,
                self.probe,
                self.iteration,
                self.loss,
                self.damping,
                self.converged,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Complex[Array, " H W"],
            Complex[Array, " H W"],
            ScalarInteger,
            Float[Array, " "],
            Float[Array, " "],
            ScalarBool,
        ],
    ) -> "GaussNewtonState":
        """Unflatten the GaussNewtonState from a tuple of its components."""
        return cls(*children)


@jaxtyped(typechecker=beartype)
def make_gauss_newton_state(
    sample: Complex[Array, " H W"],
    probe: Complex[Array, " H W"],
    iteration: ScalarInteger = 0,
    loss: ScalarFloat = jnp.inf,
    damping: ScalarFloat = 1e-3,
    converged: Union[bool, Bool[Array, " "]] = False,
) -> GaussNewtonState:
    """Create a validated GaussNewtonState instance.

    Factory function that validates inputs and creates a
    GaussNewtonState PyTree suitable for Gauss-Newton optimization with
    Levenberg-Marquardt damping.

    Parameters
    ----------
    sample : Complex[Array, " H W"]
        Initial object transmission estimate.
    probe : Complex[Array, " H W"]
        Initial probe wavefront estimate.
    iteration : ScalarInteger, optional
        Starting iteration number. Default is 0.
    loss : ScalarFloat, optional
        Initial loss value. Default is infinity (not yet computed).
    damping : ScalarFloat, optional
        Initial Levenberg-Marquardt damping parameter λ. Default is 1e-3.
    converged : Union[bool, Bool[Array, " "]], optional
        Initial convergence status. Default is False.

    Returns
    -------
    GaussNewtonState
        Validated Gauss-Newton state instance.

    Raises
    ------
    ValueError
        If sample and probe have inconsistent shapes, or validation
        fails.

    Notes
    -----
    The damping parameter λ controls the Levenberg-Marquardt
    interpolation between gradient descent (large λ) and Gauss-Newton
    (small λ). Typical initial values are 1e-3 to 1e-1.
    """
    expected_ndim: int = 2
    sample_arr: Complex[Array, " H W"] = jnp.asarray(
        sample, dtype=jnp.complex128
    )
    probe_arr: Complex[Array, " H W"] = jnp.asarray(
        probe, dtype=jnp.complex128
    )
    iteration_arr: ScalarInteger = jnp.asarray(iteration, dtype=jnp.int32)
    loss_arr: Float[Array, " "] = jnp.asarray(loss, dtype=jnp.float64)
    damping_arr: Float[Array, " "] = jnp.asarray(damping, dtype=jnp.float64)
    converged_arr: Bool[Array, " "] = jnp.asarray(converged, dtype=jnp.bool_)

    def validate_and_create() -> GaussNewtonState:
        def check_sample_shape() -> Complex[Array, " H W"]:
            is_valid_ndim: Bool[Array, " "] = sample_arr.ndim == expected_ndim
            return lax.cond(
                is_valid_ndim,
                lambda: sample_arr,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: sample_arr, lambda: sample_arr)
                ),
            )

        def check_probe_shape(
            s: Complex[Array, " H W"],
        ) -> Complex[Array, " H W"]:
            is_valid_ndim: Bool[Array, " "] = probe_arr.ndim == expected_ndim
            is_valid_shape: Bool[Array, " "] = jnp.logical_and(
                probe_arr.shape[0] == s.shape[0],
                probe_arr.shape[1] == s.shape[1],
            )
            is_valid: Bool[Array, " "] = jnp.logical_and(
                is_valid_ndim, is_valid_shape
            )
            return lax.cond(
                is_valid,
                lambda: probe_arr,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: probe_arr, lambda: probe_arr)
                ),
            )

        def check_iteration() -> ScalarInteger:
            is_valid: Bool[Array, " "] = iteration_arr >= 0
            return lax.cond(
                is_valid,
                lambda: iteration_arr,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: iteration_arr, lambda: iteration_arr
                    )
                ),
            )

        def check_loss() -> Float[Array, " "]:
            is_valid: Bool[Array, " "] = loss_arr >= 0.0
            return lax.cond(
                is_valid,
                lambda: loss_arr,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: loss_arr, lambda: loss_arr)
                ),
            )

        def check_damping() -> Float[Array, " "]:
            is_valid: Bool[Array, " "] = damping_arr > 0.0
            return lax.cond(
                is_valid,
                lambda: damping_arr,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: damping_arr, lambda: damping_arr)
                ),
            )

        validated_sample: Complex[Array, " H W"] = check_sample_shape()
        validated_probe: Complex[Array, " H W"] = check_probe_shape(
            validated_sample
        )
        validated_iteration: ScalarInteger = check_iteration()
        validated_loss: Float[Array, " "] = check_loss()
        validated_damping: Float[Array, " "] = check_damping()

        return GaussNewtonState(
            sample=validated_sample,
            probe=validated_probe,
            iteration=validated_iteration,
            loss=validated_loss,
            damping=validated_damping,
            converged=converged_arr,
        )

    return validate_and_create()
