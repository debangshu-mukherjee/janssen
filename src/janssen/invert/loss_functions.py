"""Loss function implementations for ptychography optimization.

Extended Summary
----------------
This module provides loss functions for comparing model outputs
with experimental data in ptychography applications. All functions
are JAX-compatible and support automatic differentiation for
optimization.

Routine Listings
----------------
create_loss_function : function
    Creates a JIT-compatible loss function for comparing model output
    with experimental data.
mae_loss : function
    Mean Absolute Error loss function (internal)
mse_loss : function
    Mean Squared Error loss function (internal)
rmse_loss : function
    Root Mean Squared Error loss function (internal)

Notes
-----
All loss functions are designed to work with JAX transformations
including
jit, grad, and vmap. The create_loss_function factory returns a
JIT-compiled
function that can be used with various optimization algorithms.
"""

import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import Array, Float, PyTree


def create_loss_function(
    forward_function: Callable[..., Array],
    experimental_data: Array,
    loss_type: str = "mae",
) -> Callable[..., Float[Array, " "]]:
    """
    Create a JIT-compatible loss function.

    This function returns a new function that computes the loss between
    the output of a forward model and experimental data. The returned
    function is JIT-compatible and can be used with various optimization
    algorithms.

    Parameters
    ----------
    forward_function : Callable[..., Array]
        The forward model function (e.g., stem_4d).
    experimental_data : Array
        The experimental data to compare against.
    loss_type : str, optional
        The type of loss to use. Options are "mae" (Mean Absolute
        Error),
        "mse" (Mean Squared Error), or "rmse" (Root Mean Squared Error),
        by default "mae".

    Returns
    -------
    loss_fn : Callable[[PyTree, ...], Float[Array, " "]]
        A JIT-compatible function that computes the loss given the model
        parameters and any additional arguments required by the forward
        function.

    Notes
    -----
    - Define internal loss functions (mae_loss, mse_loss, rmse_loss).
    - Select the appropriate loss function based on loss_type.
    - Create a JIT-compiled function that:
        - Computes the forward model output.
        - Calculates the difference between model and experimental data.
        - Applies the selected loss function.
    - Return the compiled loss function.
    """

    def mae_loss(
        model_output: Float[Array, "..."], data: Float[Array, "..."]
    ) -> Float[Array, " "]:
        return jnp.mean(jnp.abs(model_output - data))

    def mse_loss(
        model_output: Float[Array, "..."], data: Float[Array, "..."]
    ) -> Float[Array, " "]:
        return jnp.mean(jnp.square(model_output - data))

    def rmse_loss(
        model_output: Float[Array, "..."], data: Float[Array, "..."]
    ) -> Float[Array, " "]:
        return jnp.sqrt(jnp.mean(jnp.square(model_output - data)))

    def poisson_loss(
        model_output: Float[Array, "..."], data: Float[Array, "..."]
    ) -> Float[Array, " "]:
        """Poisson negative log-likelihood loss.

        For Poisson-distributed data, the negative log-likelihood is:
        L = sum(model - data * log(model))
        We add a small epsilon to avoid log(0).
        """
        eps = 1e-10
        model_safe = jnp.maximum(model_output, eps)
        return jnp.mean(model_safe - data * jnp.log(model_safe))

    loss_functions = {
        "mae": mae_loss,
        "mse": mse_loss,
        "rmse": rmse_loss,
        "poisson": poisson_loss,
    }

    selected_loss_fn = loss_functions[loss_type]

    @jax.jit
    def loss_fn(params: PyTree, *args: Any) -> Float[Array, " "]:
        model_output = forward_function(params, *args)
        return selected_loss_fn(model_output, experimental_data)

    return loss_fn
