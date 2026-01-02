"""Bessel functions for JAX.

Extended Summary
----------------
Differentiable Bessel functions written in JAx for use throughout janssen.

Routine Listings
----------------
bessel_j0 : function
    Compute J_0(x), regular Bessel function of the first kind, order 0.
bessel_jn : function
    Compute J_n(x), regular Bessel function of the first kind, order n.
bessel_iv_series : function
    Compute I_v(x) using series expansion for Bessel function.
bessel_k0_series : function
    Compute K_0(x) using series expansion.
bessel_kn_recurrence : function
    Compute K_n(x) using recurrence relation.
bessel_kv_small_non_integer : function
    Compute K_v(x) for small x and non-integer v.
bessel_kv_small_integer : function
    Compute K_v(x) for small x and integer v.
bessel_kv : function
    Compute K_v(x), modified Bessel function of the second kind.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from janssen.utils import ScalarFloat, ScalarInteger


@jax.jit
@jaxtyped(typechecker=beartype)
def bessel_j0(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    r"""Compute J_0(x), regular Bessel function of the first kind, order 0.

    Parameters
    ----------
    x : Float[Array, "..."]
        Input array.

    Returns
    -------
    Float[Array, " ..."]
        Values of J_0(x).

    Notes
    -----
    This is a wrapper around JAX's scipy implementation for consistency.
    J_0(x) is the regular Bessel function of the first kind of order 0.

    The function is differentiable, JIT-compatible, and supports broadcasting.

    Examples
    --------
    >>> x = jnp.linspace(0, 10, 100)
    >>> j0_vals = bessel_j0(x)
    """
    # bessel_jn returns (n_orders, ...input_shape), extract first element
    # Handle x=0 edge case where jax.scipy returns NaN but J_0(0) = 1
    result = jax.scipy.special.bessel_jn(x, v=0)[0]
    return jnp.where(x == 0.0, 1.0, result)


@jaxtyped(typechecker=beartype)
def bessel_jn(
    n: ScalarInteger, x: Float[Array, " ..."]
) -> Float[Array, " ..."]:
    r"""Compute J_n(x), regular Bessel function of the first kind, order n.

    Parameters
    ----------
    n : ScalarInteger
        Order of the Bessel function (integer). Must be a compile-time
        constant for JIT compilation.
    x : Float[Array, "..."]
        Input array.

    Returns
    -------
    Float[Array, " ..."]
        Values of J_n(x).

    Notes
    -----
    This is a wrapper around JAX's scipy implementation for consistency.
    J_n(x) is the regular Bessel function of the first kind of order n.

    The function is differentiable and supports broadcasting. Note that
    the order `n` must be a compile-time constant (not a traced value)
    when used inside JIT-compiled functions.

    Examples
    --------
    >>> x = jnp.linspace(0, 10, 100)
    >>> j1_vals = bessel_jn(1, x)
    >>> j2_vals = bessel_jn(2, x)
    """
    # bessel_jn returns shape (n+1, ...input_shape) where [k] is J_k(x)
    # Extract the n-th element for J_n(x)
    # Handle x=0 edge case: J_0(0) = 1, J_n(0) = 0 for n > 0
    result = jax.scipy.special.bessel_jn(x, v=n)[n]
    zero_value = jnp.where(n == 0, 1.0, 0.0)
    return jnp.where(x == 0.0, zero_value, result)


@jaxtyped(typechecker=beartype)
def bessel_iv_series(
    v_order: ScalarFloat, x_val: Float[Array, " ..."], dtype: jnp.dtype
) -> Float[Array, " ..."]:
    """Compute I_v(x) using series expansion for Bessel function."""
    x_half: Float[Array, " ..."] = x_val / 2.0
    x_half_v: Float[Array, " ..."] = jnp.power(x_half, v_order)
    x2_quarter: Float[Array, " ..."] = (x_val * x_val) / 4.0

    max_terms: int = 20
    k_arr: Float[Array, " 20"] = jnp.arange(max_terms, dtype=dtype)

    gamma_v_plus_1: Float[Array, ""] = jax.scipy.special.gamma(v_order + 1)
    gamma_terms: Float[Array, " 20"] = jax.scipy.special.gamma(
        k_arr + v_order + 1
    )
    factorial_terms: Float[Array, " 20"] = jax.scipy.special.factorial(k_arr)

    powers: Float[Array, " ... 20"] = jnp.power(
        x2_quarter[..., jnp.newaxis], k_arr
    )
    series_terms: Float[Array, " ... 20"] = powers / (
        factorial_terms * gamma_terms / gamma_v_plus_1
    )

    result: Float[Array, " ..."] = (
        x_half_v / gamma_v_plus_1 * jnp.sum(series_terms, axis=-1)
    )
    return result


@jaxtyped(typechecker=beartype)
def bessel_k0_series(
    x: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Compute K_0(x) using series expansion."""
    i0: Float[Array, " ..."] = jax.scipy.special.i0(x)
    coeffs: Float[Array, " 7"] = jnp.array(
        [
            -0.57721566,
            0.42278420,
            0.23069756,
            0.03488590,
            0.00262698,
            0.00010750,
            0.00000740,
        ],
        dtype=jnp.float64,
    )
    x2: Float[Array, " ..."] = (x * x) / 4.0
    powers: Float[Array, " ... 7"] = jnp.power(
        x2[..., jnp.newaxis], jnp.arange(7)
    )
    poly: Float[Array, " ..."] = jnp.sum(coeffs * powers, axis=-1)
    log_term: Float[Array, " ..."] = -jnp.log(x / 2.0) * i0
    result: Float[Array, " ..."] = log_term + poly
    return result


@jaxtyped(typechecker=beartype)
def bessel_kn_recurrence(
    n: ScalarInteger,
    x: Float[Array, " ..."],
    k0: Float[Array, " ..."],
    k1: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Compute K_n(x) using recurrence relation."""

    def _compute_kn() -> Float[Array, " ..."]:
        init = (k0, k1)
        max_n = 20
        indices = jnp.arange(1, max_n, dtype=jnp.float32)

        def masked_step(
            carry: Tuple[Float[Array, " ..."], Float[Array, " ..."]],
            i: Float[Array, ""],
        ) -> Tuple[
            Tuple[Float[Array, " ..."], Float[Array, " ..."]],
            Float[Array, " ..."],
        ]:
            k_prev2, k_prev1 = carry
            mask = i < n
            two_i_over_x: Float[Array, " ..."] = 2.0 * i / x
            k_curr: Float[Array, " ..."] = two_i_over_x * k_prev1 + k_prev2
            k_curr = jnp.where(mask, k_curr, k_prev1)
            return (k_prev1, k_curr), k_curr

        carry, k_vals = jax.lax.scan(masked_step, init, indices)
        final_k: Float[Array, " ..."] = carry[1]
        return final_k

    kn_result: Float[Array, " ..."] = jnp.where(
        n == 0, k0, jnp.where(n == 1, k1, _compute_kn())
    )
    return kn_result


@jaxtyped(typechecker=beartype)
def bessel_kv_small_non_integer(
    v: ScalarFloat, x: Float[Array, " ..."], dtype: jnp.dtype
) -> Float[Array, " ..."]:
    """Compute K_v(x) for small x and non-integer v."""
    error_bound: Float[Array, ""] = jnp.asarray(1e-10)
    iv_pos: Float[Array, " ..."] = bessel_iv_series(v, x, dtype)
    iv_neg: Float[Array, " ..."] = bessel_iv_series(-v, x, dtype)
    sin_piv: Float[Array, ""] = jnp.sin(jnp.pi * v)
    pi_over_2sin: Float[Array, ""] = jnp.pi / (2.0 * sin_piv)
    iv_diff: Float[Array, " ..."] = iv_neg - iv_pos
    result: Float[Array, " ..."] = jnp.where(
        jnp.abs(sin_piv) > error_bound, pi_over_2sin * iv_diff, 0.0
    )
    return result


@jaxtyped(typechecker=beartype)
def bessel_kv_small_integer(
    v: Float[Array, ""],
    x: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Compute K_v(x) for small x and integer v."""
    v_int: Float[Array, ""] = jnp.round(v)
    n: Int[Array, ""] = jnp.abs(v_int).astype(jnp.int32)

    k0: Float[Array, " ..."] = bessel_k0_series(x)

    i1: Float[Array, " ..."] = jax.scipy.special.i1(x)
    k1_coeffs: Float[Array, " 5"] = jnp.array(
        [1.0, -0.5, 0.0625, -0.03125, 0.0234375], dtype=jnp.float64
    )
    x2: Float[Array, " ..."] = (x * x) / 4.0
    k1_powers: Float[Array, " ... 5"] = jnp.power(
        x2[..., jnp.newaxis], jnp.arange(5)
    )
    k1_poly: Float[Array, " ..."] = jnp.sum(k1_coeffs * k1_powers, axis=-1)
    log_i1_term: Float[Array, " ..."] = -jnp.log(x / 2.0) * i1
    k1: Float[Array, " ..."] = log_i1_term + k1_poly / x

    kn_result: Float[Array, " ..."] = bessel_kn_recurrence(n, x, k0, k1)
    pos_v_result: Float[Array, " ..."] = jnp.where(
        v >= 0, kn_result, kn_result
    )
    return pos_v_result


def _bessel_kv_large(
    v: ScalarFloat, x: Float[Array, " ..."]
) -> Float[Array, " ..."]:
    """Asymptotic expansion for K_v(x) for large x."""
    sqrt_term: Float[Array, " ..."] = jnp.sqrt(jnp.pi / (2.0 * x))
    exp_term: Float[Array, " ..."] = jnp.exp(-x)

    v2: Float[Array, ""] = v * v
    four_v2: Float[Array, ""] = 4.0 * v2
    a0: Float[Array, ""] = 1.0
    a1: Float[Array, ""] = (four_v2 - 1.0) / 8.0
    a2: Float[Array, ""] = (four_v2 - 1.0) * (four_v2 - 9.0) / (2.0 * 64.0)
    a3: Float[Array, ""] = (
        (four_v2 - 1.0) * (four_v2 - 9.0) * (four_v2 - 25.0) / (6.0 * 512.0)
    )
    a4: Float[Array, ""] = (
        (four_v2 - 1.0)
        * (four_v2 - 9.0)
        * (four_v2 - 25.0)
        * (four_v2 - 49.0)
        / (24.0 * 4096.0)
    )

    z: Float[Array, " ..."] = 1.0 / x
    poly: Float[Array, " ..."] = a0 + z * (a1 + z * (a2 + z * (a3 + z * a4)))

    large_x_result: Float[Array, " ..."] = sqrt_term * exp_term * poly
    return large_x_result


@jaxtyped(typechecker=beartype)
def bessel_k_half(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    """Special case K_{1/2}(x) = sqrt(π/(2x)) * exp(-x)."""
    sqrt_pi_over_2x: Float[Array, " ..."] = jnp.sqrt(jnp.pi / (2.0 * x))
    exp_neg_x: Float[Array, " ..."] = jnp.exp(-x)
    k_half_result: Float[Array, " ..."] = sqrt_pi_over_2x * exp_neg_x
    return k_half_result


@jax.jit
@jaxtyped(typechecker=beartype)
def bessel_kv(v: ScalarFloat, x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    r"""Compute the modified Bessel function of the second kind K_v(x).

    Parameters
    ----------
    v : ScalarFloat
        Order of the Bessel function (v >= 0).
    x : Float[Array, "..."]
        Positive real input array.

    Returns
    -------
    Float[Array, " ..."]
        Approximated values of K_v(x).

    Notes
    -----
    Computes K_v(x) for real order v >= 0 and x > 0, using a numerically stable
    and differentiable JAX-compatible approximation.

    - Valid for v >= 0 and x > 0
    - Supports broadcasting and autodiff
    - JIT-safe and VMAP-safe
    - Uses series expansion for small x (x <= 2.0) and asymptotic expansion
      for large x
    - For non-integer v, uses the reflection formula:
      K_v = π/(2sin(πv)) * (I_{-v} - I_v)
    - For integer v, uses specialized series and recurrence relations
    - Special exact formula for v = 0.5: K_{1/2}(x) = sqrt(π/(2x)) * exp(-x)
    - The transition point between small and large x approximations is
      set at x = 2.0

    Algorithm
    ---------
    - For integer orders n > 1, uses recurrence relations with masked updates
      to only update values within the target range
    """
    v: Float[Array, ""] = jnp.asarray(v)
    x: Float[Array, " ..."] = jnp.asarray(x)
    dtype: jnp.dtype = x.dtype

    v_int: Float[Array, ""] = jnp.round(v)
    epsilon_tolerance: float = 1e-10
    is_integer: Bool[Array, ""] = jnp.abs(v - v_int) < epsilon_tolerance

    small_x_non_int: Float[Array, " ..."] = bessel_kv_small_non_integer(
        v, x, dtype
    )
    small_x_int: Float[Array, " ..."] = bessel_kv_small_integer(v, x)
    small_x_vals: Float[Array, " ..."] = jnp.where(
        is_integer, small_x_int, small_x_non_int
    )

    large_x_vals: Float[Array, " ..."] = _bessel_kv_large(v, x)

    small_x_threshold: float = 2.0
    general_result: Float[Array, " ..."] = jnp.where(
        x <= small_x_threshold, small_x_vals, large_x_vals
    )

    k_half_vals: Float[Array, " ..."] = bessel_k_half(x)
    is_half: Bool[Array, ""] = jnp.abs(v - 0.5) < epsilon_tolerance
    final_result: Float[Array, " ..."] = jnp.where(
        is_half, k_half_vals, general_result
    )

    return final_result
