"""General-purpose Gauss-Newton optimization with JAX autodiff.

Extended Summary
----------------
This module provides Jacobian-free Gauss-Newton optimization for solving
nonlinear least-squares problems:

    min_θ  0.5 * ||r(θ)||^2

where r: R^n → R^m is a residual function. The key insight is that JAX's
autodifferentiation primitives (jvp/vjp) enable computing Jacobian-vector
products without ever materializing the Jacobian matrix, making second-order
optimization tractable for large-scale problems.

The Gauss-Newton method approximates the Hessian as H ≈ J^T J and solves:

    (J^T J + λI) δ = -J^T r

where λ is the Levenberg-Marquardt damping parameter. This module provides
all the linear algebra primitives to solve this system efficiently using
conjugate gradient with automatic damping adaptation.

Routine Listings
----------------
make_jtj_matvec : function
    Create (J^T J + λI) matrix-vector product operator.
compute_jt_residual : function
    Compute residuals and J^T @ r simultaneously.
make_hessian_matvec : function
    Create exact Hessian-vector product operator.
gauss_newton_step : function
    Generic Gauss-Newton step with trust-region damping.
estimate_max_eigenvalue : function
    Estimate largest eigenvalue of J^T J via power iteration.
estimate_jtj_diagonal : function
    Estimate diagonal of J^T J for preconditioning.

Notes
-----
All functions are JAX-compatible (jit/grad/vmap) and follow functional
programming conventions. The optimization state is managed via the
GaussNewtonState PyTree from janssen.types.

This module is domain-agnostic. For specific applications (ptychography,
tomography, etc.), provide an appropriate residual function r(θ).

References
----------
.. [1] Nocedal & Wright, "Numerical Optimization", 2nd ed., Chapter 10
.. [2] Kelley, "Iterative Methods for Optimization", SIAM (1999)
.. [3] Bradbury et al., "JAX: Autograd and XLA", MLSys (2018)
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as sparse_linalg
from beartype import beartype
from beartype.typing import Callable, Optional, Tuple
from jaxtyping import Array, Bool, Complex, Float, jaxtyped

from janssen.types import GaussNewtonState, ScalarInteger

from .math import flatten_params, unflatten_params

TRUST_REGION_EXCELLENT = 0.75
TRUST_REGION_ACCEPTABLE = 0.25
CONVERGENCE_TOL = 1e-8


def make_jtj_matvec(
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    params: Float[Array, " n"],
    damping: Float[Array, " "],
) -> Callable[[Float[Array, " n"]], Float[Array, " n"]]:
    """Create matrix-vector product operator for (J^T J + λI).

    This is the core of Jacobian-free Gauss-Newton optimization. Rather
    than forming the Jacobian matrix J explicitly, we compute products
    with (J^T J) using JAX's autodiff primitives:

        (J^T J) @ v = J^T @ (J @ v)
                    = vjp(jvp(v))

    where jvp is forward-mode autodiff and vjp is reverse-mode.

    Parameters
    ----------
    residual_fn : Callable[[Float[Array, " n"]], Float[Array, " m"]]
        Function mapping parameters to residuals r(θ).
    params : Float[Array, " n"]
        Current parameters (linearization point).
    damping : Float[Array, " "]
        Levenberg-Marquardt damping parameter λ ≥ 0.

    Returns
    -------
    matvec : Callable[[Float[Array, " n"]], Float[Array, " n"]]
        Function computing (J^T J + λI) @ v for any vector v.

    Notes
    -----
    Memory complexity is O(m + n) per matrix-vector product, compared to
    O(m * n) to store J explicitly. For large-scale problems where
    m, n >> 1, this is essential for tractability.

    Examples
    --------
    >>> def residual_fn(x):
    ...     return jnp.array([x[0]**2 - 1, x[1]**2 - 4])
    >>> params = jnp.array([1.5, 2.5])
    >>> matvec = make_jtj_matvec(residual_fn, params, jnp.array(1e-3))
    >>> result = matvec(jnp.array([1.0, 0.0]))
    """
    _: Float[Array, " m"]
    vjp_fn: Callable[[Float[Array, " m"]], Tuple[Float[Array, " n"]]]
    _, vjp_fn = jax.vjp(residual_fn, params)

    def matvec(v: Float[Array, " n"]) -> Float[Array, " n"]:
        _: Float[Array, " m"]
        jv: Float[Array, " m"]
        _, jv = jax.jvp(residual_fn, (params,), (v,))
        jtjv: Float[Array, " n"]
        (jtjv,) = vjp_fn(jv)
        return jtjv + damping * v

    return matvec


@partial(jax.jit, static_argnums=(0,))
def compute_jt_residual(
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    params: Float[Array, " n"],
    weights: Optional[Float[Array, " m"]] = None,
) -> Tuple[Float[Array, " m"], Float[Array, " n"]]:
    """Compute residuals and J^T @ W @ r simultaneously.

    Uses reverse-mode autodiff to compute the gradient of the loss
    L = 0.5 * r^T W r, which equals J^T @ W @ r.

    For unweighted least squares (W = I), this reduces to J^T @ r.

    Parameters
    ----------
    residual_fn : Callable[[Float[Array, " n"]], Float[Array, " m"]]
        Function mapping parameters to residuals.
    params : Float[Array, " n"]
        Current parameters.
    weights : Float[Array, " m"], optional
        Diagonal weight matrix entries for weighted least squares.
        If None (default), assumes unweighted (W = I).

    Returns
    -------
    residuals : Float[Array, " m"]
        Current residual vector r(θ).
    jt_wr : Float[Array, " n"]
        Weighted gradient J^T @ W @ r = ∇(0.5 * r^T W r).

    Notes
    -----
    For diagonal weighting W = diag(w), the weighted gradient is
    J^T @ W @ r where each residual r_i is weighted by w_i.

    Examples
    --------
    >>> def residual_fn(x):
    ...     return x**2 - jnp.array([1.0, 4.0])
    >>> params = jnp.array([1.5, 2.5])
    >>> r, jt_r = compute_jt_residual(residual_fn, params)
    >>> # Weighted version:
    >>> weights = jnp.array([2.0, 0.5])
    >>> r, jt_wr = compute_jt_residual(residual_fn, params, weights)
    """
    residuals: Float[Array, " m"]
    vjp_fn: Callable[[Float[Array, " m"]], Tuple[Float[Array, " n"]]]
    residuals, vjp_fn = jax.vjp(residual_fn, params)
    weighted_residuals: Float[Array, " m"]
    weighted_residuals = residuals if weights is None else weights * residuals
    jt_wr: Float[Array, " n"]
    (jt_wr,) = vjp_fn(weighted_residuals)
    return residuals, jt_wr


def make_hessian_matvec(
    loss_fn: Callable[[Float[Array, " n"]], Float[Array, " "]],
    params: Float[Array, " n"],
) -> Callable[[Float[Array, " n"]], Float[Array, " n"]]:
    """Create exact Hessian-vector product operator.

    For problems where the Gauss-Newton approximation H ≈ J^T J is
    insufficient, this computes products with the exact Hessian:

        H @ v = d/dε [∇L(θ + εv)]|_{ε=0}

    using forward-over-reverse autodiff (jvp through grad).

    Parameters
    ----------
    loss_fn : Callable[[Float[Array, " n"]], Float[Array, " "]]
        Scalar loss function L(θ).
    params : Float[Array, " n"]
        Current parameters (linearization point).

    Returns
    -------
    hvp : Callable[[Float[Array, " n"]], Float[Array, " n"]]
        Function computing H @ v for any vector v.

    Notes
    -----
    The exact Hessian includes second-order derivative information
    beyond J^T J, capturing curvature from the residual structure
    itself. This is important when residuals are highly nonlinear.

    Examples
    --------
    >>> def loss_fn(x):
    ...     return jnp.sum(x**4)
    >>> params = jnp.array([1.0, 2.0])
    >>> hvp = make_hessian_matvec(loss_fn, params)
    >>> result = hvp(jnp.array([1.0, 0.0]))
    """
    grad_fn: Callable[[Float[Array, " n"]], Float[Array, " n"]]
    grad_fn = jax.grad(loss_fn)

    def hvp(v: Float[Array, " n"]) -> Float[Array, " n"]:
        _: Float[Array, " n"]
        hv: Float[Array, " n"]
        _, hv = jax.jvp(grad_fn, (params,), (v,))
        return hv

    return hvp


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
@jaxtyped(typechecker=beartype)
def gauss_newton_step(
    state: GaussNewtonState,
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    cg_maxiter: int = 50,
    cg_tol: float = 1e-5,
    use_preconditioner: bool = False,
) -> GaussNewtonState:
    """Perform Gauss-Newton step with Levenberg-Marquardt damping.

    Solves the trust-region subproblem using conjugate gradient:

        (J^T J + λI) δ = -J^T r

    then updates parameters: θ_{k+1} = θ_k + δ

    The damping λ adapts based on actual vs predicted reduction.

    Parameters
    ----------
    state : GaussNewtonState
        Current optimization state containing sample, probe, iteration,
        loss, damping, and convergence status.
    residual_fn : Callable[[Float[Array, " n"]], Float[Array, " m"]]
        Function mapping flattened parameters to residuals.
    cg_maxiter : int, optional
        Maximum conjugate gradient iterations. Default is 50.
    cg_tol : float, optional
        CG convergence tolerance. Default is 1e-5.
    use_preconditioner : bool, optional
        Whether to use diagonal preconditioning for CG. Preconditioning
        can significantly improve convergence rate for ill-conditioned
        problems. Default is False.

    Returns
    -------
    new_state : GaussNewtonState
        Updated optimization state after the step.

    Notes
    -----
    This function is generic and works with any residual function. The
    parameters are flattened from (sample, probe) to a real vector
    before passing to residual_fn.

    Trust-region adaptation:
    - If ρ > 0.75: very good step, decrease damping (be aggressive)
    - If 0.25 < ρ < 0.75: acceptable step, keep damping
    - If ρ < 0.25: poor step, increase damping (be conservative)

    where ρ = actual_reduction / predicted_reduction.

    When use_preconditioner=True, a diagonal preconditioner is computed
    using Hutchinson's estimator: M = diag(diag(J^T J) + λ)^{-1}.
    This improves CG convergence for ill-conditioned problems.

    Examples
    --------
    >>> state = make_gauss_newton_state(sample, probe)
    >>> new_state = gauss_newton_step(state, my_residual_fn)
    >>> # With preconditioning for better convergence:
    >>> new_state = gauss_newton_step(state, my_residual_fn,
    ...                               use_preconditioner=True)
    """
    sample_shape: Tuple[int, int] = state.sample.shape
    probe_shape: Tuple[int, int] = state.probe.shape
    params: Float[Array, " n"] = flatten_params(state.sample, state.probe)
    residuals: Float[Array, " m"]
    jt_r: Float[Array, " n"]
    residuals, jt_r = compute_jt_residual(residual_fn, params)
    current_loss: Float[Array, " "] = 0.5 * jnp.sum(residuals**2)
    matvec: Callable[[Float[Array, " n"]], Float[Array, " n"]] = (
        make_jtj_matvec(residual_fn, params, state.damping)
    )
    _preconditioner: Optional[
        Callable[[Float[Array, " n"]], Float[Array, " n"]]
    ] = None
    
    if use_preconditioner:
        diag_jtj: Float[Array, " n"] = estimate_jtj_diagonal(
            residual_fn, params, num_samples=10
        )
        diag_with_damping: Float[Array, " n"] = (
            jnp.maximum(diag_jtj, 0.0) + state.damping
        )

        def _preconditioner(v: Float[Array, " n"]) -> Float[Array, " n"]:
            return v / diag_with_damping

    delta: Float[Array, " n"]
    cg_info: None
    delta, cg_info = sparse_linalg.cg(
        matvec,
        -jt_r,
        x0=jnp.zeros_like(params),
        maxiter=cg_maxiter,
        tol=cg_tol,
        M=_preconditioner,
    )
    params_new: Float[Array, " n"] = params + delta
    sample_new: Complex[Array, " Hs Ws"]
    probe_new: Complex[Array, " Hp Wp"]
    sample_new, probe_new = unflatten_params(
        params_new, sample_shape, probe_shape
    )
    residuals_new: Float[Array, " m"] = residual_fn(params_new)
    new_loss: Float[Array, " "] = 0.5 * jnp.sum(residuals_new**2)
    h_delta: Float[Array, " n"] = matvec(delta)
    predicted_reduction: Float[Array, " "] = 0.5 * jnp.dot(delta, h_delta)
    actual_reduction: Float[Array, " "] = current_loss - new_loss
    pred_positive: Bool[Array, " "] = predicted_reduction > 0.0
    rho: Float[Array, " "] = jnp.where(
        pred_positive,
        actual_reduction / (predicted_reduction + 1e-12),
        0.0,
    )
    accept: Bool[Array, " "] = (
        (actual_reduction > 0.0) & pred_positive & (rho > 0.0)
    )
    new_damping: Float[Array, " "] = jax.lax.cond(
        pred_positive,
        lambda: jax.lax.cond(
            rho > TRUST_REGION_EXCELLENT,
            lambda: state.damping * 0.33,
            lambda: jax.lax.cond(
                rho > TRUST_REGION_ACCEPTABLE,
                lambda: state.damping,
                lambda: state.damping * 3.0,
            ),
        ),
        lambda: state.damping * 10.0,
    )
    new_damping = jnp.clip(new_damping, 1e-12, 1e8)
    final_sample: Complex[Array, " hh ww"] = jax.lax.cond(
        accept, lambda: sample_new, lambda: state.sample
    )
    final_probe: Complex[Array, " hh ww"] = jax.lax.cond(
        accept, lambda: probe_new, lambda: state.probe
    )
    final_loss: Float[Array, " "] = jax.lax.cond(
        accept, lambda: new_loss, lambda: current_loss
    )
    rel_improvement: Float[Array, " "] = jnp.abs(current_loss - final_loss) / (
        current_loss + 1e-12
    )
    converged: Bool[Array, " "] = accept & (rel_improvement < CONVERGENCE_TOL)

    return GaussNewtonState(
        sample=final_sample,
        probe=final_probe,
        iteration=state.iteration + 1,
        loss=final_loss,
        damping=new_damping,
        converged=converged,
    )


@partial(jax.jit, static_argnums=(0, 2))
@jaxtyped(typechecker=beartype)
def estimate_max_eigenvalue(
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    params: Float[Array, " n"],
    num_iterations: int = 20,
) -> Float[Array, " "]:
    """Estimate largest eigenvalue of J^T J via power iteration.

    A large eigenvalue indicates potential ill-conditioning, which can
    cause convergence issues. Use this diagnostic to tune damping.

    Parameters
    ----------
    residual_fn : Callable[[Float[Array, " n"]], Float[Array, " m"]]
        Residual function.
    params : Float[Array, " n"]
        Current parameters.
    num_iterations : int, optional
        Number of power iterations. Default is 20.

    Returns
    -------
    lambda_max : Float[Array, " "]
        Estimate of the largest eigenvalue of J^T J.

    Notes
    -----
    Power iteration computes: v_{k+1} = (J^T J) @ v_k / ||(J^T J) @ v_k||

    After convergence, the Rayleigh quotient v^T (J^T J) v gives λ_max.

    Examples
    --------
    >>> lambda_max = estimate_max_eigenvalue(residual_fn, params)
    >>> print(f"Maximum eigenvalue: {lambda_max:.2e}")
    """
    matvec: Callable[[Float[Array, " n"]], Float[Array, " n"]] = (
        make_jtj_matvec(residual_fn, params, jnp.array(0.0))
    )
    key: Array = jax.random.PRNGKey(42)
    v: Float[Array, " n"] = jax.random.normal(key, params.shape)
    v = v / jnp.linalg.norm(v)

    def power_step(
        v_curr: Float[Array, " n"], _: None
    ) -> Tuple[Float[Array, " n"], None]:
        av: Float[Array, " n"] = matvec(v_curr)
        v_next: Float[Array, " n"] = av / jnp.linalg.norm(av)
        return v_next, None

    v_final: Float[Array, " n"]
    _: None
    v_final, _ = jax.lax.scan(power_step, v, None, length=num_iterations)
    av: Float[Array, " n"] = matvec(v_final)
    result: Float[Array, " "] = jnp.dot(v_final, av)
    return result


@partial(jax.jit, static_argnums=(0, 2))
@jaxtyped(typechecker=beartype)
def estimate_jtj_diagonal(
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    params: Float[Array, " n"],
    num_samples: int = 10,
) -> Float[Array, " n"]:
    """Estimate diagonal of J^T J via Hutchinson's trace estimator.

    The diagonal approximates per-parameter sensitivities and is useful
    for diagonal preconditioning in conjugate gradient.

    Parameters
    ----------
    residual_fn : Callable[[Float[Array, " n"]], Float[Array, " m"]]
        Residual function.
    params : Float[Array, " n"]
        Current parameters.
    num_samples : int, optional
        Number of random probes. Default is 10.

    Returns
    -------
    diagonal : Float[Array, " n"]
        Estimated diagonal of J^T J.

    Notes
    -----
    Uses Hutchinson's trick: diag(A) ≈ E[z ⊙ (A @ z)] for random
    z ∈ {-1, +1}^n (Rademacher distribution).

    The diagonal can be used to construct a preconditioner:
        M = diag(J^T J + ε)^{-1/2}

    Examples
    --------
    >>> diag = estimate_jtj_diagonal(residual_fn, params)
    >>> precond = 1.0 / jnp.sqrt(diag + 1e-6)
    """
    n: ScalarInteger = params.shape[0]
    matvec: Callable[[Float[Array, " n"]], Float[Array, " n"]] = (
        make_jtj_matvec(residual_fn, params, jnp.array(0.0))
    )

    def estimate_one(key: Array) -> Float[Array, " n"]:
        z: Float[Array, " n"] = jax.random.rademacher(
            key, (n,), dtype=params.dtype
        )
        az: Float[Array, " n"] = matvec(z)
        result: Float[Array, " n"] = z * az
        return result

    keys: Array = jax.random.split(jax.random.PRNGKey(0), num_samples)
    estimates: Float[Array, " num_samples n"] = jax.vmap(estimate_one)(keys)
    diagonal: Float[Array, " n"] = jnp.mean(estimates, axis=0)
    return diagonal
