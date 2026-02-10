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
jtj_matvec : function
    Create (J^T J + λI) matrix-vector product operator.
jt_residual : function
    Compute residuals and J^T @ r simultaneously.
hessian_matvec : function
    Create exact Hessian-vector product operator.
gn_step : function
    Generic Gauss-Newton step with trust-region damping.
gn_solve : function
    High-level solver that runs Gauss-Newton until convergence.
gn_history : function
    Gauss-Newton solver with per-iteration history tracking.
gn_loss_history : function
    Gauss-Newton solver with per-iteration loss tracking only.
max_eigenval : function
    Estimate largest eigenvalue of J^T J via power iteration.
jtj_diag : function
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
from beartype.typing import Callable, Tuple, Union
from jaxtyping import Array, Bool, Complex, Float, jaxtyped

from janssen.types import (
    GaussNewtonState,
    ScalarBool,
    ScalarInteger,
    ScalarNumeric,
)

from .math import flatten_params, unflatten_params

TRUST_REGION_EXCELLENT = 0.75
TRUST_REGION_ACCEPTABLE = 0.25
CONVERGENCE_TOL = 1e-8
DIVISION_EPSILON = 1e-12
LOSS_ZERO_TOL = 1e-14
MIN_DAMPING = 1e-12
MAX_DAMPING = 1e8


def jtj_matvec(
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

    Implementation Logic
    --------------------
    The function uses a two-stage construction:

    1. **Setup Phase** (executed once, outside returned closure):
       - Computes vjp_fn = jax.vjp(residual_fn, params)[1]
       - This captures the reverse-mode autodiff operator J^T @ (·)
       - Computing vjp_fn once and reusing it in the closure eliminates
         redundant computation when matvec is called multiple times
       - Critical optimization: without this, each matvec call would
         recompute vjp, doubling the number of forward passes

    2. **Matvec Phase** (executed per matrix-vector product):
       Internal function _matvec(v) computes:
       a. jv = jax.jvp(residual_fn, (params,), (v,))[1]
          - Forward-mode autodiff: directional derivative J @ v
          - Cost: 1 forward pass through residual_fn
       b. jtjv = vjp_fn(jv)[0]
          - Reverse-mode autodiff: J^T @ (J @ v)
          - Cost: 1 backward pass through residual_fn
       c. return jtjv + damping * v
          - Adds Levenberg-Marquardt regularization λI

    The key insight: J^T J never materialized. Each matvec costs only
    1 forward + 1 backward pass, vs O(mn) storage and O(mn) flops for
    explicit formation.

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

    The vjp_fn is computed once and captured in the closure. This is
    critical for performance: conjugate gradient will call the returned
    matvec 10-100 times per Gauss-Newton step. Without caching vjp_fn,
    we'd waste 50-100 extra forward passes.

    Examples
    --------
    >>> def residual_fn(x):
    ...     return jnp.array([x[0]**2 - 1, x[1]**2 - 4])
    >>> params = jnp.array([1.5, 2.5])
    >>> matvec = jtj_matvec(residual_fn, params, jnp.array(1e-3))
    >>> result = matvec(jnp.array([1.0, 0.0]))
    """
    _: Float[Array, " m"]
    vjp_fn: Callable[[Float[Array, " m"]], Tuple[Float[Array, " n"]]]
    _, vjp_fn = jax.vjp(residual_fn, params)

    def _matvec(v: Float[Array, " n"]) -> Float[Array, " n"]:
        _: Float[Array, " m"]
        jv: Float[Array, " m"]
        _, jv = jax.jvp(residual_fn, (params,), (v,))
        jtjv: Float[Array, " n"]
        (jtjv,) = vjp_fn(jv)
        return jtjv + damping * v

    return _matvec


@partial(jax.jit, static_argnums=(0,))
def jt_residual(
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    params: Float[Array, " n"],
    weights: Union[ScalarNumeric, Float[Array, " m"]] = -1.0,
) -> Tuple[Float[Array, " m"], Float[Array, " n"]]:
    """Compute residuals and J^T @ W @ r simultaneously.

    Uses reverse-mode autodiff to compute the gradient of the loss
    L = 0.5 * r^T W r, which equals J^T @ W @ r.

    For unweighted least squares (W = I), this reduces to J^T @ r.

    Implementation Logic
    --------------------
    The function exploits the chain rule via reverse-mode autodiff:

    1. **Compute residuals and VJP function**:
       residuals, vjp_fn = jax.vjp(residual_fn, params)
       - Evaluates r(θ) at current parameters
       - Constructs vjp_fn: vector → J^T @ vector
       - Single forward pass through residual_fn

    2. **Weight handling** (conditional based on weights parameter):
       - If weights < 0 (sentinel): set weights_normalized = ones(m)
         → Unweighted case W = I
       - Else: broadcast weights to residual shape
         → Weighted case W = diag(weights)
       - Uses jax.lax.cond for JIT compatibility (control flow traced)

    3. **Apply weighting and compute weighted gradient**:
       weighted_residuals = weights_normalized * residuals
       jt_wr = vjp_fn(weighted_residuals)[0]
       - Element-wise multiplication: W @ r for diagonal W
       - vjp_fn(W @ r) = J^T @ W @ r by chain rule
       - Single backward pass through residual_fn

    Total cost: 1 forward + 1 backward pass through residual_fn,
    regardless of whether weighting is used.

    Mathematical Derivation
    -----------------------
    For loss L(θ) = 0.5 * ||W^{1/2} r(θ)||² = 0.5 r(θ)^T W r(θ):

    ∇L = J^T W r

    where J is the Jacobian ∂r/∂θ. The VJP operator computes J^T @ v
    for any vector v. Setting v = W @ r gives J^T @ W @ r directly.

    Parameters
    ----------
    residual_fn : Callable[[Float[Array, " n"]], Float[Array, " m"]]
        Function mapping parameters to residuals.
    params : Float[Array, " n"]
        Current parameters.
    weights : Union[ScalarNumeric, Float[Array, " m"]], optional
        Diagonal weight matrix entries for weighted least squares.
        If -1.0 (default), assumes unweighted (W = I).

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
    >>> r, jt_r = jt_residual(residual_fn, params)
    >>> # Weighted version:
    >>> weights = jnp.array([2.0, 0.5])
    >>> r, jt_wr = jt_residual(residual_fn, params, weights)
    """
    residuals: Float[Array, " m"]
    vjp_fn: Callable[[Float[Array, " m"]], Tuple[Float[Array, " n"]]]
    residuals, vjp_fn = jax.vjp(residual_fn, params)
    weights_arr: Float[Array, "..."] = jnp.asarray(weights)
    weights_normalized: Float[Array, " m"]
    weights_normalized = jax.lax.cond(
        jnp.sum(weights_arr) < 0,
        lambda: jnp.ones_like(residuals),
        lambda: jnp.broadcast_to(weights_arr, residuals.shape),
    )
    weighted_residuals: Float[Array, " m"] = weights_normalized * residuals
    jt_wr: Float[Array, " n"]
    (jt_wr,) = vjp_fn(weighted_residuals)
    return residuals, jt_wr


def hessian_matvec(
    loss_fn: Callable[[Float[Array, " n"]], Float[Array, " "]],
    params: Float[Array, " n"],
) -> Callable[[Float[Array, " n"]], Float[Array, " n"]]:
    """Create exact Hessian-vector product operator.

    For problems where the Gauss-Newton approximation H ≈ J^T J is
    insufficient, this computes products with the exact Hessian:

        H @ v = d/dε [∇L(θ + εv)]|_{ε=0}

    using forward-over-reverse autodiff (jvp through grad).

    Implementation Logic
    --------------------
    The function uses forward-over-reverse composition:

    1. **Setup Phase** (executed once, outside returned closure):
       grad_fn = jax.grad(loss_fn)
       - Constructs the gradient function ∇L: R^n → R^n
       - This is reverse-mode autodiff: cost O(n) per gradient eval
       - Computing once and reusing avoids redundant graph construction

    2. **Hessian-Vector Product** (executed per matrix-vector product):
       Internal function _hvp(v) computes:
       hv = jax.jvp(grad_fn, (params,), (v,))[1]
       - Forward-mode autodiff through the gradient function
       - Computes directional derivative: d/dε [∇L(θ + εv)]|_{ε=0}
       - This equals H @ v by definition of the Hessian

    Mathematical Background
    -----------------------
    The Hessian H = ∇²L is the Jacobian of the gradient:

    H @ v = J_∇L @ v = d/dε [∇L(θ + εv)]|_{ε=0}

    Forward-over-reverse (fwd ∘ rev) is efficient for matrix-vector
    products, with cost roughly 2× a gradient evaluation.

    Contrast with Gauss-Newton approximation J^T J:
    - Exact Hessian: H = J^T J + Σᵢ rᵢ ∇²rᵢ
    - GN approximation: H ≈ J^T J (drops second term)
    - When residuals rᵢ are small or linear, J^T J is accurate
    - When residuals are large and nonlinear, exact H is needed

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

    Cost: ~2× gradient evaluation per HVP. For large n, this is much
    cheaper than forming H explicitly (which would cost O(n) gradient
    evaluations and O(n²) storage).

    Examples
    --------
    >>> def loss_fn(x):
    ...     return jnp.sum(x**4)
    >>> params = jnp.array([1.0, 2.0])
    >>> hvp = hessian_matvec(loss_fn, params)
    >>> result = hvp(jnp.array([1.0, 0.0]))
    """
    grad_fn: Callable[[Float[Array, " n"]], Float[Array, " n"]]
    grad_fn = jax.grad(loss_fn)

    def _hvp(v: Float[Array, " n"]) -> Float[Array, " n"]:
        _: Float[Array, " n"]
        hv: Float[Array, " n"]
        _, hv = jax.jvp(grad_fn, (params,), (v,))
        return hv

    return _hvp


@partial(jax.jit, static_argnums=(1, 2, 3))
@jaxtyped(typechecker=beartype)
def gn_step(
    state: GaussNewtonState,
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    cg_maxiter: int = 50,
    cg_tol: float = 1e-5,
    use_preconditioner: ScalarBool = False,
) -> GaussNewtonState:
    """Perform Gauss-Newton step with Levenberg-Marquardt damping.

    Solves the trust-region subproblem using conjugate gradient:

        (J^T J + λI) δ = -J^T r

    then updates parameters: θ_{k+1} = θ_k + δ

    The damping λ adapts based on actual vs predicted reduction.

    Implementation Logic
    --------------------
    The function executes a complete trust-region Gauss-Newton step:

    **Phase 1: Setup and Gradient Computation**
    1. Extract sample and probe shapes from state
    2. Flatten complex arrays (sample, probe) → real vector params
    3. Compute residuals r and gradient J^T @ r via jt_residual
       - Cost: 1 forward + 1 backward pass through residual_fn
    4. Compute current loss: 0.5 ||r||²

    **Phase 2: Linear System Construction**
    5. Build matrix-vector operator: matvec = jtj_matvec(...)
       - Computes (J^T J + λI) @ v using JVP/VJP composition
       - Caches vjp_fn to avoid redundant computation in CG
    6. If use_preconditioner=True:
       a. Estimate diagonal via Hutchinson: diag_jtj ≈ diag(J^T J)
          - Uses 10 Rademacher samples (random ±1 vectors)
          - Cost: 10 × (1 forward + 1 backward pass)
       b. Clamp negative estimates: max(diag_jtj, 0) + λ
          - Ensures positive definite preconditioner
          - Negative estimates can occur due to finite sampling
       c. Construct preconditioner: M(v) = v / (diag + λ)
          - Diagonal scaling: approximates (diag(H))^{-1}

    **Phase 3: Solve Trust-Region Subproblem**
    7. Solve (J^T J + λI) δ = -J^T r via conjugate gradient
       - Inputs: matvec operator, RHS = -J^T r, preconditioner M
       - Iterative solver: converges in typically 10-50 iterations
       - Cost per iteration: 1 matvec call = 1 forward + 1 backward
       - Total cost: ~20 × (1 forward + 1 backward) without precond
                     ~10 × (1 forward + 1 backward) with precond
    8. Compute step: params_new = params + δ
    9. Unflatten params_new → (sample_new, probe_new)

    **Phase 4: Evaluate New Point**
    10. Compute residuals_new = residual_fn(params_new)
        - Cost: 1 forward pass
    11. Compute new_loss = 0.5 ||residuals_new||²

    **Phase 5: Trust-Region Decision**
    12. Compute predicted reduction: pred = 0.5 δ^T (J^T J + λI) δ
        - Uses h_delta = matvec(δ) already computed
        - Predicted decrease in quadratic model
    13. Compute actual reduction: actual = current_loss - new_loss
    14. Compute reduction ratio: ρ = actual / predicted
        - If predicted ≤ 0: set ρ = 0 (non-descent direction)
    15. Accept/reject decision:
        accept = (actual > 0) AND (predicted > 0) AND (ρ > 0)
        - Requires all three: loss decreased, descent direction, positive ratio

    **Phase 6: Damping Adaptation**
    16. Update damping λ based on ρ (using nested jax.lax.cond):
        - If predicted ≤ 0: λ ← 10λ (drastic increase, bad step)
        - Else if ρ > 0.75: λ ← λ/3 (excellent step, be aggressive)
        - Else if ρ > 0.25: λ ← λ (acceptable step, maintain)
        - Else: λ ← 3λ (poor step, be conservative)
    17. Clamp damping: λ ∈ [10^{-12}, 10^8]
        - Prevents both over-damping (slow convergence) and
          under-damping (instability)

    **Phase 7: State Update**
    18. If accepted:
        - Use new sample, probe, loss
    19. If rejected:
        - Keep current sample, probe, loss
    20. Compute relative improvement |current_loss - final_loss| / current_loss
    21. Check convergence: accept AND (rel_improvement < 10^{-8})
    22. Return updated GaussNewtonState with:
        - sample, probe (updated or kept)
        - iteration counter incremented
        - loss (new or current)
        - damping (adapted)
        - converged flag

    Internal Helper Functions
    -------------------------
    Uses nested function _preconditioner_matvec(v) when preconditioning:
    - Computes element-wise: v / (diag_with_damping)
    - Passed as M parameter to jax.scipy.sparse.linalg.cg
    - Transforms system from (J^T J + λI) δ = -J^T r
      to M (J^T J + λI) δ = M (-J^T r)
    - Accelerates CG by improving condition number

    Computational Complexity
    ------------------------
    Without preconditioning:
    - Setup: 1 forward + 1 backward (gradient)
    - CG solve: ~20 × (1 forward + 1 backward) typical
    - Evaluation: 1 forward (new residuals)
    - Total: ~22 forward + ~21 backward passes

    With preconditioning:
    - Setup: 1 forward + 1 backward (gradient)
    - Diagonal estimation: 10 × (1 forward + 1 backward)
    - CG solve: ~10 × (1 forward + 1 backward) typical
    - Evaluation: 1 forward
    - Total: ~21 forward + ~21 backward passes

    Preconditioning breaks even immediately (fewer CG iterations offset
    the diagonal estimation cost), and provides 2-5× speedup for
    ill-conditioned problems.

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

    Convergence detection:
    - If loss < 1e-14: already at optimum, mark converged
    - If residuals contain NaN/Inf: mark converged (failure mode)
    - If relative improvement < 1e-8: normal convergence

    CG convergence limitation: JAX's sparse_linalg.cg currently returns
    None for the convergence info (second return value). This means we
    cannot detect when CG fails to converge within cg_maxiter iterations.
    The only recourse is choosing appropriate cg_maxiter and cg_tol, or
    enabling preconditioning for ill-conditioned problems. Future JAX
    versions may provide convergence diagnostics.

    Examples
    --------
    >>> state = make_gauss_newton_state(sample, probe)
    >>> new_state = gn_step(state, my_residual_fn)
    >>> # With preconditioning for better convergence:
    >>> new_state = gn_step(state, my_residual_fn,
    ...                               use_preconditioner=True)
    """
    sample_shape: Tuple[int, int] = state.sample.shape
    probe_shape: Tuple[int, int] = state.probe.shape
    params: Float[Array, " n"] = flatten_params(state.sample, state.probe)
    residuals: Float[Array, " m"]
    jt_r: Float[Array, " n"]
    residuals, jt_r = jt_residual(residual_fn, params)
    residuals_finite: Bool[Array, " "] = jnp.all(jnp.isfinite(residuals))
    residuals = jnp.where(
        residuals_finite, residuals, jnp.zeros_like(residuals)
    )
    current_loss: Float[Array, " "] = 0.5 * jnp.sum(residuals**2)
    loss_near_zero: Bool[Array, " "] = current_loss < LOSS_ZERO_TOL
    matvec: Callable[[Float[Array, " n"]], Float[Array, " n"]] = (
        jtj_matvec(residual_fn, params, state.damping)
    )
    diag_jtj: Float[Array, " n"] = jax.lax.cond(
        use_preconditioner,
        lambda: jtj_diag(residual_fn, params, num_samples=10),
        lambda: jnp.ones_like(params),
    )
    diag_with_damping: Float[Array, " n"] = (
        jnp.maximum(diag_jtj, 0.0) + state.damping
    )

    def _preconditioner(v: Float[Array, " n"]) -> Float[Array, " n"]:
        return jax.lax.cond(
            use_preconditioner,
            lambda: v / diag_with_damping,
            lambda: v,
        )

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
        actual_reduction / (predicted_reduction + DIVISION_EPSILON),
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
    new_damping = jnp.clip(new_damping, MIN_DAMPING, MAX_DAMPING)
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
        current_loss + DIVISION_EPSILON
    )
    step_converged: Bool[Array, " "] = accept & (
        rel_improvement < CONVERGENCE_TOL
    )
    converged: Bool[Array, " "] = (
        loss_near_zero | step_converged | (~residuals_finite)
    )

    return GaussNewtonState(
        sample=final_sample,
        probe=final_probe,
        iteration=state.iteration + 1,
        loss=final_loss,
        damping=new_damping,
        converged=converged,
    )


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
@jaxtyped(typechecker=beartype)
def gn_solve(
    state: GaussNewtonState,
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    max_iterations: int = 100,
    cg_maxiter: int = 50,
    cg_tol: float = 1e-5,
    use_preconditioner: bool = False,
) -> GaussNewtonState:
    """Run Gauss-Newton optimization until convergence or max iterations.

    High-level solver that repeatedly calls gn_step until the
    optimization converges or reaches the maximum iteration limit. This
    provides a single entry point for running the full optimization.

    Implementation Logic
    --------------------
    The function uses jax.lax.scan for efficient iteration:

    1. **Internal Step Function** step_fn(carry, _):
       Wraps gn_step with early stopping logic:
       a. Check if carry.converged is True
       b. If converged: return carry unchanged (no-op)
       c. If not converged: call gn_step(carry, ...)
       d. Return updated state

       Uses jax.lax.cond for control flow:
       - JIT-compatible conditional execution
       - When converged=True, subsequent iterations are no-ops
       - Ensures fixed iteration count for XLA compilation

    2. **Main Loop**:
       jax.lax.scan(step_fn, state, None, length=max_iterations)
       - Iterates exactly max_iterations times (required for JIT)
       - Carries GaussNewtonState through iterations
       - Stops updating when converged=True (via lax.cond guard)
       - Returns final_state after max_iterations or convergence

    Why scan instead of while_loop?
    --------------------------------
    JAX provides two iteration primitives:

    - jax.lax.while_loop: Stops when condition is False
      → Non-deterministic iteration count
      → Harder to JIT-compile (variable-length trace)

    - jax.lax.scan: Runs fixed number of iterations
      → Deterministic iteration count
      → Efficient JIT compilation
      → Early stopping via lax.cond guard inside body

    We use scan with internal lax.cond guard to get both:
    - Deterministic compilation (fixed max_iterations)
    - Early stopping (when converged=True, step_fn returns carry)

    Trade-off: If convergence happens at iteration 20, iterations
    21-100 are no-ops (just carry forwarding). Cost is negligible
    compared to actual Gauss-Newton steps.

    Computational Cost
    ------------------
    If convergence happens at iteration k < max_iterations:
    - Active iterations: k × cost(gn_step)
    - No-op iterations: (max_iterations - k) × cost(lax.cond)
    - No-op cost is ~microseconds, negligible vs GN step (~seconds)

    Total cost ≈ k × [20-40 forward/backward passes per step]

    For 100 max iterations with typical convergence at k=30:
    - Active cost: 30 × 40 = 1200 forward+backward passes
    - No-op cost: 70 × 0 ≈ 0 (compiler optimizes)

    Parameters
    ----------
    state : GaussNewtonState
        Initial optimization state containing sample, probe, iteration,
        loss, damping, and convergence status.
    residual_fn : Callable[[Float[Array, " n"]], Float[Array, " m"]]
        Function mapping flattened parameters to residuals.
    max_iterations : int, optional
        Maximum number of Gauss-Newton iterations. Default is 100.
    cg_maxiter : int, optional
        Maximum conjugate gradient iterations per step. Default is 50.
    cg_tol : float, optional
        CG convergence tolerance. Default is 1e-5.
    use_preconditioner : bool, optional
        Whether to use diagonal preconditioning for CG. Default is False.

    Returns
    -------
    final_state : GaussNewtonState
        Final optimization state after convergence or max iterations.

    Notes
    -----
    The solver stops when either:
    - The state's converged flag becomes True
    - The iteration count reaches max_iterations

    For monitoring progress during optimization, consider using this
    function with jax.lax.scan and a custom body function that logs
    intermediate state. This function is fully JIT-compatible.

    Examples
    --------
    >>> state = make_gauss_newton_state(sample, probe)
    >>> final_state = gn_solve(state, my_residual_fn,
    ...                                   max_iterations=50)
    >>> print(f"Converged: {final_state.converged}")
    >>> print(f"Final loss: {final_state.loss}")
    """

    def step_fn(
        carry: GaussNewtonState, _: None
    ) -> Tuple[GaussNewtonState, None]:
        result: GaussNewtonState = jax.lax.cond(
            carry.converged,
            lambda: carry,
            lambda: gn_step(
                carry,
                residual_fn,
                cg_maxiter=cg_maxiter,
                cg_tol=cg_tol,
                use_preconditioner=use_preconditioner,
            ),
        )
        return result, None

    final_state: GaussNewtonState
    _: None
    final_state, _ = jax.lax.scan(step_fn, state, None, length=max_iterations)
    return final_state


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
@jaxtyped(typechecker=beartype)
def gn_history(
    state: GaussNewtonState,
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    max_iterations: int = 100,
    cg_maxiter: int = 50,
    cg_tol: float = 1e-5,
    use_preconditioner: bool = False,
) -> Tuple[GaussNewtonState, GaussNewtonState, Float[Array, " N"]]:
    """Run Gauss-Newton optimization with per-iteration history tracking.

    Like gn_solve but returns intermediate states and losses
    at each iteration, enabling convergence diagnostics and visualization.

    Implementation Logic
    --------------------
    Same iteration strategy as gn_solve using jax.lax.scan,
    but scan outputs capture per-iteration states:

    1. **Internal Step Function** step_fn(carry, _):
       a. Check if carry.converged is True
       b. If converged: return carry unchanged (no-op)
       c. If not converged: call gn_step(carry, ...)
       d. Return updated state and output tuple (state, loss)

    2. **Main Loop**:
       jax.lax.scan(step_fn, state, None, length=max_iterations)
       - Returns both final_state and outputs PyTree
       - Outputs contain all intermediate states and losses

    The scan outputs are PyTrees where each leaf has an additional
    dimension of size max_iterations. For example, if state.sample
    has shape (H, W), then all_states.sample has shape (H, W, N)
    where N = max_iterations.

    Parameters
    ----------
    state : GaussNewtonState
        Initial optimization state containing sample, probe, iteration,
        loss, damping, and convergence status.
    residual_fn : Callable[[Float[Array, " n"]], Float[Array, " m"]]
        Function mapping flattened parameters to residuals.
    max_iterations : int, optional
        Maximum number of Gauss-Newton iterations. Default is 100.
    cg_maxiter : int, optional
        Maximum conjugate gradient iterations per step. Default is 50.
    cg_tol : float, optional
        CG convergence tolerance. Default is 1e-5.
    use_preconditioner : bool, optional
        Whether to use diagonal preconditioning for CG. Default is False.

    Returns
    -------
    final_state : GaussNewtonState
        Final optimization state after convergence or max iterations.
    all_states : GaussNewtonState
        PyTree with all intermediate states. Each field has an extra
        dimension of size max_iterations stacked along the last axis.
        For example, if state.sample is (H, W), all_states.sample
        is (H, W, max_iterations).
    all_losses : Float[Array, " N"]
        Loss value at each iteration, shape (max_iterations,).

    Notes
    -----
    This function is useful for:
    - Convergence diagnostics and plotting
    - Creating visualizations of optimization progress
    - Debugging optimization issues
    - Resume/continuation with full history tracking

    If you only need the final state (not intermediate history), use
    gn_solve instead for slightly lower memory usage.

    Examples
    --------
    >>> state = make_gauss_newton_state(sample, probe)
    >>> final, history, losses = gn_history(
    ...     state, my_residual_fn, max_iterations=50
    ... )
    >>> print(f"Converged: {final.converged}")
    >>> print(f"Loss history shape: {losses.shape}")  # (50,)
    >>> print(f"Sample history shape: {history.sample.shape}")  # (H, W, 50)
    """

    def step_fn(
        carry: GaussNewtonState, _: None
    ) -> Tuple[GaussNewtonState, Tuple[GaussNewtonState, Float[Array, " "]]]:
        result: GaussNewtonState = jax.lax.cond(
            carry.converged,
            lambda: carry,
            lambda: gn_step(
                carry,
                residual_fn,
                cg_maxiter=cg_maxiter,
                cg_tol=cg_tol,
                use_preconditioner=use_preconditioner,
            ),
        )
        return result, (result, result.loss)

    final_state: GaussNewtonState
    outputs: Tuple[GaussNewtonState, Float[Array, " N"]]
    final_state, outputs = jax.lax.scan(
        step_fn, state, None, length=max_iterations
    )
    all_states: GaussNewtonState
    all_losses: Float[Array, " N"]
    all_states, all_losses = outputs
    return final_state, all_states, all_losses


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
@jaxtyped(typechecker=beartype)
def gn_loss_history(
    state: GaussNewtonState,
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    max_iterations: int = 100,
    cg_maxiter: int = 50,
    cg_tol: float = 1e-5,
    use_preconditioner: bool = False,
) -> Tuple[GaussNewtonState, Float[Array, " N"]]:
    """Run Gauss-Newton optimization and return per-iteration losses.

    This variant tracks only the scalar loss per iteration and not the
    full state history, which keeps memory overhead low for large
    ptychography problems.
    """

    def step_fn(
        carry: GaussNewtonState, _: None
    ) -> Tuple[GaussNewtonState, Float[Array, " "]]:
        result: GaussNewtonState = jax.lax.cond(
            carry.converged,
            lambda: carry,
            lambda: gn_step(
                carry,
                residual_fn,
                cg_maxiter=cg_maxiter,
                cg_tol=cg_tol,
                use_preconditioner=use_preconditioner,
            ),
        )
        return result, result.loss

    final_state: GaussNewtonState
    all_losses: Float[Array, " N"]
    final_state, all_losses = jax.lax.scan(
        step_fn, state, None, length=max_iterations
    )
    return final_state, all_losses


@partial(jax.jit, static_argnums=(0, 2))
@jaxtyped(typechecker=beartype)
def max_eigenval(
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    params: Float[Array, " n"],
    num_iterations: int = 20,
) -> Float[Array, " "]:
    """Estimate largest eigenvalue of J^T J via power iteration.

    A large eigenvalue indicates potential ill-conditioning, which can
    cause convergence issues. Use this diagnostic to tune damping.

    Implementation Logic
    --------------------
    The function implements the power iteration algorithm:

    1. **Setup**:
       - Construct matvec operator: (J^T J) @ v via jtj_matvec
       - Initialize random unit vector v₀ ~ N(0, I), normalized
       - Uses fixed seed (42) for reproducibility

    2. **Power Iteration Loop** (num_iterations times):
       Internal function power_step(v_curr) performs:
       a. Compute Av = (J^T J) @ v_curr via matvec
          - Cost: 1 forward + 1 backward pass through residual_fn
       b. Normalize: v_next = Av / ||Av||
          - Ensures numerical stability, prevents overflow
       c. Return v_next (carry state for next iteration)

       Uses jax.lax.scan for efficient iteration:
       - Compiled as single fused kernel by XLA
       - Avoids Python loop overhead
       - scan(power_step, v₀, None, length=num_iterations)

    3. **Eigenvalue Extraction**:
       - Compute final Rayleigh quotient: λ = v^T (J^T J) v
       - After k iterations, vₖ ≈ dominant eigenvector
       - Rayleigh quotient converges to λ_max(J^T J)

    Algorithm: Power Iteration
    --------------------------
    For symmetric positive semi-definite A = J^T J:

    vₖ₊₁ = A vₖ / ||A vₖ||

    Converges to dominant eigenvector v₁ at rate (λ₂/λ₁)^k where
    λ₁ ≥ λ₂ ≥ ... are eigenvalues. Convergence is geometric.

    With k=20 iterations and typical spectral gaps, relative error
    is usually < 1%. For very ill-conditioned problems (λ₁ ≈ λ₂),
    increase num_iterations to 50-100.

    Computational Cost
    ------------------
    - Total: num_iterations × (1 forward + 1 backward pass)
    - For num_iterations=20: 20 forward + 20 backward passes
    - Plus 1 final matvec for Rayleigh quotient
    - Memory: O(n) for storing vₖ

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

    This is useful for:
    - Diagnosing ill-conditioning: condition number κ ≈ λ_max / λ_min
    - Tuning initial damping: set λ₀ ~ 10^{-3} λ_max as starting point
    - Detecting rank deficiency: if λ_max ≈ 0, J has null space

    Examples
    --------
    >>> lambda_max = max_eigenval(residual_fn, params)
    >>> print(f"Maximum eigenvalue: {lambda_max:.2e}")
    """
    matvec: Callable[[Float[Array, " n"]], Float[Array, " n"]] = (
        jtj_matvec(residual_fn, params, jnp.array(0.0))
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
def jtj_diag(
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    params: Float[Array, " n"],
    num_samples: int = 10,
) -> Float[Array, " n"]:
    """Estimate diagonal of J^T J via Hutchinson's trace estimator.

    The diagonal approximates per-parameter sensitivities and is useful
    for diagonal preconditioning in conjugate gradient.

    Implementation Logic
    --------------------
    The function implements Hutchinson's stochastic trace estimator:

    1. **Setup**:
       - Construct matvec operator: (J^T J) @ v via jtj_matvec
       - Generate num_samples independent random keys

    2. **Per-Sample Estimation** (parallelized via vmap):
       Internal function estimate_one(key) performs:
       a. Sample probe vector z ~ Rademacher({-1, +1}^n)
          - Each entry independently ±1 with equal probability
          - Uses params.dtype to match precision (float32/float64)
       b. Compute Az = (J^T J) @ z via matvec
          - Cost: 1 forward + 1 backward pass through residual_fn
       c. Compute element-wise product: z ⊙ Az
          - This is the single-sample diagonal estimate
       d. Return z ⊙ Az

       Uses jax.vmap(estimate_one)(keys):
       - Vectorizes over num_samples keys
       - Compiles to parallel batch of matrix-vector products
       - Returns shape (num_samples, n) stacked estimates

    3. **Averaging**:
       - Compute mean across samples: diagonal = mean(estimates, axis=0)
       - Reduces variance by √(num_samples)

    Algorithm: Hutchinson's Estimator
    ----------------------------------
    For any symmetric matrix A, diagonal entries satisfy:

    A_ii = E_z[z_i (A @ z)_i]

    where z ~ Rademacher({-1, +1}^n). This is unbiased:

    E[z_i (A @ z)_i] = E[z_i Σⱼ A_ij z_j]
                      = Σⱼ A_ij E[z_i z_j]
                      = A_ii

    since E[z_i z_j] = δ_ij (orthogonality of Rademacher).

    Variance and Sampling
    ---------------------
    The variance of the single-sample estimator is:

    Var[z_i (A @ z)_i] ≈ 2 Σⱼ A_ij²

    For matrices with strong off-diagonal structure (like J^T J in
    ptychography due to overlapping scan positions), this variance
    can be large. The estimator variance scales as:

    Var[diagonal_estimate] ~ (2/num_samples) Σⱼ A_ij²

    With num_samples=10, standard error is ~√(0.2 Σⱼ A_ij²). For
    ill-conditioned problems, increase to 50-100 samples.

    Why Rademacher? Could also use Gaussian, but Rademacher:
    - Has lower variance for heavy-tailed eigenvalue distributions
    - Is memory-efficient (binary values)
    - Allows exact integer arithmetic in some contexts

    Note: Individual diagonal estimates can be negative due to
    finite sampling, even though true diagonal entries are non-negative
    (J^T J is positive semi-definite). When using for preconditioning,
    clamp to zero: max(diagonal, 0) before inversion.

    Computational Cost
    ------------------
    - Total: num_samples × (1 forward + 1 backward pass)
    - For num_samples=10: 10 forward + 10 backward passes
    - All samples computed in parallel via vmap (batch size = num_samples)
    - Memory: O(num_samples × n) for stacked estimates

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

    Trade-off: More samples → lower variance, higher cost. For most
    problems, 10 samples is sufficient. For high accuracy or when
    diagonal entries vary by orders of magnitude, use 50-100 samples.

    Examples
    --------
    >>> diag = jtj_diag(residual_fn, params)
    >>> precond = 1.0 / jnp.sqrt(diag + 1e-6)
    """
    n: ScalarInteger = params.shape[0]
    matvec: Callable[[Float[Array, " n"]], Float[Array, " n"]] = (
        jtj_matvec(residual_fn, params, jnp.array(0.0))
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
