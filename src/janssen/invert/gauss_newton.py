"""Jacobian-free Gauss-Newton optimization for ptychography.

Extended Summary
----------------
Second-order optimization using JAX's autodifferentiation primitives
to solve normal equations without materializing the Jacobian. This module
demonstrates why JAX's functional model is well-suited for large-scale
inverse problems in computational imaging.

The core insight is that conjugate gradient methods require only
matrix-vector products (J^T J) v, not the matrix itself. JAX provides:

- ``jax.jvp``: computes J @ v via forward-mode autodiff
- ``jax.vjp``: computes J^T @ u via reverse-mode autodiff

Composing these yields (J^T J) @ v = vjp(jvp(v)) without ever forming J.
For ptychography with K scan positions on N×N grids, this reduces memory
from O(K·N^4) to O(K·N^2 + N^2), making second-order optimization tractable.

Routine Listings
----------------
gauss_newton_step : function
    Single Gauss-Newton update with Levenberg-Marquardt damping.
gauss_newton_reconstruct : function
    Full reconstruction loop.
make_jtj_matvec : function
    Create (J^T J + λI) matrix-vector product operator.
estimate_jtj_diagonal : function
    Estimate diagonal of J^T J for preconditioning.
estimate_condition_number : function
    Diagnose ill-conditioning via power iteration.

Notes
-----
All functions follow Janssen's conventions: jaxtyped/beartype validation,
pure functions operating on PyTrees, and compatibility with jit/grad/vmap.

References
----------
.. [1] Nocedal & Wright, "Numerical Optimization", Chapter 10
.. [2] Thibault et al., "Probe retrieval in ptychographic CDI"
"""

from functools import partial
from beartype.typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as sparse_linalg
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from janssen.types import GaussNewtonState, make_gauss_newton_state


@jax.jit
@jaxtyped(typechecker=beartype)
def flatten_params(
    sample: Complex[Array, " H W"],
    probe: Complex[Array, " H W"],
) -> Float[Array, " n"]:
    """Flatten complex arrays to real parameter vector.

    Layout: [sample_real, sample_imag, probe_real, probe_imag]

    Parameters
    ----------
    sample : Complex[Array, " H W"]
        Object transmission function.
    probe : Complex[Array, " H W"]
        Probe wavefront.

    Returns
    -------
    params : Float[Array, " n"]
        Flattened real vector with n = 4 * H * W.
    """
    return jnp.concatenate(
        [
            sample.real.ravel(),
            sample.imag.ravel(),
            probe.real.ravel(),
            probe.imag.ravel(),
        ]
    )


@jax.jit
def unflatten_params(
    params: Float[Array, " n"],
    shape: Tuple[int, int],
) -> Tuple[Complex[Array, " H W"], Complex[Array, " H W"]]:
    """Unflatten real vector to complex arrays.

    Parameters
    ----------
    params : Float[Array, " n"]
        Flattened real parameter vector.
    shape : Tuple[int, int]
        Shape (H, W) of each array.

    Returns
    -------
    sample : Complex[Array, " H W"]
        Object transmission function.
    probe : Complex[Array, " H W"]
        Probe wavefront.
    """
    size = shape[0] * shape[1]
    sample = params[:size].reshape(shape) + 1j * params[
        size : 2 * size
    ].reshape(shape)
    probe = params[2 * size : 3 * size].reshape(shape) + 1j * params[
        3 * size :
    ].reshape(shape)
    return sample, probe


# =============================================================================
# Forward Model and Residuals
# =============================================================================


@partial(jax.jit, static_argnums=(4,))
@jaxtyped(typechecker=beartype)
def ptychography_forward(
    sample: Complex[Array, " H W"],
    probe: Complex[Array, " H W"],
    positions: Float[Array, " K 2"],
    wavelength: Float[Array, " "],
    shape: Tuple[int, int],
) -> Float[Array, " K H W"]:
    """Ptychographic forward model: parameters -> diffraction intensities.

    For each scan position, computes:
        I_k = |FFT(probe_shifted * sample)|^2

    Parameters
    ----------
    sample : Complex[Array, " H W"]
        Object transmission function.
    probe : Complex[Array, " H W"]
        Probe wavefront.
    positions : Float[Array, " K 2"]
        Scan positions in pixels (x, y).
    wavelength : Float[Array, " "]
        Wavelength (for future propagation extensions).
    shape : Tuple[int, int]
        Grid shape (static for JIT).

    Returns
    -------
    intensities : Float[Array, " K H W"]
        Predicted diffraction intensities at each position.
    """
    h, w = shape

    def single_position(pos: Float[Array, " 2"]) -> Float[Array, " H W"]:
        """Forward model for one scan position."""
        # Fourier shift probe to scan position
        ky = jnp.fft.fftfreq(h)[:, None]
        kx = jnp.fft.fftfreq(w)[None, :]
        phase_ramp = jnp.exp(-2j * jnp.pi * (kx * pos[0] + ky * pos[1]))
        probe_shifted = jnp.fft.ifft2(jnp.fft.fft2(probe) * phase_ramp)

        # Exit wave and far-field propagation
        exit_wave = sample * probe_shifted
        far_field = jnp.fft.fftshift(jnp.fft.fft2(exit_wave))

        return jnp.abs(far_field) ** 2

    # Vectorize over scan positions using vmap
    return jax.vmap(single_position)(positions)


@partial(jax.jit, static_argnums=(3,))
@jaxtyped(typechecker=beartype)
def amplitude_residuals(
    params: Float[Array, " n"],
    measured: Float[Array, " K H W"],
    positions: Float[Array, " K 2"],
    shape: Tuple[int, int],
) -> Float[Array, " m"]:
    """Compute amplitude residuals: r = sqrt(I_meas) - sqrt(I_pred).

    The amplitude formulation provides better-conditioned gradients
    than intensity residuals, especially in low-signal regions.

    Parameters
    ----------
    params : Float[Array, " n"]
        Flattened parameter vector.
    measured : Float[Array, " K H W"]
        Measured diffraction intensities.
    positions : Float[Array, " K 2"]
        Scan positions in pixels.
    shape : Tuple[int, int]
        Grid shape (H, W).

    Returns
    -------
    residuals : Float[Array, " m"]
        Flattened residual vector with m = K * H * W.
    """
    sample, probe = unflatten_params(params, shape)
    predicted = ptychography_forward(
        sample, probe, positions, jnp.array(1.0), shape
    )

    amp_meas = jnp.sqrt(jnp.maximum(measured, 1e-12))
    amp_pred = jnp.sqrt(jnp.maximum(predicted, 1e-12))

    return (amp_meas - amp_pred).ravel()


# =============================================================================
# Jacobian-Free Linear Algebra via jvp/vjp
# =============================================================================


def make_jtj_matvec(
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    params: Float[Array, " n"],
    damping: Float[Array, " "],
) -> Callable[[Float[Array, " n"]], Float[Array, " n"]]:
    """Create matrix-vector product operator for (J^T J + λI).

    This is the core of Jacobian-free Gauss-Newton. We never form J,
    instead computing products via JAX's autodiff primitives:

        (J^T J) v = J^T (J v)
                  = vjp(jvp(v))

    Parameters
    ----------
    residual_fn : Callable
        Function mapping parameters to residuals.
    params : Float[Array, " n"]
        Current parameters (linearization point).
    damping : Float[Array, " "]
        Levenberg-Marquardt damping λ.

    Returns
    -------
    matvec : Callable
        Function computing (J^T J + λI) @ v.

    Notes
    -----
    Memory complexity is O(m + n) per product, versus O(m * n) to store J.
    For ptychography: m ~ K * N^2, n ~ N^2, so this is essential.
    """

    def matvec(v: Float[Array, " n"]) -> Float[Array, " n"]:
        # Forward-mode: J @ v
        # jax.jvp returns (f(x), J @ v) given (x, v)
        _, jv = jax.jvp(residual_fn, (params,), (v,))

        # Reverse-mode: J^T @ (J @ v)
        # jax.vjp returns (f(x), vjp_fn) where vjp_fn(u) = J^T @ u
        _, vjp_fn = jax.vjp(residual_fn, params)
        (jtjv,) = vjp_fn(jv)

        # Add Levenberg-Marquardt damping
        return jtjv + damping * v

    return matvec


def compute_jt_residual(
    residual_fn: Callable[[Float[Array, " n"]], Float[Array, " m"]],
    params: Float[Array, " n"],
) -> Tuple[Float[Array, " m"], Float[Array, " n"]]:
    """Compute residuals and J^T @ r simultaneously.

    Uses reverse-mode autodiff to get the gradient of the loss
    L = 0.5 * ||r||^2, which equals J^T @ r.

    Parameters
    ----------
    residual_fn : Callable
        Function mapping parameters to residuals.
    params : Float[Array, " n"]
        Current parameters.

    Returns
    -------
    residuals : Float[Array, " m"]
        Current residual vector.
    jt_r : Float[Array, " n"]
        J^T @ r = gradient of 0.5 * ||r||^2.
    """
    residuals, vjp_fn = jax.vjp(residual_fn, params)
    (jt_r,) = vjp_fn(residuals)
    return residuals, jt_r


# =============================================================================
# Gauss-Newton Step with JAX's CG Solver
# =============================================================================


@partial(jax.jit, static_argnums=(3, 4, 5))
@jaxtyped(typechecker=beartype)
def gauss_newton_step(
    state: GaussNewtonState,
    measured: Float[Array, " K H W"],
    positions: Float[Array, " K 2"],
    shape: Tuple[int, int],
    cg_maxiter: int = 50,
    cg_tol: float = 1e-5,
) -> GaussNewtonState:
    """Single Gauss-Newton step with Levenberg-Marquardt damping.

    Solves the normal equations using JAX's conjugate gradient:

        (J^T J + λI) δ = J^T r

    Then updates: θ_{k+1} = θ_k - δ

    The damping λ adapts based on actual vs predicted reduction
    (trust-region style).

    Parameters
    ----------
    state : GaussNewtonState
        Current optimization state.
    measured : Float[Array, " K H W"]
        Measured diffraction intensities.
    positions : Float[Array, " K 2"]
        Scan positions.
    shape : Tuple[int, int]
        Grid shape (H, W), static for JIT.
    cg_maxiter : int
        Maximum conjugate gradient iterations.
    cg_tol : float
        CG convergence tolerance.

    Returns
    -------
    new_state : GaussNewtonState
        Updated optimization state.
    """
    # Flatten current parameters
    params = flatten_params(state.sample, state.probe)

    # Define residual function for this problem
    def residual_fn(p: Float[Array, " n"]) -> Float[Array, " m"]:
        return amplitude_residuals(p, measured, positions, shape)

    # Compute residuals and J^T @ r
    residuals, jt_r = compute_jt_residual(residual_fn, params)
    current_loss = 0.5 * jnp.sum(residuals**2)

    # Build (J^T J + λI) operator
    matvec = make_jtj_matvec(residual_fn, params, state.damping)

    # Solve normal equations using JAX's CG
    # jax.scipy.sparse.linalg.cg(A, b) where A is a matvec function
    delta, _ = sparse_linalg.cg(
        matvec,
        jt_r,
        x0=jnp.zeros_like(params),
        maxiter=cg_maxiter,
        tol=cg_tol,
    )

    # Candidate update
    params_new = params - delta
    sample_new, probe_new = unflatten_params(params_new, shape)

    # Evaluate new loss
    residuals_new = residual_fn(params_new)
    new_loss = 0.5 * jnp.sum(residuals_new**2)

    # Trust-region damping adaptation
    # Predicted reduction from quadratic model
    predicted_reduction = jnp.dot(jt_r, delta) - 0.5 * jnp.dot(
        delta, matvec(delta) - state.damping * delta
    )
    actual_reduction = current_loss - new_loss
    rho = actual_reduction / (predicted_reduction + 1e-12)

    # Accept if improvement
    accept = rho > 0.0

    # Adapt damping: good step -> decrease (more aggressive)
    #                poor step -> increase (more conservative)
    new_damping = jax.lax.cond(
        rho > 0.75,
        lambda: state.damping * 0.33,  # Very good: be aggressive
        lambda: jax.lax.cond(
            rho > 0.25,
            lambda: state.damping,  # Acceptable: keep damping
            lambda: state.damping * 3.0,  # Poor: be conservative
        ),
    )
    new_damping = jnp.clip(new_damping, 1e-12, 1e8)

    # Conditional update
    final_sample = jax.lax.cond(
        accept, lambda: sample_new, lambda: state.sample
    )
    final_probe = jax.lax.cond(accept, lambda: probe_new, lambda: state.probe)
    final_loss = jax.lax.cond(accept, lambda: new_loss, lambda: current_loss)

    # Check convergence
    rel_improvement = jnp.abs(current_loss - final_loss) / (
        current_loss + 1e-12
    )
    converged = rel_improvement < 1e-8

    return GaussNewtonState(
        sample=final_sample,
        probe=final_probe,
        iteration=state.iteration + 1,
        loss=final_loss,
        damping=new_damping,
        converged=converged,
    )


# =============================================================================
# Full Reconstruction Loop
# =============================================================================


@jaxtyped(typechecker=beartype)
def gauss_newton_reconstruct(
    initial_sample: Complex[Array, " H W"],
    initial_probe: Complex[Array, " H W"],
    measured: Float[Array, " K H W"],
    positions: Float[Array, " K 2"],
    max_iterations: int = 100,
    initial_damping: float = 1e-3,
    cg_maxiter: int = 50,
    cg_tol: float = 1e-5,
) -> GaussNewtonState:
    """Run Gauss-Newton ptychography reconstruction.

    Parameters
    ----------
    initial_sample : Complex[Array, " H W"]
        Initial object estimate.
    initial_probe : Complex[Array, " H W"]
        Initial probe estimate.
    measured : Float[Array, " K H W"]
        Measured diffraction intensities.
    positions : Float[Array, " K 2"]
        Scan positions in pixels.
    max_iterations : int
        Maximum Gauss-Newton iterations.
    initial_damping : float
        Initial Levenberg-Marquardt damping.
    cg_maxiter : int
        Max CG iterations per GN step.
    cg_tol : float
        CG convergence tolerance.

    Returns
    -------
    final_state : GaussNewtonState
        Converged reconstruction state.

    Notes
    -----
    Uses jax.lax.while_loop for JIT-compatible iteration.
    """
    shape = initial_sample.shape
    state = make_gauss_newton_state(
        initial_sample, initial_probe, initial_damping
    )

    def cond_fn(s: GaussNewtonState) -> bool:
        return (s.iteration < max_iterations) & (~s.converged)

    def body_fn(s: GaussNewtonState) -> GaussNewtonState:
        return gauss_newton_step(
            s, measured, positions, shape, cg_maxiter, cg_tol
        )

    return jax.lax.while_loop(cond_fn, body_fn, state)


# =============================================================================
# Diagnostics: Condition Number via Power Iteration
# =============================================================================


@partial(jax.jit, static_argnums=(4, 5))
@jaxtyped(typechecker=beartype)
def estimate_condition_number(
    sample: Complex[Array, " H W"],
    probe: Complex[Array, " H W"],
    measured: Float[Array, " K H W"],
    positions: Float[Array, " K 2"],
    shape: Tuple[int, int],
    num_iterations: int = 20,
) -> Float[Array, " "]:
    """Estimate largest eigenvalue of J^T J via power iteration.

    A high value indicates potential ill-conditioning. Use this
    to diagnose convergence issues and tune damping.

    Parameters
    ----------
    sample, probe : Complex arrays
        Current reconstruction state.
    measured : Float[Array, " K H W"]
        Measured data.
    positions : Float[Array, " K 2"]
        Scan positions.
    shape : Tuple[int, int]
        Grid shape.
    num_iterations : int
        Power iteration count.

    Returns
    -------
    lambda_max : Float[Array, " "]
        Estimate of largest eigenvalue of J^T J.
    """
    params = flatten_params(sample, probe)

    def residual_fn(p):
        return amplitude_residuals(p, measured, positions, shape)

    # Undamped J^T J
    matvec = make_jtj_matvec(residual_fn, params, jnp.array(0.0))

    # Power iteration using jax.lax.scan
    key = jax.random.PRNGKey(42)
    v = jax.random.normal(key, params.shape)
    v = v / jnp.linalg.norm(v)

    def power_step(v, _):
        Av = matvec(v)
        v_new = Av / jnp.linalg.norm(Av)
        return v_new, None

    v_final, _ = jax.lax.scan(power_step, v, None, length=num_iterations)

    # Rayleigh quotient gives eigenvalue estimate
    Av = matvec(v_final)
    lambda_max = jnp.dot(v_final, Av)

    return lambda_max


# =============================================================================
# Diagnostics: Diagonal Estimation via Hutchinson's Trace Estimator
# =============================================================================


@partial(jax.jit, static_argnums=(4, 5))
@jaxtyped(typechecker=beartype)
def estimate_jtj_diagonal(
    sample: Complex[Array, " H W"],
    probe: Complex[Array, " H W"],
    measured: Float[Array, " K H W"],
    positions: Float[Array, " K 2"],
    shape: Tuple[int, int],
    num_samples: int = 10,
) -> Float[Array, " n"]:
    """Estimate diagonal of J^T J via Hutchinson's trace estimator.

    The diagonal approximates per-parameter sensitivities, useful for:
    - Diagonal preconditioning in CG
    - Understanding which parameters are well/poorly constrained

    Parameters
    ----------
    sample, probe : Complex arrays
        Current state.
    measured : Float[Array, " K H W"]
        Measured data.
    positions : Float[Array, " K 2"]
        Scan positions.
    shape : Tuple[int, int]
        Grid shape.
    num_samples : int
        Number of random probes for estimation.

    Returns
    -------
    diagonal : Float[Array, " n"]
        Estimated diagonal of J^T J.

    Notes
    -----
    Uses Hutchinson's trick: diag(A) ≈ E[z ⊙ (A @ z)] for random z ∈ {-1, +1}.
    """
    params = flatten_params(sample, probe)
    n = params.shape[0]

    def residual_fn(p):
        return amplitude_residuals(p, measured, positions, shape)

    matvec = make_jtj_matvec(residual_fn, params, jnp.array(0.0))

    def estimate_one(key):
        # Rademacher random vector
        z = jax.random.rademacher(key, (n,), dtype=jnp.float32)
        Az = matvec(z)
        return z * Az  # Element-wise product

    keys = jax.random.split(jax.random.PRNGKey(0), num_samples)
    estimates = jax.vmap(estimate_one)(keys)

    return jnp.mean(estimates, axis=0)


# =============================================================================
# Preconditioned Gauss-Newton
# =============================================================================


@partial(jax.jit, static_argnums=(4, 5, 6))
@jaxtyped(typechecker=beartype)
def preconditioned_gauss_newton_step(
    state: GaussNewtonState,
    measured: Float[Array, " K H W"],
    positions: Float[Array, " K 2"],
    preconditioner: Float[Array, " n"],
    shape: Tuple[int, int],
    cg_maxiter: int = 50,
    cg_tol: float = 1e-5,
) -> GaussNewtonState:
    """Gauss-Newton step with diagonal preconditioning.

    Preconditioning improves CG convergence for ill-conditioned problems
    by rescaling so all parameters have similar sensitivity.

    Parameters
    ----------
    state : GaussNewtonState
        Current state.
    measured : Float[Array, " K H W"]
        Measured data.
    positions : Float[Array, " K 2"]
        Scan positions.
    preconditioner : Float[Array, " n"]
        Diagonal preconditioner (typically 1/sqrt(diag(J^T J) + ε)).
    shape : Tuple[int, int]
        Grid shape.
    cg_maxiter : int
        Max CG iterations.
    cg_tol : float
        CG tolerance.

    Returns
    -------
    new_state : GaussNewtonState
        Updated state.
    """
    params = flatten_params(state.sample, state.probe)

    def residual_fn(p):
        return amplitude_residuals(p, measured, positions, shape)

    residuals, jt_r = compute_jt_residual(residual_fn, params)
    current_loss = 0.5 * jnp.sum(residuals**2)

    # Preconditioned operator: M^{-1} (J^T J + λI) M^{-1}
    M_inv = preconditioner

    def precond_matvec(v):
        Mv = M_inv * v
        base_matvec = make_jtj_matvec(residual_fn, params, state.damping)
        AMv = base_matvec(Mv)
        return M_inv * AMv

    precond_rhs = M_inv * jt_r

    y, _ = sparse_linalg.cg(
        precond_matvec,
        precond_rhs,
        x0=jnp.zeros_like(params),
        maxiter=cg_maxiter,
        tol=cg_tol,
    )
    delta = M_inv * y

    # Standard trust-region update logic
    params_new = params - delta
    sample_new, probe_new = unflatten_params(params_new, shape)

    residuals_new = residual_fn(params_new)
    new_loss = 0.5 * jnp.sum(residuals_new**2)

    base_matvec = make_jtj_matvec(residual_fn, params, state.damping)
    predicted_reduction = jnp.dot(jt_r, delta) - 0.5 * jnp.dot(
        delta, base_matvec(delta) - state.damping * delta
    )
    actual_reduction = current_loss - new_loss
    rho = actual_reduction / (predicted_reduction + 1e-12)

    accept = rho > 0.0
    new_damping = jax.lax.cond(
        rho > 0.75,
        lambda: state.damping * 0.33,
        lambda: jax.lax.cond(
            rho > 0.25, lambda: state.damping, lambda: state.damping * 3.0
        ),
    )
    new_damping = jnp.clip(new_damping, 1e-12, 1e8)

    final_sample = jax.lax.cond(
        accept, lambda: sample_new, lambda: state.sample
    )
    final_probe = jax.lax.cond(accept, lambda: probe_new, lambda: state.probe)
    final_loss = jax.lax.cond(accept, lambda: new_loss, lambda: current_loss)

    rel_improvement = jnp.abs(current_loss - final_loss) / (
        current_loss + 1e-12
    )
    converged = rel_improvement < 1e-8

    return GaussNewtonState(
        sample=final_sample,
        probe=final_probe,
        iteration=state.iteration + 1,
        loss=final_loss,
        damping=new_damping,
        converged=converged,
    )


# =============================================================================
# Exact Hessian-Vector Products (for Newton-CG)
# =============================================================================


def make_hessian_matvec(
    loss_fn: Callable[[Float[Array, " n"]], Float[Array, " "]],
    params: Float[Array, " n"],
) -> Callable[[Float[Array, " n"]], Float[Array, " n"]]:
    """Create exact Hessian-vector product operator.

    For problems where Gauss-Newton approximation (H ≈ J^T J) is
    insufficient, the exact Hessian captures full curvature:

        H @ v = d/dε [∇L(θ + εv)]|_{ε=0}

    JAX computes this efficiently via forward-over-reverse autodiff.

    Parameters
    ----------
    loss_fn : Callable
        Scalar loss function L(θ).
    params : Float[Array, " n"]
        Current parameters.

    Returns
    -------
    hvp : Callable
        Function computing H @ v.
    """

    def hvp(v: Float[Array, " n"]) -> Float[Array, " n"]:
        # grad_fn: θ -> ∇L(θ)
        grad_fn = jax.grad(loss_fn)
        # Forward-mode through grad gives Hessian-vector product
        _, hv = jax.jvp(grad_fn, (params,), (v,))
        return hv

    return hvp


# =============================================================================
# Convenience: Loss Function Factory
# =============================================================================


def make_ptychography_loss(
    measured: Float[Array, " K H W"],
    positions: Float[Array, " K 2"],
    shape: Tuple[int, int],
) -> Callable[[Float[Array, " n"]], Float[Array, " "]]:
    """Create scalar loss function for ptychography.

    Returns L(θ) = 0.5 * ||r(θ)||^2 suitable for use with
    make_hessian_matvec or standard gradient-based optimization.

    Parameters
    ----------
    measured : Float[Array, " K H W"]
        Measured data.
    positions : Float[Array, " K 2"]
        Scan positions.
    shape : Tuple[int, int]
        Grid shape.

    Returns
    -------
    loss_fn : Callable
        Scalar loss function.
    """

    @jax.jit
    def loss_fn(params: Float[Array, " n"]) -> Float[Array, " "]:
        r = amplitude_residuals(params, measured, positions, shape)
        return 0.5 * jnp.sum(r**2)

    return loss_fn


# =============================================================================
# Gradient Computation (for comparison / hybrid methods)
# =============================================================================


def compute_gradient(
    params: Float[Array, " n"],
    measured: Float[Array, " K H W"],
    positions: Float[Array, " K 2"],
    shape: Tuple[int, int],
) -> Float[Array, " n"]:
    """Compute gradient of ptychography loss.

    Uses JAX's grad transformation for reverse-mode autodiff.
    Equivalent to J^T @ r but computed via the loss formulation.

    Parameters
    ----------
    params : Float[Array, " n"]
        Current parameters.
    measured : Float[Array, " K H W"]
        Measured data.
    positions : Float[Array, " K 2"]
        Scan positions.
    shape : Tuple[int, int]
        Grid shape.

    Returns
    -------
    gradient : Float[Array, " n"]
        Gradient ∇L = J^T @ r.
    """
    loss_fn = make_ptychography_loss(measured, positions, shape)
    return jax.grad(loss_fn)(params)
