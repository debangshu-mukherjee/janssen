"""Coherent mode generation and decomposition.

Extended Summary
----------------
This module provides functions for generating and manipulating coherent
modes for partially coherent field representation. Any partially coherent
field can be decomposed into orthogonal coherent modes (Mercer's theorem):

    J(r1, r2) = sum_n lambda_n * phi_n*(r1) * phi_n(r2)

where phi_n are orthonormal modes and lambda_n are their weights (eigenvalues).

The total intensity is the incoherent sum of mode intensities:
    I(r) = sum_n lambda_n * |phi_n(r)|^2

This representation enables efficient simulation by propagating a finite
number of coherent modes and summing intensities.

Routine Listings
----------------
hermite_gaussian_modes : function
    Generate Hermite-Gaussian coherent mode set.
gaussian_schell_model_modes : function
    Generate modes for Gaussian Schell-model source.
eigenmode_decomposition : function
    Decompose mutual intensity into coherent modes.
effective_mode_count : function
    Calculate effective number of modes (participation ratio).
modes_from_wavefront : function
    Wrap a coherent field as a single-mode set.
mutual_intensity_from_modes : function
    Reconstruct mutual intensity from coherent modes.
_hermite_gaussian_modes_impl : function, internal (pure JAX)
    JIT-compiled Hermite-Gaussian mode generation.
_gaussian_schell_model_modes_impl : function, internal (pure JAX)
    JIT-compiled Gaussian Schell-model mode generation.
_eigenmode_decomposition_impl : function, internal (pure JAX)
    JIT-compiled eigenmode decomposition computation.

Notes
-----
For a Gaussian Schell-model source, the modes are Hermite-Gaussian with
analytically known eigenvalues. This provides an efficient representation
where typically only 5-20 modes capture most of the field energy.

References
----------
1. Starikov, A. & Wolf, E. "Coherent-mode representation of Gaussian
   Schell-model sources" JOSA A (1982)
2. Mandel, L. & Wolf, E. "Optical Coherence and Quantum Optics" (1995)
"""

from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from janssen.types import (
    CoherentModeSet,
    MutualIntensity,
    ScalarFloat,
    ScalarInteger,
    make_coherent_mode_set,
    make_mutual_intensity,
)

MAX_HERMITE_ORDER: int = 50


def _hermite_polynomial(
    order: Float[Array, " "], x: Float[Array, " hh ww"]
) -> Float[Array, " hh ww"]:
    """Compute physicist's Hermite polynomial H_n(x) using recurrence.

    Uses jax.lax.scan for autodiff-compatible iteration over orders.

    Parameters
    ----------
    order : Float[Array, " "]
        Polynomial order (will be cast to int internally).
    x : Float[Array, " hh ww"]
        Input coordinates.

    Returns
    -------
    h_n : Float[Array, " hh ww"]
        Hermite polynomial H_order(x).
    """
    h0: Float[Array, " hh ww"] = jnp.ones_like(x)
    h1: Float[Array, " hh ww"] = 2.0 * x

    def _scan_fn(
        carry: Tuple[Float[Array, " hh ww"], Float[Array, " hh ww"], int],
        _: None,
    ) -> Tuple[
        Tuple[Float[Array, " hh ww"], Float[Array, " hh ww"], int], None
    ]:
        h_prev2, h_prev1, k = carry
        h_curr: Float[Array, " hh ww"] = 2.0 * x * h_prev1 - 2.0 * k * h_prev2
        h_prev2_new: Float[Array, " hh ww"] = jnp.where(
            k < order, h_prev1, h_prev2
        )
        h_prev1_new: Float[Array, " hh ww"] = jnp.where(
            k < order, h_curr, h_prev1
        )
        return (h_prev2_new, h_prev1_new, k + 1), None

    (_, h_n, _), _ = jax.lax.scan(
        _scan_fn, (h0, h1, 1), None, length=MAX_HERMITE_ORDER
    )
    result: Float[Array, " hh ww"] = jnp.where(
        order == 0, h0, jnp.where(order == 1, h1, h_n)
    )
    return result


def _generate_mode_indices(max_order: int) -> Float[Array, " num_modes 2"]:
    """Generate mode indices (n, m) for HG modes up to max_order.

    Parameters
    ----------
    max_order : int
        Maximum mode order. Generates modes with n + m <= max_order.

    Returns
    -------
    mode_indices : Float[Array, " num_modes 2"]
        Array of (n, m) index pairs, ordered by total order then n.
    """
    mode_indices_list = []
    for total in range(max_order + 1):
        for n in range(total + 1):
            m = total - n
            mode_indices_list.append((n, m))
    mode_indices: Float[Array, " num_modes 2"] = jnp.array(
        mode_indices_list, dtype=jnp.float64
    )
    return mode_indices


@partial(jax.jit, static_argnums=(2, 3))
def _hermite_gaussian_modes_impl(
    dx: Float[Array, " "],
    beam_waist: Float[Array, " "],
    hh: int,
    ww: int,
    mode_indices: Float[Array, " num_modes 2"],
) -> Complex[Array, " num_modes hh ww"]:
    """JIT-compiled Hermite-Gaussian mode generation.

    Pure JAX implementation with static grid dimensions for efficient
    JIT compilation. Uses vmap for vectorized mode generation.

    Parameters
    ----------
    dx : Float[Array, " "]
        Pixel spacing in meters.
    beam_waist : Float[Array, " "]
        Gaussian beam waist parameter w0 in meters.
    hh : int
        Grid height in pixels (static).
    ww : int
        Grid width in pixels (static).
    mode_indices : Float[Array, " num_modes 2"]
        Array of (n, m) index pairs for modes to generate.

    Returns
    -------
    modes : Complex[Array, " num_modes hh ww"]
        Normalized Hermite-Gaussian mode fields.
    """
    w: Float[Array, " "] = beam_waist
    y: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * dx
    x: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)
    x_norm: Float[Array, " hh ww"] = xx * jnp.sqrt(2.0) / w
    y_norm: Float[Array, " hh ww"] = yy * jnp.sqrt(2.0) / w
    gaussian: Float[Array, " hh ww"] = jnp.exp(-(xx**2 + yy**2) / w**2)

    def _compute_single_mode_int(
        indices: Float[Array, " 2"],
    ) -> Complex[Array, " hh ww"]:
        n: Float[Array, " "] = indices[0]
        m: Float[Array, " "] = indices[1]
        h_n: Float[Array, " hh ww"] = _hermite_polynomial(n, x_norm)
        h_m: Float[Array, " hh ww"] = _hermite_polynomial(m, y_norm)
        mode: Float[Array, " hh ww"] = h_n * h_m * gaussian
        mode_energy: Float[Array, " "] = jnp.sum(jnp.abs(mode) ** 2)
        normalized: Complex[Array, " hh ww"] = (
            mode / jnp.sqrt(mode_energy + 1e-20)
        ).astype(jnp.complex128)
        return normalized

    modes: Complex[Array, " num_modes hh ww"] = jax.vmap(
        _compute_single_mode_int
    )(mode_indices)
    return modes


@partial(jax.jit, static_argnums=(0,))
def thermal_mode_weights(num_modes: int) -> Float[Array, " num_modes"]:
    """Generate thermal (exponentially decreasing) mode weights.

    Parameters
    ----------
    num_modes : int
        Number of modes (static).

    Returns
    -------
    weights : Float[Array, " num_modes"]
        Normalized thermal distribution weights.
    """
    mode_numbers: Float[Array, " num_modes"] = jnp.arange(
        num_modes, dtype=jnp.float64
    )
    unnormalized: Float[Array, " num_modes"] = jnp.exp(-mode_numbers / 2.0)
    weights: Float[Array, " num_modes"] = unnormalized / jnp.sum(unnormalized)
    return weights


@jaxtyped(typechecker=beartype)
def hermite_gaussian_modes(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Tuple[ScalarInteger, ScalarInteger],
    beam_waist: ScalarFloat,
    max_order: ScalarInteger,
    mode_weights: Float[Array, " n"],
) -> CoherentModeSet:
    """Generate Hermite-Gaussian coherent mode set.

    Creates a set of 2D Hermite-Gaussian modes HG_nm up to specified order.
    Useful for modeling laser sources with partial coherence or multimode
    laser beams.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength in meters.
    dx : ScalarFloat
        Pixel spacing in meters.
    grid_size : Tuple[ScalarInteger, ScalarInteger]
        (height, width) of the grid in pixels.
    beam_waist : ScalarFloat
        1/e^2 intensity radius (waist) in meters.
    max_order : ScalarInteger
        Maximum mode order. Generates modes with n + m <= max_order.
    mode_weights : Float[Array, " n"]
        Weights for each mode. Use thermal_mode_weights(num_modes) for
        default thermal distribution (exponentially decreasing).

    Returns
    -------
    mode_set : CoherentModeSet
        Coherent mode set with Hermite-Gaussian modes.

    Notes
    -----
    The 2D Hermite-Gaussian modes are:
        HG_nm(x, y) = H_n(sqrt(2)*x/w) * H_m(sqrt(2)*y/w)
                      * exp(-(x^2+y^2)/w^2)

    where H_n is the physicist's Hermite polynomial.

    Modes are indexed in order: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...

    For thermal distribution weights, use:
        num_modes = (max_order + 1) * (max_order + 2) // 2
        weights = thermal_mode_weights(num_modes)
    """
    hh: int = int(grid_size[0])
    ww: int = int(grid_size[1])
    max_ord: int = int(max_order)
    dx_arr: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    w_arr: Float[Array, " "] = jnp.asarray(beam_waist, dtype=jnp.float64)

    mode_indices: Float[Array, " num_modes 2"] = _generate_mode_indices(
        max_ord
    )
    modes = _hermite_gaussian_modes_impl(dx_arr, w_arr, hh, ww, mode_indices)

    weights: Float[Array, " num_modes"] = jnp.asarray(
        mode_weights, dtype=jnp.float64
    )
    weights = weights / jnp.sum(weights)

    return make_coherent_mode_set(
        modes=modes,
        weights=weights,
        wavelength=wavelength,
        dx=dx,
        z_position=0.0,
        polarization=False,
    )


@partial(jax.jit, static_argnums=(3, 4, 5))
def _gaussian_schell_model_modes_impl(
    dx: Float[Array, " "],
    beam_width: Float[Array, " "],
    coherence_width: Float[Array, " "],
    hh: int,
    ww: int,
    n_modes: int,
) -> Tuple[Complex[Array, " num_modes hh ww"], Float[Array, " num_modes"]]:
    """JIT-compiled Gaussian Schell-model mode generation.

    Pure JAX implementation with static grid dimensions for efficient
    JIT compilation. Use this directly in pure JAX workflows.

    Parameters
    ----------
    dx : Float[Array, " "]
        Pixel spacing in meters.
    beam_width : Float[Array, " "]
        Source beam width (sigma_i) in meters.
    coherence_width : Float[Array, " "]
        Spatial coherence width (sigma_mu) in meters.
    hh : int
        Grid height in pixels (static).
    ww : int
        Grid width in pixels (static).
    n_modes : int
        Number of modes to generate (static).

    Returns
    -------
    modes : Complex[Array, " num_modes hh ww"]
        Normalized Hermite-Gaussian mode fields.
    eigenvalues : Float[Array, " num_modes"]
        Analytical eigenvalues (mode weights).
    """
    sigma_i: Float[Array, " "] = beam_width
    sigma_mu: Float[Array, " "] = coherence_width

    beam_to_coherence_ratio: Float[Array, " "] = 2.0 * sigma_i / sigma_mu
    gsm_parameter: Float[Array, " "] = 1.0 / jnp.sqrt(
        1.0 + beam_to_coherence_ratio**2
    )

    effective_width: Float[Array, " "] = sigma_i * gsm_parameter
    mode_width: Float[Array, " "] = jnp.sqrt(sigma_i * effective_width)

    n_arr: Float[Array, " num_modes"] = jnp.arange(n_modes, dtype=jnp.float64)
    q: Float[Array, " "] = (1.0 - gsm_parameter) / (1.0 + gsm_parameter)
    eigenvalues: Float[Array, " num_modes"] = q**n_arr

    y: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * dx
    x: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)
    x_norm: Float[Array, " hh ww"] = xx * jnp.sqrt(2.0) / mode_width
    gaussian: Float[Array, " hh ww"] = jnp.exp(
        -(xx**2 + yy**2) / mode_width**2
    )

    def _compute_single_mode_int(
        n: Float[Array, " "],
    ) -> Complex[Array, " hh ww"]:
        h_n: Float[Array, " hh ww"] = _hermite_polynomial(n, x_norm)
        mode: Float[Array, " hh ww"] = h_n * gaussian
        mode_energy: Float[Array, " "] = jnp.sum(jnp.abs(mode) ** 2)
        normalized: Complex[Array, " hh ww"] = (
            mode / jnp.sqrt(mode_energy + 1e-20)
        ).astype(jnp.complex128)
        return normalized

    modes: Complex[Array, " num_modes hh ww"] = jax.vmap(
        _compute_single_mode_int
    )(n_arr)
    return modes, eigenvalues


@jaxtyped(typechecker=beartype)
def gaussian_schell_model_modes(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Tuple[ScalarInteger, ScalarInteger],
    beam_width: ScalarFloat,
    coherence_width: ScalarFloat,
    num_modes: ScalarInteger,
) -> CoherentModeSet:
    """Generate modes for Gaussian Schell-model source.

    The Gaussian Schell-model (GSM) source has Gaussian intensity and
    Gaussian coherence functions. Its coherent mode decomposition has
    analytical solutions with Hermite-Gaussian modes.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength in meters.
    dx : ScalarFloat
        Pixel spacing in meters.
    grid_size : Tuple[ScalarInteger, ScalarInteger]
        (height, width) of the grid in pixels.
    beam_width : ScalarFloat
        1/e^2 intensity width (sigma_I) in meters.
    coherence_width : ScalarFloat
        Coherence width (sigma_mu) in meters. The 1/e width of the
        complex degree of coherence.
    num_modes : ScalarInteger
        Number of modes to generate.

    Returns
    -------
    mode_set : CoherentModeSet
        Coherent mode set for GSM source.

    Notes
    -----
    For a GSM source, the analytical eigenvalues are:
        lambda_n = ((1-a)/(1+a)) * (a/(1+a))^n

    where a = 1 / sqrt(1 + (2*sigma_I/sigma_mu)^2).

    The mode width is:
        w_mode = sqrt(sigma_I * sigma_eff)

    where sigma_eff = sigma_I / sqrt(1 + (2*sigma_I/sigma_mu)^2).

    The effective number of modes for a GSM source is:
        M_eff = (1 + (2*sigma_I/sigma_mu)^2)^(1/2)

    References
    ----------
    Starikov, A. & Wolf, E. JOSA A 72, 923-928 (1982)
    """
    hh: int = int(grid_size[0])
    ww: int = int(grid_size[1])
    n_modes: int = int(num_modes)
    dx_arr: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    bw_arr: Float[Array, " "] = jnp.asarray(beam_width, dtype=jnp.float64)
    cw_arr: Float[Array, " "] = jnp.asarray(coherence_width, dtype=jnp.float64)
    modes, eigenvalues = _gaussian_schell_model_modes_impl(
        dx_arr, bw_arr, cw_arr, hh, ww, n_modes
    )
    mode_set: CoherentModeSet = make_coherent_mode_set(
        modes=modes,
        weights=eigenvalues,
        wavelength=wavelength,
        dx=dx,
        z_position=0.0,
        polarization=False,
        normalize_weights=True,
    )
    return mode_set


@partial(jax.jit, static_argnums=(1, 2, 3))
def _eigenmode_decomposition_impl(
    j_matrix: Complex[Array, " hh ww hh ww"],
    hh: int,
    ww: int,
    n_modes: int,
) -> Tuple[Complex[Array, " num_modes hh ww"], Float[Array, " num_modes"]]:
    """JIT-compiled eigenmode decomposition computation.

    Pure JAX implementation with static grid dimensions for efficient
    JIT compilation. Use this directly in pure JAX workflows.

    Parameters
    ----------
    j_matrix : Complex[Array, " hh ww hh ww"]
        Mutual intensity matrix J(r1, r2).
    hh : int
        Grid height in pixels (static).
    ww : int
        Grid width in pixels (static).
    n_modes : int
        Number of modes to extract (static).

    Returns
    -------
    modes : Complex[Array, " num_modes hh ww"]
        Coherent mode fields (eigenvectors reshaped to 2D).
    eigenvalues : Float[Array, " num_modes"]
        Eigenvalues (mode weights), sorted descending.
    """
    n_pixels: int = hh * ww
    j_2d: Complex[Array, " n n"] = j_matrix.reshape(n_pixels, n_pixels)
    eigenvalues: Float[Array, " n"]
    eigenvectors: Complex[Array, " n n"]
    eigenvalues, eigenvectors = jax.scipy.linalg.eigh(j_2d)
    eigenvalues_descending = eigenvalues[::-1]
    eigenvectors_descending = eigenvectors[:, ::-1]
    top_eigenvalues: Float[Array, " num_modes"] = eigenvalues_descending[
        :n_modes
    ]
    top_eigenvectors: Complex[Array, " n num_modes"] = eigenvectors_descending[
        :, :n_modes
    ]
    clipped_eigenvalues = jnp.maximum(top_eigenvalues, 0.0)
    modes: Complex[Array, " num_modes hh ww"] = top_eigenvectors.T.reshape(
        n_modes, hh, ww
    )
    return modes, clipped_eigenvalues


@jaxtyped(typechecker=beartype)
def eigenmode_decomposition(
    mutual_intensity: MutualIntensity,
    num_modes: ScalarInteger,
) -> CoherentModeSet:
    """Decompose mutual intensity into coherent modes via eigendecomposition.

    Uses truncated eigendecomposition of the mutual intensity matrix
    J(r1, r2) to find the dominant coherent modes and their weights.

    Parameters
    ----------
    mutual_intensity : MutualIntensity
        Full mutual intensity representation.
    num_modes : ScalarInteger
        Number of modes to extract (ordered by decreasing eigenvalue).

    Returns
    -------
    mode_set : CoherentModeSet
        Coherent mode set with the dominant modes.

    Notes
    -----
    The mutual intensity is a Hermitian positive semi-definite operator,
    so its eigenvalues are real and non-negative. The eigenvectors form
    an orthonormal basis for the field.

    Warning: This function reshapes the 4D mutual intensity into a 2D
    matrix for eigendecomposition. For an N×N grid, this creates an
    N²×N² matrix, which can be very large.

    For efficiency, this uses jax.scipy.linalg.eigh which is optimized
    for Hermitian matrices.
    """
    j_matrix: Complex[Array, " hh ww hh ww"] = mutual_intensity.j_matrix
    hh: int = int(j_matrix.shape[0])
    ww: int = int(j_matrix.shape[1])
    n_modes: int = int(num_modes)
    modes: Complex[Array, " num_modes hh ww"]
    clipped_eigenvalues: Float[Array, " num_modes"]
    modes, clipped_eigenvalues = _eigenmode_decomposition_impl(
        j_matrix, hh, ww, n_modes
    )
    mode_set: CoherentModeSet = make_coherent_mode_set(
        modes=modes,
        weights=clipped_eigenvalues,
        wavelength=mutual_intensity.wavelength,
        dx=mutual_intensity.dx,
        z_position=mutual_intensity.z_position,
        polarization=False,
        normalize_weights=True,
    )
    return mode_set


@jaxtyped(typechecker=beartype)
def effective_mode_count(
    mode_set: CoherentModeSet,
) -> Float[Array, " "]:
    """Calculate effective number of modes (participation ratio).

    The effective mode count quantifies the degree of partial coherence:
    - M_eff = 1: fully coherent (single mode dominates)
    - M_eff > 1: partially coherent (multiple modes contribute)
    - M_eff = N: maximally incoherent (all modes equal weight)

    Parameters
    ----------
    mode_set : CoherentModeSet
        Coherent mode set containing modal weights.

    Returns
    -------
    m_eff : Float[Array, " "]
        Effective number of modes.

    Notes
    -----
    The participation ratio is defined as:
        M_eff = (sum w_n)^2 / sum(w_n^2) = 1 / sum(p_n^2)

    where p_n = w_n / sum(w_n) are normalized probabilities.

    This is also known as the inverse participation ratio (IPR) and
    appears in many contexts in physics (localization, entropy, etc.).
    """
    weights: Float[Array, " n"] = mode_set.weights
    weights_sum: Float[Array, " "] = jnp.sum(weights)
    weights_sq_sum: Float[Array, " "] = jnp.sum(weights**2)
    m_eff: Float[Array, " "] = weights_sum**2 / (weights_sq_sum + 1e-20)
    return m_eff


@jaxtyped(typechecker=beartype)
def modes_from_wavefront(
    field: Complex[Array, " hh ww"],
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    z_position: ScalarFloat = 0.0,
) -> CoherentModeSet:
    """Create a single-mode CoherentModeSet from a fully coherent wavefront.

    Convenience function to wrap a coherent field as a CoherentModeSet
    for use with partial coherence propagation functions.

    Parameters
    ----------
    field : Complex[Array, " hh ww"]
        Complex field amplitude.
    wavelength : ScalarFloat
        Wavelength in meters.
    dx : ScalarFloat
        Pixel spacing in meters.
    z_position : ScalarFloat
        Axial position in meters.

    Returns
    -------
    mode_set : CoherentModeSet
        Single-mode coherent mode set (fully coherent).
    """
    modes: Complex[Array, " 1 hh ww"] = field[jnp.newaxis, ...]
    weights: Float[Array, " 1"] = jnp.array([1.0])
    mode_set: CoherentModeSet = make_coherent_mode_set(
        modes=modes,
        weights=weights,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
        polarization=False,
    )
    return mode_set


@jaxtyped(typechecker=beartype)
def mutual_intensity_from_modes(
    mode_set: CoherentModeSet,
) -> MutualIntensity:
    """Reconstruct mutual intensity J(r1, r2) from coherent modes.

    Computes the full mutual intensity matrix from the mode decomposition:
        J(r1, r2) = sum_n weights[n] * modes[n]^*(r1) * modes[n](r2)

    Parameters
    ----------
    mode_set : CoherentModeSet
        Coherent mode set.

    Returns
    -------
    mutual_intensity : MutualIntensity
        Full mutual intensity representation.

    Warnings
    --------
    This creates an O(N^4) array from an O(M*N^2) representation.
    Use with caution for large grids.
    """
    modes: Complex[Array, " num_modes hh ww"] = mode_set.modes
    weights: Float[Array, " num_modes"] = mode_set.weights

    def _compute_j_term_int(
        mode: Complex[Array, " hh ww"], weight: Float[Array, " "]
    ) -> Complex[Array, " hh ww hh ww"]:
        """Compute weighted outer product for single mode."""
        j_n: Complex[Array, " hh ww hh ww"] = weight * jnp.einsum(
            "ij,kl->ijkl", jnp.conj(mode), mode
        )
        return j_n

    all_j_terms: Complex[Array, " num_modes hh ww hh ww"] = jax.vmap(
        _compute_j_term_int
    )(modes, weights)
    j_matrix: Complex[Array, " hh ww hh ww"] = jnp.sum(all_j_terms, axis=0)
    mutual_intensity: MutualIntensity = make_mutual_intensity(
        j_matrix=j_matrix,
        wavelength=mode_set.wavelength,
        dx=mode_set.dx,
        z_position=mode_set.z_position,
    )
    return mutual_intensity
