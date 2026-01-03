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
    Generate Hermite-Gaussian coherent mode set
gaussian_schell_model_modes : function
    Generate modes for Gaussian Schell-model source
eigenmode_decomposition : function
    Decompose mutual intensity into coherent modes
effective_mode_count : function
    Calculate effective number of modes (participation ratio)
orthonormalize_modes : function
    Gram-Schmidt orthonormalization of mode set

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
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from janssen.utils import (
    CoherentModeSet,
    MutualIntensity,
    ScalarFloat,
    ScalarInteger,
    make_coherent_mode_set,
)


def _hermite_polynomial_static(
    order: int, x: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Compute physicist's Hermite polynomial H_n(x) with static order."""
    if order == 0:
        return jnp.ones_like(x)
    if order == 1:
        return 2.0 * x
    h_prev2 = jnp.ones_like(x)
    h_prev1 = 2.0 * x
    for k in range(2, order + 1):
        h_curr = 2.0 * x * h_prev1 - 2.0 * (k - 1) * h_prev2
        h_prev2 = h_prev1
        h_prev1 = h_curr
    return h_prev1


def _generate_mode_indices(max_order: int) -> list:
    """Generate mode indices (n, m) for HG modes up to max_order."""
    mode_indices = []
    for total in range(max_order + 1):
        for n in range(total + 1):
            m = total - n
            mode_indices.append((n, m))
    return mode_indices


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _hermite_gaussian_modes_impl(
    dx: Float[Array, " "],
    beam_waist: Float[Array, " "],
    hh: int,
    ww: int,
    max_order: int,
    mode_indices_tuple: Tuple[Tuple[int, int], ...],
) -> Complex[Array, " num_modes hh ww"]:
    """JIT-compiled implementation of hermite_gaussian_modes."""
    w: Float[Array, " "] = beam_waist
    y: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * dx
    x: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)

    x_norm: Float[Array, " hh ww"] = xx * jnp.sqrt(2.0) / w
    y_norm: Float[Array, " hh ww"] = yy * jnp.sqrt(2.0) / w

    gaussian: Float[Array, " hh ww"] = jnp.exp(-(xx**2 + yy**2) / w**2)

    modes_list = []
    for n, m in mode_indices_tuple:
        h_n = _hermite_polynomial_static(n, x_norm)
        h_m = _hermite_polynomial_static(m, y_norm)
        mode = h_n * h_m * gaussian

        mode_energy = jnp.sum(jnp.abs(mode) ** 2)
        mode = mode / jnp.sqrt(mode_energy + 1e-20)
        modes_list.append(mode)

    modes: Complex[Array, " num_modes hh ww"] = jnp.stack(
        modes_list, axis=0
    ).astype(jnp.complex128)
    return modes


@jaxtyped(typechecker=beartype)
def hermite_gaussian_modes(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Tuple[ScalarInteger, ScalarInteger],
    beam_waist: ScalarFloat,
    max_order: ScalarInteger,
    mode_weights: Optional[Float[Array, " n"]] = None,
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
    mode_weights : Optional[Float[Array, " n"]]
        Weights for each mode. If None, uses thermal distribution
        (exponentially decreasing with mode number).

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
    """
    hh: int = int(grid_size[0])
    ww: int = int(grid_size[1])
    max_ord: int = int(max_order)
    dx_arr: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    w_arr: Float[Array, " "] = jnp.asarray(beam_waist, dtype=jnp.float64)

    num_modes: int = (max_ord + 1) * (max_ord + 2) // 2
    mode_indices = _generate_mode_indices(max_ord)
    mode_indices_tuple = tuple(mode_indices)

    modes = _hermite_gaussian_modes_impl(
        dx_arr, w_arr, hh, ww, max_ord, mode_indices_tuple
    )

    if mode_weights is None:
        mode_numbers = jnp.arange(num_modes)
        weights: Float[Array, " num_modes"] = jnp.exp(-mode_numbers / 2.0)
        weights = weights / jnp.sum(weights)
    else:
        weights = jnp.asarray(mode_weights, dtype=jnp.float64)
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
    """JIT-compiled implementation of gaussian_schell_model_modes."""
    sigma_i: Float[Array, " "] = beam_width
    sigma_mu: Float[Array, " "] = coherence_width

    beam_to_coherence_ratio: Float[Array, " "] = 2.0 * sigma_i / sigma_mu
    gsm_parameter: Float[Array, " "] = 1.0 / jnp.sqrt(
        1.0 + beam_to_coherence_ratio**2
    )

    effective_width: Float[Array, " "] = sigma_i * gsm_parameter
    mode_width: Float[Array, " "] = jnp.sqrt(sigma_i * effective_width)

    n_arr: Float[Array, " num_modes"] = jnp.arange(n_modes, dtype=jnp.float64)
    eigenvalues: Float[Array, " num_modes"] = (
        (1.0 - gsm_parameter) / (1.0 + gsm_parameter)
    ) * (gsm_parameter / (1.0 + gsm_parameter)) ** n_arr

    y: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * dx
    x: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)

    x_norm: Float[Array, " hh ww"] = xx * jnp.sqrt(2.0) / mode_width

    gaussian: Float[Array, " hh ww"] = jnp.exp(
        -(xx**2 + yy**2) / mode_width**2
    )

    modes_list = []
    for n in range(n_modes):
        h_n = _hermite_polynomial_static(n, x_norm)
        mode = h_n * gaussian

        mode_energy = jnp.sum(jnp.abs(mode) ** 2)
        modes_list.append(
            (mode / jnp.sqrt(mode_energy + 1e-20)).astype(jnp.complex128)
        )

    modes: Complex[Array, " num_modes hh ww"] = jnp.stack(modes_list, axis=0)
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

    return make_coherent_mode_set(
        modes=modes,
        weights=eigenvalues,
        wavelength=wavelength,
        dx=dx,
        z_position=0.0,
        polarization=False,
        normalize_weights=True,
    )


@partial(jax.jit, static_argnums=(1, 2, 3))
def _eigenmode_decomposition_impl(
    j_matrix: Complex[Array, " hh ww hh ww"],
    hh: int,
    ww: int,
    n_modes: int,
) -> Tuple[Complex[Array, " num_modes hh ww"], Float[Array, " num_modes"]]:
    """JIT-compiled implementation of eigenmode_decomposition."""
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

    modes, clipped_eigenvalues = _eigenmode_decomposition_impl(
        j_matrix, hh, ww, n_modes
    )

    return make_coherent_mode_set(
        modes=modes,
        weights=clipped_eigenvalues,
        wavelength=mutual_intensity.wavelength,
        dx=mutual_intensity.dx,
        z_position=mutual_intensity.z_position,
        polarization=False,
        normalize_weights=True,
    )


@jaxtyped(typechecker=beartype)
def effective_mode_count(
    weights: Float[Array, " n"],
) -> Float[Array, " "]:
    """Calculate effective number of modes (participation ratio).

    The effective mode count quantifies the degree of partial coherence:
    - M_eff = 1: fully coherent (single mode dominates)
    - M_eff > 1: partially coherent (multiple modes contribute)
    - M_eff = N: maximally incoherent (all modes equal weight)

    Parameters
    ----------
    weights : Float[Array, " n"]
        Modal weights (eigenvalues).

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

    return make_coherent_mode_set(
        modes=modes,
        weights=weights,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
        polarization=False,
    )


@partial(jax.jit, static_argnums=(2,))
def _mutual_intensity_from_modes_impl(
    modes: Complex[Array, " num_modes hh ww"],
    weights: Float[Array, " num_modes"],
    n_modes: int,
) -> Complex[Array, " hh ww hh ww"]:
    """JIT-compiled implementation of mutual_intensity_from_modes."""

    def compute_j_term(n: int) -> Complex[Array, " hh ww hh ww"]:
        mode_n = modes[n]
        weight_n = weights[n]
        j_n = weight_n * jnp.einsum("ij,kl->ijkl", jnp.conj(mode_n), mode_n)
        return j_n

    j_matrix: Complex[Array, " hh ww hh ww"] = jnp.sum(
        jnp.stack([compute_j_term(n) for n in range(n_modes)], axis=0),
        axis=0,
    )
    return j_matrix


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
    from janssen.utils import make_mutual_intensity

    modes: Complex[Array, " num_modes hh ww"] = mode_set.modes
    weights: Float[Array, " num_modes"] = mode_set.weights
    n_modes: int = int(modes.shape[0])

    j_matrix = _mutual_intensity_from_modes_impl(modes, weights, n_modes)

    return make_mutual_intensity(
        j_matrix=j_matrix,
        wavelength=mode_set.wavelength,
        dx=mode_set.dx,
        z_position=mode_set.z_position,
    )
