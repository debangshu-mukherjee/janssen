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
    w: Float[Array, " "] = jnp.asarray(beam_waist, dtype=jnp.float64)
    max_ord: int = int(max_order)

    # Create coordinate grids
    y: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * dx
    x: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)

    # Normalized coordinates
    x_norm: Float[Array, " hh ww"] = xx * jnp.sqrt(2.0) / w
    y_norm: Float[Array, " hh ww"] = yy * jnp.sqrt(2.0) / w

    # Gaussian envelope
    gaussian: Float[Array, " hh ww"] = jnp.exp(-(xx**2 + yy**2) / w**2)

    # Count modes: sum of (n+m <= max_order) = (max_order+1)*(max_order+2)/2
    num_modes: int = (max_ord + 1) * (max_ord + 2) // 2

    # Generate mode indices
    mode_indices: list = []
    for total in range(max_ord + 1):
        for n in range(total + 1):
            m = total - n
            mode_indices.append((n, m))

    def _hermite_polynomial(
        order: int, x: Float[Array, " hh ww"]
    ) -> Float[Array, " hh ww"]:
        """Compute physicist's Hermite polynomial H_n(x)."""
        if order == 0:
            return jnp.ones_like(x)
        elif order == 1:
            return 2.0 * x
        else:
            h_prev2 = jnp.ones_like(x)
            h_prev1 = 2.0 * x
            for k in range(2, order + 1):
                h_curr = 2.0 * x * h_prev1 - 2.0 * (k - 1) * h_prev2
                h_prev2 = h_prev1
                h_prev1 = h_curr
            return h_prev1

    # Generate all modes
    modes_list = []
    for n, m in mode_indices:
        h_n = _hermite_polynomial(n, x_norm)
        h_m = _hermite_polynomial(m, y_norm)
        mode = h_n * h_m * gaussian

        # Normalize to unit energy
        energy = jnp.sum(jnp.abs(mode) ** 2)
        mode = mode / jnp.sqrt(energy + 1e-20)
        modes_list.append(mode)

    modes: Complex[Array, " num_modes hh ww"] = jnp.stack(
        modes_list, axis=0
    ).astype(jnp.complex128)

    # Set weights
    if mode_weights is None:
        # Thermal distribution: exponentially decreasing
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

    sigma_i: Float[Array, " "] = jnp.asarray(beam_width, dtype=jnp.float64)
    sigma_mu: Float[Array, " "] = jnp.asarray(
        coherence_width, dtype=jnp.float64
    )

    # GSM parameter
    ratio: Float[Array, " "] = 2.0 * sigma_i / sigma_mu
    a: Float[Array, " "] = 1.0 / jnp.sqrt(1.0 + ratio**2)

    # Mode width
    sigma_eff: Float[Array, " "] = sigma_i * a
    w_mode: Float[Array, " "] = jnp.sqrt(sigma_i * sigma_eff)

    # Eigenvalues (analytical formula)
    n_arr: Float[Array, " num_modes"] = jnp.arange(n_modes, dtype=jnp.float64)
    eigenvalues: Float[Array, " num_modes"] = ((1.0 - a) / (1.0 + a)) * (
        a / (1.0 + a)
    ) ** n_arr

    # Create coordinate grids
    y: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * dx
    x: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)

    # For 1D GSM, modes are 1D Hermite-Gaussians
    # For 2D, we use product of 1D modes
    # Here we generate 1D modes and form 2D products
    x_norm: Float[Array, " hh ww"] = xx * jnp.sqrt(2.0) / w_mode
    y_norm: Float[Array, " hh ww"] = yy * jnp.sqrt(2.0) / w_mode

    gaussian: Float[Array, " hh ww"] = jnp.exp(-(xx**2 + yy**2) / w_mode**2)

    def _hermite_polynomial_jax(
        order: Int[Array, " "], x: Float[Array, " hh ww"]
    ) -> Float[Array, " hh ww"]:
        """JAX-compatible Hermite polynomial using lax.fori_loop."""

        def body_fn(
            k: int,
            carry: Tuple[Float[Array, " hh ww"], Float[Array, " hh ww"]],
        ) -> Tuple[Float[Array, " hh ww"], Float[Array, " hh ww"]]:
            h_km1, h_km2 = carry
            h_k = 2.0 * x * h_km1 - 2.0 * (k - 1) * h_km2
            return (h_k, h_km1)

        h_0 = jnp.ones_like(x)
        h_1 = 2.0 * x

        result = jax.lax.cond(
            order == 0,
            lambda: h_0,
            lambda: jax.lax.cond(
                order == 1,
                lambda: h_1,
                lambda: jax.lax.fori_loop(2, order + 1, body_fn, (h_1, h_0))[
                    0
                ],
            ),
        )
        return result

    # Generate modes using separable Hermite-Gaussians
    # For simplicity, 1D mode structure: mode_n = H_n(x) * H_0(y) * gaussian
    # A more complete 2D GSM would have different x and y mode indices

    def generate_mode(n: int) -> Complex[Array, " hh ww"]:
        h_n = _hermite_polynomial_jax(n, x_norm)
        # For 2D isotropic GSM, we use symmetric modes
        # Here using 1D modes in x direction for simplicity
        mode = h_n * gaussian

        # Normalize
        energy = jnp.sum(jnp.abs(mode) ** 2)
        return (mode / jnp.sqrt(energy + 1e-20)).astype(jnp.complex128)

    # Generate all modes
    modes: Complex[Array, " num_modes hh ww"] = jnp.stack(
        [generate_mode(n) for n in range(n_modes)], axis=0
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
    hh: int = j_matrix.shape[0]
    ww: int = j_matrix.shape[1]
    n_modes: int = int(num_modes)

    # Reshape 4D mutual intensity to 2D matrix: (hh*ww, hh*ww)
    n_pixels: int = hh * ww
    j_2d: Complex[Array, " n n"] = j_matrix.reshape(n_pixels, n_pixels)

    # Eigendecomposition of Hermitian matrix
    # eigenvalues are sorted in ascending order by eigh
    eigenvalues: Float[Array, " n"]
    eigenvectors: Complex[Array, " n n"]
    eigenvalues, eigenvectors = jax.scipy.linalg.eigh(j_2d)

    # Reverse to get descending order (largest eigenvalues first)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Extract top num_modes modes
    top_eigenvalues: Float[Array, " num_modes"] = eigenvalues[:n_modes]
    top_eigenvectors: Complex[Array, " n num_modes"] = eigenvectors[
        :, :n_modes
    ]

    # Clip small negative eigenvalues (numerical artifacts)
    top_eigenvalues = jnp.maximum(top_eigenvalues, 0.0)

    # Reshape eigenvectors back to 2D spatial modes
    # Each column is a flattened mode
    modes: Complex[Array, " num_modes hh ww"] = top_eigenvectors.T.reshape(
        n_modes, hh, ww
    )

    return make_coherent_mode_set(
        modes=modes,
        weights=top_eigenvalues,
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

    hh: int = modes.shape[1]
    ww: int = modes.shape[2]

    # Compute J(r1, r2) = sum_n w_n * mode_n^*(r1) * mode_n(r2)
    # Reshape modes for outer product computation
    # modes_conj[n, i, j] * modes[n, k, l] -> J[i, j, k, l]

    def compute_j_term(
        n: int,
    ) -> Complex[Array, " hh ww hh ww"]:
        mode_n = modes[n]
        weight_n = weights[n]
        # Outer product: conj(mode_n)[i,j] * mode_n[k,l]
        j_n = weight_n * jnp.einsum("ij,kl->ijkl", jnp.conj(mode_n), mode_n)
        return j_n

    # Sum over all modes
    j_matrix: Complex[Array, " hh ww hh ww"] = jnp.sum(
        jnp.stack([compute_j_term(n) for n in range(modes.shape[0])], axis=0),
        axis=0,
    )

    return make_mutual_intensity(
        j_matrix=j_matrix,
        wavelength=mode_set.wavelength,
        dx=mode_set.dx,
        z_position=mode_set.z_position,
    )
