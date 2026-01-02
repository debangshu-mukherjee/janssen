"""Spatial coherence functions.

Extended Summary
----------------
This module provides functions for modeling and manipulating spatial
coherence in optical fields. Spatial coherence describes the correlation
between the electric field at two different spatial points.

The key concept is the **mutual intensity** J(r1, r2) = <E*(r1) E(r2)>,
which describes the spatial coherence structure. The normalized form is
the **complex degree of coherence** mu(r1, r2) = J(r1,r2) / sqrt(I1 * I2).

The **van Cittert-Zernike theorem** states that for an extended incoherent
source, the mutual intensity in the far field is the Fourier transform of
the source intensity distribution.

Routine Listings
----------------
gaussian_coherence_kernel : function
    Generate Gaussian spatial coherence kernel mu(Delta r)
jinc_coherence_kernel : function
    Generate jinc coherence kernel from circular incoherent source
rectangular_coherence_kernel : function
    Generate sinc coherence kernel from rectangular incoherent source
coherence_width_from_source : function
    Calculate coherence width from source size using van Cittert-Zernike
apply_partial_coherence : function
    Apply partial spatial coherence to a fully coherent wavefront
complex_degree_of_coherence : function
    Compute normalized coherence mu(r1, r2) from mutual intensity

Notes
-----
The van Cittert-Zernike theorem gives the coherence width as:
    sigma_c ~ lambda * z / D_source

where lambda is wavelength, z is propagation distance, and D_source is
the source diameter.

References
----------
1. Born, M. & Wolf, E. "Principles of Optics" Chapter 10
2. Mandel, L. & Wolf, E. "Optical Coherence and Quantum Optics" (1995)
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from janssen.utils import ScalarFloat, ScalarInteger


@jaxtyped(typechecker=beartype)
def gaussian_coherence_kernel(
    grid_size: Tuple[ScalarInteger, ScalarInteger],
    dx: ScalarFloat,
    coherence_width: ScalarFloat,
) -> Float[Array, " hh ww"]:
    """Generate Gaussian spatial coherence kernel mu(Delta r).

    The Gaussian coherence kernel models the spatial coherence of a
    Gaussian Schell-model source, which has Gaussian intensity and
    Gaussian coherence functions.

    Parameters
    ----------
    grid_size : Tuple[ScalarInteger, ScalarInteger]
        (height, width) of the grid in pixels.
    dx : ScalarFloat
        Pixel spacing in meters.
    coherence_width : ScalarFloat
        1/e width of coherence function (sigma_c) in meters.
        This is the transverse distance at which |mu| drops to 1/e.

    Returns
    -------
    kernel : Float[Array, " hh ww"]
        Coherence kernel centered at (0, 0), values in [0, 1].
        kernel[0, 0] = 1 (full coherence at zero separation).

    Notes
    -----
    The Gaussian coherence kernel is:
        mu(Delta_r) = exp(-|Delta_r|^2 / (2 * sigma_c^2))

    This is the most common model for partial spatial coherence because:
    - It arises naturally from Gaussian Schell-model sources
    - The coherent mode decomposition has analytical solutions
    - It provides smooth, physically reasonable coherence decay

    Examples
    --------
    >>> kernel = gaussian_coherence_kernel((256, 256), 1e-6, 20e-6)
    >>> kernel[128, 128]  # Center value = 1.0
    """
    hh: int = int(grid_size[0])
    ww: int = int(grid_size[1])
    yr: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * dx
    xr: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(xr, yr)
    r2: Float[Array, " hh ww"] = (xx**2) + (yy**2)
    sigma_c: Float[Array, " "] = jnp.asarray(
        coherence_width, dtype=jnp.float64
    )
    kernel_ft: Float[Array, " hh ww"] = jnp.exp(-r2 / (2.0 * sigma_c**2))
    kernel: Float[Array, " hh ww"] = jnp.fft.ifftshift(kernel_ft)
    return kernel


@jaxtyped(typechecker=beartype)
def jinc_coherence_kernel(
    grid_size: Tuple[ScalarInteger, ScalarInteger],
    dx: ScalarFloat,
    source_diameter: ScalarFloat,
    wavelength: ScalarFloat,
    propagation_distance: ScalarFloat,
) -> Float[Array, " hh ww"]:
    """Generate jinc coherence kernel from circular incoherent source.

    Implements the van Cittert-Zernike theorem for a circular uniform
    incoherent source. The resulting coherence function is a jinc
    (J1(x)/x, the 2D analog of sinc).

    Parameters
    ----------
    grid_size : Tuple[ScalarInteger, ScalarInteger]
        (height, width) of the grid in pixels.
    dx : ScalarFloat
        Pixel spacing in meters.
    source_diameter : ScalarFloat
        Diameter of the circular incoherent source in meters.
    wavelength : ScalarFloat
        Wavelength of light in meters.
    propagation_distance : ScalarFloat
        Distance from source to observation plane in meters.

    Returns
    -------
    kernel : Float[Array, " hh ww"]
        Coherence kernel centered at (0, 0), values in [-0.13, 1].
        First zero at Delta_r = 1.22 * lambda * z / D_source.

    Notes
    -----
    The van Cittert-Zernike theorem states that the complex degree of
    coherence in the far field equals the normalized Fourier transform
    of the source intensity:

        mu(Delta_r) = 2 * J1(pi * D * Delta_r / (lambda * z))
                      / (pi * D * Delta_r / (lambda * z))

    where J1 is the Bessel function of the first kind, order 1.

    The first zero of the jinc function occurs at:
        Delta_r_0 = 1.22 * lambda * z / D

    This defines the "coherence area" - points separated by more than
    this distance are essentially incoherent.
    """
    hh: int = int(grid_size[0])
    ww: int = int(grid_size[1])
    y: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * dx
    x: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)

    # Radial distance
    r: Float[Array, " hh ww"] = jnp.sqrt(xx**2 + yy**2)

    # Scaling factor from van Cittert-Zernike
    lam: Float[Array, " "] = jnp.asarray(wavelength, dtype=jnp.float64)
    z: Float[Array, " "] = jnp.asarray(propagation_distance, dtype=jnp.float64)
    d: Float[Array, " "] = jnp.asarray(source_diameter, dtype=jnp.float64)

    # Argument for jinc function: pi * D * r / (lambda * z)
    arg: Float[Array, " hh ww"] = jnp.pi * d * r / (lam * z)

    # Jinc function: 2 * J1(x) / x, with limit = 1 at x = 0
    # Use Taylor expansion near zero for numerical stability
    j1_arg: Float[Array, " hh ww"] = jnp.where(
        arg < 1e-10,
        arg / 2.0,  # J1(x) ~ x/2 for small x
        jax_bessel_j1(arg),
    )

    kernel: Float[Array, " hh ww"] = jnp.where(
        arg < 1e-10,
        1.0,  # jinc(0) = 1
        2.0 * j1_arg / arg,
    )

    # Shift to have maximum at [0, 0] for convolution
    kernel = jnp.fft.ifftshift(kernel)

    return kernel


@jaxtyped(typechecker=beartype)
def rectangular_coherence_kernel(
    grid_size: Tuple[ScalarInteger, ScalarInteger],
    dx: ScalarFloat,
    source_width_x: ScalarFloat,
    source_width_y: ScalarFloat,
    wavelength: ScalarFloat,
    propagation_distance: ScalarFloat,
) -> Float[Array, " hh ww"]:
    """Generate sinc coherence kernel from rectangular incoherent source.

    Implements the van Cittert-Zernike theorem for a rectangular uniform
    incoherent source. The resulting coherence function is a 2D sinc.

    Parameters
    ----------
    grid_size : Tuple[ScalarInteger, ScalarInteger]
        (height, width) of the grid in pixels.
    dx : ScalarFloat
        Pixel spacing in meters.
    source_width_x : ScalarFloat
        Width of the rectangular source in x direction (meters).
    source_width_y : ScalarFloat
        Width of the rectangular source in y direction (meters).
    wavelength : ScalarFloat
        Wavelength of light in meters.
    propagation_distance : ScalarFloat
        Distance from source to observation plane in meters.

    Returns
    -------
    kernel : Float[Array, " hh ww"]
        Coherence kernel centered at (0, 0).
        First zero at Delta_x = lambda * z / W_x (and similarly for y).

    Notes
    -----
    For a rectangular source:
        mu(Delta_x, Delta_y) = sinc(W_x * Delta_x / (lambda * z))
                             * sinc(W_y * Delta_y / (lambda * z))

    where sinc(x) = sin(pi*x) / (pi*x).
    """
    hh: int = int(grid_size[0])
    ww: int = int(grid_size[1])

    # Create centered coordinate grids
    y: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * dx
    x: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)

    # Scaling factors from van Cittert-Zernike
    lam: Float[Array, " "] = jnp.asarray(wavelength, dtype=jnp.float64)
    z: Float[Array, " "] = jnp.asarray(propagation_distance, dtype=jnp.float64)
    wx: Float[Array, " "] = jnp.asarray(source_width_x, dtype=jnp.float64)
    wy: Float[Array, " "] = jnp.asarray(source_width_y, dtype=jnp.float64)

    # Arguments for sinc functions
    arg_x: Float[Array, " hh ww"] = wx * xx / (lam * z)
    arg_y: Float[Array, " hh ww"] = wy * yy / (lam * z)

    # 2D sinc kernel
    kernel: Float[Array, " hh ww"] = jnp.sinc(arg_x) * jnp.sinc(arg_y)

    # Shift to have maximum at [0, 0] for convolution
    kernel = jnp.fft.ifftshift(kernel)

    return kernel


@jaxtyped(typechecker=beartype)
def coherence_width_from_source(
    source_diameter: ScalarFloat,
    wavelength: ScalarFloat,
    propagation_distance: ScalarFloat,
) -> Float[Array, " "]:
    """Calculate coherence width from source size using van Cittert-Zernike.

    Parameters
    ----------
    source_diameter : ScalarFloat
        Diameter or characteristic size of the incoherent source in meters.
    wavelength : ScalarFloat
        Wavelength of light in meters.
    propagation_distance : ScalarFloat
        Distance from source to observation plane in meters.

    Returns
    -------
    coherence_width : Float[Array, " "]
        Coherence width (transverse coherence length) in meters.
        This is the distance at which coherence drops significantly.

    Notes
    -----
    From the van Cittert-Zernike theorem, the coherence width scales as:
        sigma_c ~ lambda * z / D_source

    For a circular source, the first zero of coherence is at:
        Delta_r_0 = 1.22 * lambda * z / D

    This function returns the 1/e width for a Gaussian approximation:
        sigma_c = 0.44 * lambda * z / D

    which corresponds to the width where |mu| ~ 1/e for the central lobe.
    """
    lam: Float[Array, " "] = jnp.asarray(wavelength, dtype=jnp.float64)
    z: Float[Array, " "] = jnp.asarray(propagation_distance, dtype=jnp.float64)
    d: Float[Array, " "] = jnp.asarray(source_diameter, dtype=jnp.float64)

    # Factor 0.44 gives the 1/e width of a Gaussian approximation
    # to the jinc function's central lobe
    coherence_width: Float[Array, " "] = 0.44 * lam * z / d

    return coherence_width


@jaxtyped(typechecker=beartype)
def complex_degree_of_coherence(
    j_matrix: Complex[Array, " hh ww hh ww"],
) -> Complex[Array, " hh ww hh ww"]:
    """Compute normalized coherence mu(r1, r2) from mutual intensity.

    The complex degree of coherence normalizes the mutual intensity by
    the geometric mean of the intensities at the two points:
        mu(r1, r2) = J(r1, r2) / sqrt(I(r1) * I(r2))

    Parameters
    ----------
    j_matrix : Complex[Array, " hh ww hh ww"]
        Mutual intensity matrix J(r1, r2).

    Returns
    -------
    mu : Complex[Array, " hh ww hh ww"]
        Complex degree of coherence. |mu| = 1 for full coherence,
        |mu| = 0 for complete incoherence. Phase of mu encodes the
        relative phase between the two points.

    Notes
    -----
    The magnitude |mu(r1, r2)| determines fringe visibility in
    interference experiments:
        V = |mu| * 2*sqrt(I1*I2) / (I1 + I2)

    For equal intensities, V = |mu|.
    """
    hh: int = j_matrix.shape[0]
    ww: int = j_matrix.shape[1]

    # Extract intensity: I(r) = J(r, r) (diagonal elements)
    # Reshape for proper indexing
    i_idx: Float[Array, " hh"] = jnp.arange(hh)
    j_idx: Float[Array, " ww"] = jnp.arange(ww)

    # Get I(r) = J(r, r) for all r
    intensity: Float[Array, " hh ww"] = jnp.real(
        jnp.diagonal(
            jnp.diagonal(j_matrix, axis1=0, axis2=2), axis1=0, axis2=1
        )
    )

    # Create 4D normalization array: sqrt(I(r1) * I(r2))
    # Shape: (hh, ww, hh, ww)
    i1: Float[Array, " hh ww 1 1"] = intensity[:, :, jnp.newaxis, jnp.newaxis]
    i2: Float[Array, " 1 1 hh ww"] = intensity[jnp.newaxis, jnp.newaxis, :, :]
    normalization: Float[Array, " hh ww hh ww"] = jnp.sqrt(i1 * i2 + 1e-20)

    # Normalize
    mu: Complex[Array, " hh ww hh ww"] = j_matrix / normalization

    return mu


def jax_bessel_j1(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Bessel function of the first kind, order 1.

    Uses polynomial approximation for efficiency with JAX.

    Parameters
    ----------
    x : Float[Array, "..."]
        Input array.

    Returns
    -------
    j1 : Float[Array, "..."]
        J1(x) values.
    """
    # Use rational approximation for |x| < 8 and asymptotic for larger
    ax = jnp.abs(x)

    # For small x, use series approximation
    # J1(x) ~ x/2 - x^3/16 + x^5/384 - ...
    def small_x(x: Float[Array, "..."]) -> Float[Array, "..."]:
        x2 = x * x
        return x * (
            0.5 - x2 * (0.0625 - x2 * (0.00260417 - x2 * 0.0000542535))
        )

    # For larger x, use polynomial approximation
    def large_x(x: Float[Array, "..."]) -> Float[Array, "..."]:
        # Rational approximation coefficients (Abramowitz & Stegun)
        y = x / 3.0
        y2 = y * y
        p1 = 0.5 + y2 * (
            -0.56249985
            + y2 * (0.21093573 + y2 * (-0.03954289 + y2 * 0.00443319))
        )
        p2 = 1.0 + y2 * (
            -0.00031619
            + y2 * (-0.00024846 + y2 * (0.00017105 - y2 * 0.00004058))
        )
        return x * p1 / p2

    # Asymptotic expansion for very large x
    def asymptotic(x: Float[Array, "..."]) -> Float[Array, "..."]:
        z = 8.0 / x
        z2 = z * z
        theta = x - 0.75 * jnp.pi
        p0 = 1.0 + z2 * (-0.00145 + z2 * 0.0006)
        q0 = 0.125 / x * (1.0 + z2 * (-0.00278 + z2 * 0.00079))
        return jnp.sqrt(0.6366197724 / x) * (
            p0 * jnp.cos(theta) - q0 * jnp.sin(theta)
        )

    result = jnp.where(
        ax < 1.0,
        small_x(x),
        jnp.where(ax < 8.0, large_x(x), asymptotic(ax) * jnp.sign(x)),
    )

    return result
