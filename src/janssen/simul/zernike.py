"""Zernike polynomial functions for optical aberration modeling.

Extended Summary
----------------
This module provides functions for generating Zernike polynomials and
creating optical aberrations based on them. Zernike polynomials form a
complete orthogonal basis over the unit circle and are widely used in
optics to describe wavefront aberrations.

The module supports:
- Individual Zernike polynomial generation (Noll and OSA/ANSI indexing)
- Common aberration types (defocus, astigmatism, coma, spherical, etc.)
- Wavefront aberration synthesis from Zernike coefficients
- Conversion between different indexing conventions

Routine Listings
----------------
zernike_polynomial : function
    Generate a single Zernike polynomial
zernike_radial : function
    Radial component of Zernike polynomial
factorial : function
    JAX-compatible factorial computation
noll_to_nm : function
    Convert Noll index to (n, m) indices
nm_to_noll : function
    Convert (n, m) indices to Noll index
generate_aberration : function
    Generate aberration phase map from Zernike coefficients
defocus : function
    Generate defocus aberration (Z4)
astigmatism : function
    Generate astigmatism aberration (Z5, Z6)
coma : function
    Generate coma aberration (Z7, Z8)
spherical_aberration : function
    Generate spherical aberration (Z11)
trefoil : function
    Generate trefoil aberration (Z9, Z10)
apply_aberration : function
    Apply aberration to optical wavefront

Notes
-----
Zernike polynomials are defined on the unit circle with normalization
such that the RMS value over the unit circle equals 1. The polynomials
use the Noll indexing convention by default, which starts at j=1 for
piston. OSA/ANSI indexing is also supported.

References
----------
.. [1] Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence".
       JOSA, 66(3), 207-211.
.. [2] Born, M., & Wolf, E. (1999). Principles of optics (7th ed.).
       Cambridge University Press.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple, Union
from jaxtyping import Array, Float, Int, jaxtyped

from janssen.utils import (
    OpticalWavefront,
    make_optical_wavefront,
    scalar_float,
)

from .helper import add_phase_screen

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def factorial(n: Int[Array, ""]) -> Int[Array, ""]:
    """JAX-compatible factorial computation.

    Parameters
    ----------
    n : Int[Array, ""]
        Non-negative integer

    Returns
    -------
    Int[Array, ""]
        n! (n factorial)
    """
    return jnp.round(jnp.exp(jax.scipy.special.gammaln(n + 1))).astype(
        jnp.int64
    )


@jaxtyped(typechecker=beartype)
def noll_to_nm(j: int) -> Tuple[int, int]:
    """Convert Noll index to (n, m) indices.

    Parameters
    ----------
    j : int
        Noll index (starting from 1)

    Returns
    -------
    n : int
        Radial order
    m : int
        Azimuthal frequency (signed)

    Notes
    -----
    Uses the standard Noll ordering where j=1 corresponds to piston (n=0, m=0).
    This implementation uses JAX-compatible operations for JIT compilation.
    """
    if j < 1:
        raise ValueError("Noll index must be >= 1")

    # Find radial order n
    n = int(jnp.ceil((-3 + jnp.sqrt(9 + 8 * j)) / 2))

    # Number of terms with radial order less than n
    n_prev = n * (n - 1) // 2

    # Position within the n group (0-based)
    p = j - n_prev - 1

    # Determine m based on position and parity using JAX-compatible operations
    # For even n
    m_even_p_even = 2 * ((p + 1) // 2)
    m_even_p_odd = -2 * ((p + 1) // 2)
    m_even = jnp.where(p % 2 == 0, m_even_p_even, m_even_p_odd)
    
    # For odd n
    m_odd_p_even = -2 * ((p + 2) // 2) + 1
    m_odd_p_odd = 2 * ((p + 2) // 2) - 1
    m_odd = jnp.where(p % 2 == 0, m_odd_p_even, m_odd_p_odd)
    
    # Select based on n parity
    m = jnp.where(n % 2 == 0, m_even, m_odd)

    return n, int(m)


@jaxtyped(typechecker=beartype)
def nm_to_noll(n: int, m: int) -> int:
    """Convert (n, m) indices to Noll index.

    Parameters
    ----------
    n : int
        Radial order (n >= 0)
    m : int
        Azimuthal frequency (|m| <= n, n-|m| must be even)

    Returns
    -------
    int
        Noll index (starting from 1)
    
    Notes
    -----
    This implementation uses JAX-compatible operations for JIT compilation.
    """
    if abs(m) > n or (n - abs(m)) % 2 != 0:
        raise ValueError(f"Invalid (n, m) = ({n}, {m})")

    # Number of terms with radial order less than n
    j_base = n * (n - 1) // 2

    # Position within the n group using JAX-compatible operations
    # For even n
    p_even_m_pos = m - 1
    p_even_m_neg = -m - 1
    p_even_m_zero = 0
    p_even = jnp.where(m > 0, p_even_m_pos, 
                       jnp.where(m < 0, p_even_m_neg, p_even_m_zero))
    
    # For odd n (m=0 should not occur for odd n)
    p_odd_m_pos = m
    p_odd_m_neg = -m - 1
    p_odd = jnp.where(m > 0, p_odd_m_pos, p_odd_m_neg)
    
    # Select based on n parity
    p = jnp.where(n % 2 == 0, p_even, p_odd)

    return int(j_base + p + 1)


@jaxtyped(typechecker=beartype)
def zernike_radial(
    rho: Float[Array, " *batch"],
    n: int,
    m: int,
) -> Float[Array, " *batch"]:
    """Compute the radial component of Zernike polynomial.

    Parameters
    ----------
    rho : Float[Array, " *batch"]
        Normalized radial coordinate (0 to 1)
    n : int
        Radial order
    m : int
        Azimuthal frequency (absolute value used)

    Returns
    -------
    Float[Array, " *batch"]
        Radial polynomial R_n^|m|(rho)
    """
    m_abs = abs(m)

    if (n - m_abs) % 2 != 0:
        return jnp.zeros_like(rho)

    result = jnp.zeros_like(rho)
    for s in range((n - m_abs) // 2 + 1):
        sign = (-1) ** s
        num = factorial(jnp.array(n - s))
        denom = (
            factorial(jnp.array(s))
            * factorial(jnp.array((n + m_abs) // 2 - s))
            * factorial(jnp.array((n - m_abs) // 2 - s))
        )
        coeff = sign * num / denom
        result = result + coeff * (rho ** (n - 2 * s))

    return result


@jaxtyped(typechecker=beartype)
def zernike_polynomial(
    rho: Float[Array, " *batch"],
    theta: Float[Array, " *batch"],
    n: int,
    m: int,
    normalize: bool = True,
) -> Float[Array, " *batch"]:
    """Generate a single Zernike polynomial.

    Parameters
    ----------
    rho : Float[Array, " *batch"]
        Normalized radial coordinate (0 to 1)
    theta : Float[Array, " *batch"]
        Azimuthal angle in radians
    n : int
        Radial order (n >= 0)
    m : int
        Azimuthal frequency (|m| <= n, n-|m| must be even)
    normalize : bool, optional
        Whether to normalize for unit RMS over unit circle, by default True

    Returns
    -------
    Float[Array, " *batch"]
        Zernike polynomial Z_n^m(rho, theta)

    Notes
    -----
    The polynomial is zero outside the unit circle (rho > 1).
    Normalization follows the convention where RMS over unit circle = 1.
    """
    # Radial part
    R = zernike_radial(rho, n, abs(m))

    # Angular part (JAX-compatible)
    angular_cos = jnp.cos(abs(m) * theta)
    angular_sin = jnp.sin(abs(m) * theta)
    angular_ones = jnp.ones_like(theta)
    
    # Select angular function based on m
    angular = jnp.where(m > 0, angular_cos,
                        jnp.where(m < 0, angular_sin, angular_ones))

    # Normalization factor (JAX-compatible)
    norm_m0 = jnp.sqrt(n + 1)
    norm_m_nonzero = jnp.sqrt(2 * (n + 1))
    norm = jnp.where(
        normalize,
        jnp.where(m == 0, norm_m0, norm_m_nonzero),
        1.0
    )

    # Mask for unit circle
    mask = rho <= 1.0

    return norm * R * angular * mask


@jaxtyped(typechecker=beartype)
def generate_aberration(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    coefficients: Float[Array, " N"],
    pupil_radius: scalar_float,
    indexing: str = "noll",
) -> Float[Array, " H W"]:
    """Generate aberration phase map from Zernike coefficients.

    Parameters
    ----------
    xx : Float[Array, " H W"]
        X coordinate grid in meters
    yy : Float[Array, " H W"]
        Y coordinate grid in meters
    coefficients : Float[Array, " N"]
        Zernike coefficients in waves. Can be:
        - Array indexed by Noll index (starting from j=1)
        - Dict mapping Noll indices to coefficients
    pupil_radius : scalar_float
        Pupil radius in meters
    indexing : str, optional
        Indexing convention ("noll" or "ansi"), by default "noll"

    Returns
    -------
    Float[Array, " H W"]
        Phase aberration map in radians
    """
    # Convert to polar coordinates
    rho = jnp.sqrt(xx**2 + yy**2) / pupil_radius
    theta = jnp.arctan2(yy, xx)

    # Initialize phase map
    phase = jnp.zeros_like(xx)

    # Handle different coefficient formats
    if isinstance(coefficients, dict):
        coeff_items = coefficients.items()
    else:
        # Assume array indexed by Noll index
        coeff_items = enumerate(coefficients, start=1)

    # Sum contributions from each Zernike term
    for j, coeff in coeff_items:
        if abs(coeff) < 1e-10:  # Skip negligible coefficients
            continue

        if indexing == "noll":
            n, m = noll_to_nm(int(j))
        else:
            raise NotImplementedError(f"Indexing {indexing} not yet supported")

        Z = zernike_polynomial(rho, theta, n, m, normalize=True)
        phase = phase + coeff * Z

    # Convert from waves to radians
    return 2 * jnp.pi * phase


@jaxtyped(typechecker=beartype)
def defocus(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    amplitude: scalar_float,
    pupil_radius: scalar_float,
) -> Float[Array, " H W"]:
    """Generate defocus aberration (Z4 in Noll notation).

    Parameters
    ----------
    xx : Float[Array, " H W"]
        X coordinate grid in meters
    yy : Float[Array, " H W"]
        Y coordinate grid in meters
    amplitude : scalar_float
        Defocus amplitude in waves
    pupil_radius : scalar_float
        Pupil radius in meters

    Returns
    -------
    Float[Array, " H W"]
        Defocus phase map in radians
    """
    return generate_aberration(xx, yy, {4: amplitude}, pupil_radius)


@jaxtyped(typechecker=beartype)
def astigmatism(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    amplitude_0: scalar_float,
    amplitude_45: scalar_float,
    pupil_radius: scalar_float,
) -> Float[Array, " H W"]:
    """Generate astigmatism aberration (Z5 and Z6 in Noll notation).

    Parameters
    ----------
    xx : Float[Array, " H W"]
        X coordinate grid in meters
    yy : Float[Array, " H W"]
        Y coordinate grid in meters
    amplitude_0 : scalar_float
        Vertical/horizontal astigmatism amplitude in waves (Z6)
    amplitude_45 : scalar_float
        Oblique astigmatism amplitude in waves (Z5)
    pupil_radius : scalar_float
        Pupil radius in meters

    Returns
    -------
    Float[Array, " H W"]
        Astigmatism phase map in radians
    """
    return generate_aberration(
        xx, yy, {5: amplitude_45, 6: amplitude_0}, pupil_radius
    )


@jaxtyped(typechecker=beartype)
def coma(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    amplitude_x: scalar_float,
    amplitude_y: scalar_float,
    pupil_radius: scalar_float,
) -> Float[Array, " H W"]:
    """Generate coma aberration (Z7 and Z8 in Noll notation).

    Parameters
    ----------
    xx : Float[Array, " H W"]
        X coordinate grid in meters
    yy : Float[Array, " H W"]
        Y coordinate grid in meters
    amplitude_x : scalar_float
        Horizontal coma amplitude in waves (Z8)
    amplitude_y : scalar_float
        Vertical coma amplitude in waves (Z7)
    pupil_radius : scalar_float
        Pupil radius in meters

    Returns
    -------
    Float[Array, " H W"]
        Coma phase map in radians
    """
    return generate_aberration(
        xx, yy, {7: amplitude_y, 8: amplitude_x}, pupil_radius
    )


@jaxtyped(typechecker=beartype)
def spherical_aberration(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    amplitude: scalar_float,
    pupil_radius: scalar_float,
) -> Float[Array, " H W"]:
    """Generate primary spherical aberration (Z11 in Noll notation).

    Parameters
    ----------
    xx : Float[Array, " H W"]
        X coordinate grid in meters
    yy : Float[Array, " H W"]
        Y coordinate grid in meters
    amplitude : scalar_float
        Spherical aberration amplitude in waves
    pupil_radius : scalar_float
        Pupil radius in meters

    Returns
    -------
    Float[Array, " H W"]
        Spherical aberration phase map in radians
    """
    return generate_aberration(xx, yy, {11: amplitude}, pupil_radius)


@jaxtyped(typechecker=beartype)
def trefoil(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    amplitude_0: scalar_float,
    amplitude_30: scalar_float,
    pupil_radius: scalar_float,
) -> Float[Array, " H W"]:
    """Generate trefoil aberration (Z9 and Z10 in Noll notation).

    Parameters
    ----------
    xx : Float[Array, " H W"]
        X coordinate grid in meters
    yy : Float[Array, " H W"]
        Y coordinate grid in meters
    amplitude_0 : scalar_float
        Vertical trefoil amplitude in waves (Z10)
    amplitude_30 : scalar_float
        Oblique trefoil amplitude in waves (Z9)
    pupil_radius : scalar_float
        Pupil radius in meters

    Returns
    -------
    Float[Array, " H W"]
        Trefoil phase map in radians
    """
    return generate_aberration(
        xx, yy, {9: amplitude_30, 10: amplitude_0}, pupil_radius
    )


@jaxtyped(typechecker=beartype)
def apply_aberration(
    incoming: OpticalWavefront,
    coefficients: Union[Float[Array, " N"], dict],
    pupil_radius: scalar_float,
    indexing: str = "noll",
) -> OpticalWavefront:
    """Apply Zernike aberrations to an optical wavefront.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront
    coefficients : Union[Float[Array, " N"], dict]
        Zernike coefficients in waves
    pupil_radius : scalar_float
        Pupil radius in meters
    indexing : str, optional
        Indexing convention ("noll" or "ansi"), by default "noll"

    Returns
    -------
    OpticalWavefront
        Aberrated wavefront
    """
    # Create coordinate grids
    h, w = incoming.field.shape[:2]
    x = jnp.arange(-w // 2, w // 2) * incoming.dx
    y = jnp.arange(-h // 2, h // 2) * incoming.dx
    xx, yy = jnp.meshgrid(x, y)

    # Generate aberration phase map
    phase = generate_aberration(xx, yy, coefficients, pupil_radius, indexing)

    # Apply to field
    field_out = add_phase_screen(incoming.field, phase)

    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )