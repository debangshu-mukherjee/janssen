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
from beartype.typing import Tuple
from jaxtyping import Array, Float, Int, jaxtyped

from janssen.utils import (
    OpticalWavefront,
    make_optical_wavefront,
    scalar_float,
    scalar_integer,
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
def noll_to_nm(j: scalar_integer) -> Tuple[int, int]:
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
    n = int(jnp.ceil((-3 + jnp.sqrt(9 + 8 * j)) / 2))
    n_prev = n * (n - 1) // 2
    p = j - n_prev - 1
    m_even_p_even = 2 * ((p + 1) // 2)
    m_even_p_odd = -2 * ((p + 1) // 2)
    m_even = jnp.where(p % 2 == 0, m_even_p_even, m_even_p_odd)
    m_odd_p_even = -2 * ((p + 2) // 2) + 1
    m_odd_p_odd = 2 * ((p + 2) // 2) - 1
    m_odd = jnp.where(p % 2 == 0, m_odd_p_even, m_odd_p_odd)
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
    Calculates j_base as the number of terms with radial order less than n.
    Position within the n group is computed differently for even and odd n values.
    For even n: positive m maps to m-1, negative m to -m-1, zero m to 0.
    For odd n: positive m maps to m, negative m to -m-1.
    """
    j_base = n * (n - 1) // 2

    p_even_m_pos = m - 1
    p_even_m_neg = -m - 1
    p_even_m_zero = 0
    p_even = jnp.where(
        m > 0, p_even_m_pos, jnp.where(m < 0, p_even_m_neg, p_even_m_zero)
    )

    p_odd_m_pos = m
    p_odd_m_neg = -m - 1
    p_odd = jnp.where(m > 0, p_odd_m_pos, p_odd_m_neg)

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
    
    Notes
    -----
    Uses JAX-compatible validation that returns zeros for invalid (n,m) combinations
    where n-|m| is odd. Computes the radial polynomial using the standard formula
    with factorials for valid combinations.
    """
    m_abs = abs(m)
    valid = (n - m_abs) % 2 == 0

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

    return jnp.where(valid, result, jnp.zeros_like(rho))


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
    Angular part uses cosine for m>0, sine for m<0, and 1 for m=0.
    Normalization factor is sqrt(n+1) for m=0 and sqrt(2*(n+1)) for m≠0.
    """
    R = zernike_radial(rho, n, abs(m))

    angular_cos = jnp.cos(abs(m) * theta)
    angular_sin = jnp.sin(abs(m) * theta)
    angular_ones = jnp.ones_like(theta)

    angular = jnp.where(
        m > 0, angular_cos, jnp.where(m < 0, angular_sin, angular_ones)
    )

    norm_m0 = jnp.sqrt(n + 1)
    norm_m_nonzero = jnp.sqrt(2 * (n + 1))
    norm = jnp.where(
        normalize, jnp.where(m == 0, norm_m0, norm_m_nonzero), 1.0
    )

    mask = rho <= 1.0

    return norm * R * angular * mask


@jaxtyped(typechecker=beartype)
def zernike_even(
    rho: Float[Array, " *batch"],
    theta: Float[Array, " *batch"],
    n: int,
    m: int,
    normalize: bool = True,
) -> Float[Array, " *batch"]:
    """Generate even (cosine) Zernike polynomial.

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
        Even Zernike polynomial using cosine for angular part

    Notes
    -----
    This function always uses cosine for the angular component,
    suitable for symmetric aberrations.
    Angular part uses cos(|m|*theta) for m≠0, and 1 for m=0.
    Normalization factor is sqrt(n+1) for m=0 and sqrt(2*(n+1)) for m≠0.
    Returns zero outside the unit circle (rho > 1).
    """
    R = zernike_radial(rho, n, abs(m))

    angular = jnp.where(m != 0, jnp.cos(abs(m) * theta), jnp.ones_like(theta))

    norm_m0 = jnp.sqrt(n + 1)
    norm_m_nonzero = jnp.sqrt(2 * (n + 1))
    norm = jnp.where(
        normalize, jnp.where(m == 0, norm_m0, norm_m_nonzero), 1.0
    )

    mask = rho <= 1.0

    return norm * R * angular * mask


@jaxtyped(typechecker=beartype)
def zernike_odd(
    rho: Float[Array, " *batch"],
    theta: Float[Array, " *batch"],
    n: int,
    m: int,
    normalize: bool = True,
) -> Float[Array, " *batch"]:
    """Generate odd (sine) Zernike polynomial.

    Parameters
    ----------
    rho : Float[Array, " *batch"]
        Normalized radial coordinate (0 to 1)
    theta : Float[Array, " *batch"]
        Azimuthal angle in radians
    n : int
        Radial order (n >= 0)
    m : int
        Azimuthal frequency (|m| <= n, n-|m| must be even, m != 0)
    normalize : bool, optional
        Whether to normalize for unit RMS over unit circle, by default True

    Returns
    -------
    Float[Array, " *batch"]
        Odd Zernike polynomial using sine for angular part

    Notes
    -----
    This function always uses sine for the angular component,
    suitable for antisymmetric aberrations. Returns zero if m=0.
    Angular part uses sin(|m|*theta) for all m values.
    Normalization factor is sqrt(2*(n+1)) when normalize=True.
    Returns zero outside the unit circle (rho > 1) and for m=0.
    """
    is_m_zero = m == 0

    R = zernike_radial(rho, n, abs(m))

    angular = jnp.sin(abs(m) * theta)

    norm = jnp.where(normalize, jnp.sqrt(2 * (n + 1)), 1.0)

    mask = rho <= 1.0

    return jnp.where(is_m_zero, jnp.zeros_like(rho), norm * R * angular * mask)


@jaxtyped(typechecker=beartype)
def generate_aberration_nm(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    n_indices: Int[Array, " N"],
    m_indices: Int[Array, " N"],
    coefficients: Float[Array, " N"],
    pupil_radius: scalar_float,
) -> Float[Array, " H W"]:
    """Generate aberration from (n,m) indices and coefficients (JAX-compatible).

    Parameters
    ----------
    xx : Float[Array, " H W"]
        X coordinate grid in meters
    yy : Float[Array, " H W"]
        Y coordinate grid in meters
    n_indices : Int[Array, " N"]
        Array of radial orders
    m_indices : Int[Array, " N"]
        Array of azimuthal frequencies
    coefficients : Float[Array, " N"]
        Zernike coefficients in waves
    pupil_radius : scalar_float
        Pupil radius in meters

    Returns
    -------
    Float[Array, " H W"]
        Phase aberration map in radians

    Notes
    -----
    This version is fully JAX-compatible and can be JIT-compiled.
    Uses jax.lax.scan for efficient accumulation.
    Converts Cartesian coordinates to polar coordinates normalized by pupil radius.
    Each Zernike contribution is accumulated using scan for efficiency.
    The n and m values must be concrete integers for zernike_polynomial.
    Final phase is converted from waves to radians.
    """
    rho = jnp.sqrt(xx**2 + yy**2) / pupil_radius
    theta = jnp.arctan2(yy, xx)

    def scan_fn(phase_acc, inputs):
        n, m, coeff = inputs
        Z = zernike_polynomial(rho, theta, int(n), int(m), normalize=True)
        phase_acc = phase_acc + coeff * Z
        return phase_acc, None

    phase, _ = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(xx),
        (n_indices, m_indices, coefficients),
    )

    return 2 * jnp.pi * phase


@jaxtyped(typechecker=beartype)
def generate_aberration_noll(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    coefficients: Float[Array, " N"],
    pupil_radius: scalar_float,
) -> Float[Array, " H W"]:
    """Generate aberration from Noll-indexed coefficients.

    Parameters
    ----------
    xx : Float[Array, " H W"]
        X coordinate grid in meters
    yy : Float[Array, " H W"]
        Y coordinate grid in meters
    coefficients : Float[Array, " N"]
        Zernike coefficients in waves, indexed by Noll index.
        Element 0 corresponds to j=1 (piston), element 1 to j=2, etc.
    pupil_radius : scalar_float
        Pupil radius in meters

    Returns
    -------
    Float[Array, " H W"]
        Phase aberration map in radians

    Notes
    -----
    Converts Noll indices to (n,m) pairs and calls generate_aberration_nm.
    The conversion happens at compile time with concrete indices.
    Pre-computes all (n,m) pairs for the given number of coefficients.
    """
    n_list = []
    m_list = []
    for j in range(1, len(coefficients) + 1):
        n, m = noll_to_nm(j)
        n_list.append(n)
        m_list.append(m)

    n_indices = jnp.array(n_list, dtype=jnp.int32)
    m_indices = jnp.array(m_list, dtype=jnp.int32)

    return generate_aberration_nm(
        xx, yy, n_indices, m_indices, coefficients, pupil_radius
    )


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
    coefficients = jnp.zeros(4)
    coefficients = coefficients.at[3].set(amplitude)
    return generate_aberration_noll(xx, yy, coefficients, pupil_radius)


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
    coefficients = jnp.zeros(6)
    coefficients = coefficients.at[4].set(amplitude_45)
    coefficients = coefficients.at[5].set(amplitude_0)
    return generate_aberration_noll(xx, yy, coefficients, pupil_radius)


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
    coefficients = jnp.zeros(8)
    coefficients = coefficients.at[6].set(amplitude_y)
    coefficients = coefficients.at[7].set(amplitude_x)
    return generate_aberration_noll(xx, yy, coefficients, pupil_radius)


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
    coefficients = jnp.zeros(11)
    coefficients = coefficients.at[10].set(amplitude)
    return generate_aberration_noll(xx, yy, coefficients, pupil_radius)


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
    trefoil_wavefront : Float[Array, " H W"]
        Trefoil phase map in radians

    Notes
    -----
    This function generates a trefoil aberration phase map in radians.
    The trefoil aberration is a combination of two Zernike polynomials:
    Z9 and Z10.
    The Z9 polynomial is the vertical trefoil aberration and the Z10 polynomial
    is the oblique trefoil aberration.
    """
    coefficients = jnp.zeros(10)
    coefficients = coefficients.at[8].set(amplitude_30)
    coefficients = coefficients.at[9].set(amplitude_0)
    trefoil_wavefront: Float[Array, " H W"] = generate_aberration_noll(
        xx, yy, coefficients, pupil_radius
    )
    return trefoil_wavefront


@jaxtyped(typechecker=beartype)
def apply_aberration(
    incoming: OpticalWavefront,
    coefficients: Float[Array, " N"],
    pupil_radius: scalar_float,
) -> OpticalWavefront:
    """Apply Zernike aberrations to an optical wavefront.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront
    coefficients : Float[Array, " N"]
        Noll-indexed Zernike coefficients in waves (index i = Noll index i+1)
    pupil_radius : scalar_float
        Pupil radius in meters

    Returns
    -------
    OpticalWavefront
        Aberrated wavefront
    """
    h, w = incoming.field.shape[:2]
    x = jnp.arange(-w // 2, w // 2) * incoming.dx
    y = jnp.arange(-h // 2, h // 2) * incoming.dx
    xx, yy = jnp.meshgrid(x, y)
    phase = generate_aberration_noll(xx, yy, coefficients, pupil_radius)
    field_out = add_phase_screen(incoming.field, phase)
    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
