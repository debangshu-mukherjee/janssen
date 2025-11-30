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
zernike_even : function
    Generate even (cosine) Zernike polynomial
zernike_odd : function
    Generate odd (sine) Zernike polynomial
zernike_nm : function
    Generate Zernike polynomial from (n,m) indices
zernike_noll : function
    Generate Zernike polynomial from Noll index
factorial : function
    JAX-compatible factorial computation
noll_to_nm : function
    Convert Noll index to (n, m) indices
nm_to_noll : function
    Convert (n, m) indices to Noll index
generate_aberration_nm : function
    Generate aberration phase map from (n,m) indices and coefficients
generate_aberration_noll : function
    Generate aberration phase map from Noll-indexed coefficients
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
    ScalarFloat,
    ScalarInteger,
    make_optical_wavefront,
)

from .helper import add_phase_screen

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def factorial(n: Int[Array, " "]) -> Int[Array, " "]:
    """JAX-compatible factorial computation.

    Parameters
    ----------
    n : Int[Array, " "]
        Non-negative integer

    Returns
    -------
    Int[Array, " "]
        n! (n factorial)
    """
    gammaln_result: Float[Array, " "] = jax.scipy.special.gammaln(n + 1)
    exp_result: Float[Array, " "] = jnp.exp(gammaln_result)
    rounded: Float[Array, " "] = jnp.round(exp_result)
    result: Int[Array, " "] = rounded.astype(jnp.int64)
    return result


@jaxtyped(typechecker=beartype)
def noll_to_nm(j: ScalarInteger) -> Tuple[int, int]:
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
    sqrt_term: Float[Array, " "] = jnp.sqrt(9 + 8 * j)
    n_float: Float[Array, " "] = (-3 + sqrt_term) / 2
    n: int = int(jnp.ceil(n_float))
    n_prev: int = n * (n - 1) // 2
    p: int = j - n_prev - 1
    m_even_p_even: int = 2 * ((p + 1) // 2)
    m_even_p_odd: int = -2 * ((p + 1) // 2)
    m_even: Int[Array, " "] = jnp.where(
        p % 2 == 0, m_even_p_even, m_even_p_odd
    )
    m_odd_p_even: int = -2 * ((p + 2) // 2) + 1
    m_odd_p_odd: int = 2 * ((p + 2) // 2) - 1
    m_odd: Int[Array, " "] = jnp.where(p % 2 == 0, m_odd_p_even, m_odd_p_odd)
    m_array: Int[Array, " "] = jnp.where(n % 2 == 0, m_even, m_odd)
    m: int = int(m_array)
    return n, m


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
    Position within the n group is computed differently for even and odd n
    values.
    For even n: positive m maps to m-1, negative m to -m-1, zero m to 0.
    For odd n: positive m maps to m, negative m to -m-1.
    """
    j_base: int = n * (n - 1) // 2

    p_even_m_pos: int = m - 1
    p_even_m_neg: int = -m - 1
    p_even_m_zero: int = 0
    p_even: Int[Array, " "] = jnp.where(
        m > 0, p_even_m_pos, jnp.where(m < 0, p_even_m_neg, p_even_m_zero)
    )

    p_odd_m_pos: int = m
    p_odd_m_neg: int = -m - 1
    p_odd: Int[Array, " "] = jnp.where(m > 0, p_odd_m_pos, p_odd_m_neg)

    p: Int[Array, " "] = jnp.where(n % 2 == 0, p_even, p_odd)

    result: int = int(j_base + p + 1)
    return result


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
    Uses JAX-compatible validation that returns zeros for invalid (n,m)
    combinations where n-|m| is odd. Computes the radial polynomial using
    the standard formula with factorials for valid combinations.
    Uses jax.lax.scan for efficient accumulation of terms.
    """
    m_abs: int = abs(m)
    valid: bool = (n - m_abs) % 2 == 0

    def scan_fn(
        carry: Float[Array, " *batch"], s: Int[Array, " "]
    ) -> Tuple[Float[Array, " *batch"], None]:
        sign: Float[Array, " "] = (-1.0) ** s
        num: Int[Array, " "] = factorial(jnp.array(n - s))
        denom_s: Int[Array, " "] = factorial(s)
        denom_n_plus: Int[Array, " "] = factorial(
            jnp.array((n + m_abs) // 2 - s)
        )
        denom_n_minus: Int[Array, " "] = factorial(
            jnp.array((n - m_abs) // 2 - s)
        )
        denom: Int[Array, " "] = denom_s * denom_n_plus * denom_n_minus
        coeff: Float[Array, " "] = sign * num / denom
        power_term: Float[Array, " *batch"] = rho ** (n - 2 * s)
        updated_result: Float[Array, " *batch"] = carry + coeff * power_term
        return updated_result, None

    initial_result: Float[Array, " *batch"] = jnp.zeros_like(rho)
    s_values: Int[Array, " S"] = jnp.arange((n - m_abs) // 2 + 1)
    result: Float[Array, " *batch"]
    result, _ = jax.lax.scan(scan_fn, initial_result, s_values)

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
    r: Float[Array, " *batch"] = zernike_radial(rho, n, abs(m))

    m_abs: int = abs(m)
    angular_cos: Float[Array, " *batch"] = jnp.cos(m_abs * theta)
    angular_sin: Float[Array, " *batch"] = jnp.sin(m_abs * theta)
    angular_ones: Float[Array, " *batch"] = jnp.ones_like(theta)
    angular: Float[Array, " *batch"] = jnp.where(
        m > 0, angular_cos, jnp.where(m < 0, angular_sin, angular_ones)
    )
    norm_m0: Float[Array, " "] = jnp.sqrt(n + 1)
    norm_m_nonzero: Float[Array, " "] = jnp.sqrt(2 * (n + 1))
    norm: Float[Array, " "] = jnp.where(
        normalize, jnp.where(m == 0, norm_m0, norm_m_nonzero), 1.0
    )
    mask: Float[Array, " *batch"] = rho <= 1.0
    return norm * r * angular * mask


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
    even_polynomial : Float[Array, " *batch"]
        Even Zernike polynomial using cosine for angular part

    Notes
    -----
    This function always uses cosine for the angular component,
    suitable for symmetric aberrations.
    Angular part uses cos(|m|*theta) for m≠0, and 1 for m=0.
    Normalization factor is sqrt(n+1) for m=0 and sqrt(2*(n+1)) for m≠0.
    Returns zero outside the unit circle (rho > 1).
    """
    r: Float[Array, " *batch"] = zernike_radial(rho, n, abs(m))
    m_abs: int = abs(m)
    cos_term: Float[Array, " *batch"] = jnp.cos(m_abs * theta)
    ones_term: Float[Array, " *batch"] = jnp.ones_like(theta)
    angular: Float[Array, " *batch"] = jnp.where(m != 0, cos_term, ones_term)
    norm_m0: Float[Array, " "] = jnp.sqrt(n + 1)
    norm_m_nonzero: Float[Array, " "] = jnp.sqrt(2 * (n + 1))
    norm: Float[Array, " "] = jnp.where(
        normalize, jnp.where(m == 0, norm_m0, norm_m_nonzero), 1.0
    )
    mask: Float[Array, " *batch"] = rho <= 1.0
    even_polynomial: Float[Array, " *batch"] = norm * r * angular * mask
    return even_polynomial


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
    odd_polynomial : Float[Array, " *batch"]
        Odd Zernike polynomial using sine for angular part

    Notes
    -----
    This function always uses sine for the angular component,
    suitable for antisymmetric aberrations. Returns zero if m=0.
    Angular part uses sin(|m|*theta) for all m values.
    Normalization factor is sqrt(2*(n+1)) when normalize=True.
    Returns zero outside the unit circle (rho > 1) and for m=0.
    """
    is_m_zero: bool = m == 0
    r: Float[Array, " *batch"] = zernike_radial(rho, n, abs(m))
    m_abs: int = abs(m)
    angular: Float[Array, " *batch"] = jnp.sin(m_abs * theta)
    norm_value: Float[Array, " "] = jnp.sqrt(2 * (n + 1))
    norm: Float[Array, " "] = jnp.where(normalize, norm_value, 1.0)
    mask: Float[Array, " *batch"] = rho <= 1.0
    polynomial: Float[Array, " *batch"] = norm * r * angular * mask
    zeros: Float[Array, " *batch"] = jnp.zeros_like(rho)
    odd_polynomial: Float[Array, " *batch"] = jnp.where(
        is_m_zero, zeros, polynomial
    )
    return odd_polynomial


@jaxtyped(typechecker=beartype)
def zernike_nm(
    rho: Float[Array, " *batch"],
    theta: Float[Array, " *batch"],
    n: int,
    m: int,
    normalize: bool = True,
) -> Float[Array, " *batch"]:
    """Generate Zernike polynomial based on (n,m) indices.

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
    Determines whether to use even (cosine) or odd (sine) Zernike polynomial
    based on the sign of m. For m>=0, uses even (cosine) form.
    For m<0, uses odd (sine) form.
    """
    is_even: bool = m >= 0
    even_result: Float[Array, " *batch"] = zernike_even(
        rho, theta, n, abs(m), normalize
    )
    odd_result: Float[Array, " *batch"] = zernike_odd(
        rho, theta, n, abs(m), normalize
    )
    result: Float[Array, " *batch"] = jnp.where(
        is_even, even_result, odd_result
    )
    return result


@jaxtyped(typechecker=beartype)
def zernike_noll(
    rho: Float[Array, " *batch"],
    theta: Float[Array, " *batch"],
    j: int,
    normalize: bool = True,
) -> Float[Array, " *batch"]:
    """Generate Zernike polynomial based on Noll index.

    Parameters
    ----------
    rho : Float[Array, " *batch"]
        Normalized radial coordinate (0 to 1)
    theta : Float[Array, " *batch"]
        Azimuthal angle in radians
    j : int
        Noll index (starting from 1)
    normalize : bool, optional
        Whether to normalize for unit RMS over unit circle, by default True

    Returns
    -------
    Float[Array, " *batch"]
        Zernike polynomial for Noll index j

    Notes
    -----
    Converts Noll index to (n,m) pair and calls zernike_nm.
    The Noll indexing convention assigns j=1 to piston (n=0, m=0).
    """
    nm_tuple: Tuple[int, int] = noll_to_nm(j)
    n = nm_tuple[0]
    m = nm_tuple[1]
    result: Float[Array, " *batch"] = zernike_nm(rho, theta, n, m, normalize)
    return result


@jaxtyped(typechecker=beartype)
def generate_aberration_nm(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    n_indices: Int[Array, " N"],
    m_indices: Int[Array, " N"],
    coefficients: Float[Array, " N"],
    pupil_radius: ScalarFloat,
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
    pupil_radius : ScalarFloat
        Pupil radius in meters

    Returns
    -------
    phase_radians : Float[Array, " H W"]
        Phase aberration map in radians

    Notes
    -----
    This version is fully JAX-compatible and can be JIT-compiled.
    Uses jax.lax.scan for efficient accumulation.
    Converts Cartesian coordinates to polar coordinates normalized by pupil
    radius.
    Each Zernike contribution is accumulated using scan for efficiency.
    The n and m values must be concrete integers for zernike_polynomial.
    Final phase is converted from waves to radians.
    """
    rho: Float[Array, " H W"] = jnp.sqrt(xx**2 + yy**2) / pupil_radius
    theta: Float[Array, " H W"] = jnp.arctan2(yy, xx)

    def scan_fn(
        phase_acc: Float[Array, " H W"],
        inputs: Tuple[Int[Array, " "], Int[Array, " "], Float[Array, " "]],
    ) -> Tuple[Float[Array, " H W"], None]:
        n, m, coeff = inputs
        z: Float[Array, " H W"] = zernike_polynomial(
            rho, theta, int(n), int(m), normalize=True
        )
        updated_phase: Float[Array, " H W"] = phase_acc + coeff * z
        return updated_phase, None

    initial_phase: Float[Array, " H W"] = jnp.zeros_like(xx)
    inputs: Tuple[Int[Array, " N"], Int[Array, " N"], Float[Array, " N"]] = (
        n_indices,
        m_indices,
        coefficients,
    )
    phase: Float[Array, " H W"]
    phase, _ = jax.lax.scan(scan_fn, initial_phase, inputs)

    phase_radians: Float[Array, " H W"] = 2 * jnp.pi * phase
    return phase_radians


@jaxtyped(typechecker=beartype)
def generate_aberration_noll(
    xx: Float[Array, " hh ww"],
    yy: Float[Array, " hh ww"],
    coefficients: Float[Array, " nn"],
    pupil_radius: ScalarFloat,
) -> Float[Array, " hh ww"]:
    """Generate aberration from Noll-indexed coefficients.

    Parameters
    ----------
    xx : Float[Array, " hh ww"]
        X coordinate grid in meters
    yy : Float[Array, " hh ww"]
        Y coordinate grid in meters
    coefficients : Float[Array, " nn"]
        Zernike coefficients in waves, indexed by Noll index.
        Element 0 corresponds to j=1 (piston), element 1 to j=2, etc.
    pupil_radius : ScalarFloat
        Pupil radius in meters

    Returns
    -------
    phase : Float[Array, " hh ww"]
        Phase aberration map in radians

    Notes
    -----
    Converts Noll indices to (n,m) pairs and calls generate_aberration_nm.
    Uses vectorized JAX operations for the Noll-to-nm conversion.
    """
    num_coeffs: int = coefficients.shape[0]
    j_indices: Int[Array, " nn"] = jnp.arange(
        1, num_coeffs + 1, dtype=jnp.int32
    )
    sqrt_term: Float[Array, " nn"] = jnp.sqrt(9 + 8 * j_indices)
    n_float: Float[Array, " nn"] = (-3 + sqrt_term) / 2
    n_indices: Int[Array, " nn"] = jnp.ceil(n_float).astype(jnp.int32)
    n_prev: Int[Array, " nn"] = n_indices * (n_indices - 1) // 2
    p: Int[Array, " nn"] = j_indices - n_prev - 1
    m_even_p_even: Int[Array, " nn"] = 2 * ((p + 1) // 2)
    m_even_p_odd: Int[Array, " nn"] = -2 * ((p + 1) // 2)
    m_even: Int[Array, " nn"] = jnp.where(
        p % 2 == 0, m_even_p_even, m_even_p_odd
    )
    m_odd_p_even: Int[Array, " nn"] = -2 * ((p + 2) // 2) + 1
    m_odd_p_odd: Int[Array, " nn"] = 2 * ((p + 2) // 2) - 1
    m_odd: Int[Array, " nn"] = jnp.where(p % 2 == 0, m_odd_p_even, m_odd_p_odd)

    m_indices: Int[Array, " nn"] = jnp.where(n_indices % 2 == 0, m_even, m_odd)
    phase: Float[Array, " hh ww"] = generate_aberration_nm(
        xx, yy, n_indices, m_indices, coefficients, pupil_radius
    )
    return phase


@jaxtyped(typechecker=beartype)
def defocus(
    xx: Float[Array, " hh ww"],
    yy: Float[Array, " hh ww"],
    amplitude: ScalarFloat,
    pupil_radius: ScalarFloat,
) -> Float[Array, " hh ww"]:
    """Generate defocus aberration (Z4 in Noll notation).

    Parameters
    ----------
    xx : Float[Array, " hh ww"]
        X coordinate grid in meters
    yy : Float[Array, " hh ww"]
        Y coordinate grid in meters
    amplitude : ScalarFloat
        Defocus amplitude in waves
    pupil_radius : ScalarFloat
        Pupil radius in meters

    Returns
    -------
    phase : Float[Array, " hh ww"]
        Defocus phase map in radians
    """
    coefficients: Float[Array, " 4"] = jnp.zeros(4)
    coefficients = coefficients.at[3].set(amplitude)
    phase: Float[Array, " hh ww"] = generate_aberration_noll(
        xx, yy, coefficients, pupil_radius
    )
    return phase


@jaxtyped(typechecker=beartype)
def astigmatism(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    amplitude_0: ScalarFloat,
    amplitude_45: ScalarFloat,
    pupil_radius: ScalarFloat,
) -> Float[Array, " H W"]:
    """Generate astigmatism aberration (Z5 and Z6 in Noll notation).

    Parameters
    ----------
    xx : Float[Array, " H W"]
        X coordinate grid in meters
    yy : Float[Array, " H W"]
        Y coordinate grid in meters
    amplitude_0 : ScalarFloat
        Vertical/horizontal astigmatism amplitude in waves (Z6)
    amplitude_45 : ScalarFloat
        Oblique astigmatism amplitude in waves (Z5)
    pupil_radius : ScalarFloat
        Pupil radius in meters

    Returns
    -------
    phase : Float[Array, " H W"]
        Astigmatism phase map in radians
    """
    coefficients: Float[Array, " 6"] = jnp.zeros(6)
    coefficients = coefficients.at[4].set(amplitude_45)
    coefficients = coefficients.at[5].set(amplitude_0)
    phase: Float[Array, " H W"] = generate_aberration_noll(
        xx, yy, coefficients, pupil_radius
    )
    return phase


@jaxtyped(typechecker=beartype)
def coma(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    amplitude_x: ScalarFloat,
    amplitude_y: ScalarFloat,
    pupil_radius: ScalarFloat,
) -> Float[Array, " H W"]:
    """Generate coma aberration (Z7 and Z8 in Noll notation).

    Parameters
    ----------
    xx : Float[Array, " H W"]
        X coordinate grid in meters
    yy : Float[Array, " H W"]
        Y coordinate grid in meters
    amplitude_x : ScalarFloat
        Horizontal coma amplitude in waves (Z8)
    amplitude_y : ScalarFloat
        Vertical coma amplitude in waves (Z7)
    pupil_radius : ScalarFloat
        Pupil radius in meters

    Returns
    -------
    phase : Float[Array, " H W"]
        Coma phase map in radians
    """
    coefficients: Float[Array, " 8"] = jnp.zeros(8)
    coefficients = coefficients.at[6].set(amplitude_y)
    coefficients = coefficients.at[7].set(amplitude_x)
    phase: Float[Array, " H W"] = generate_aberration_noll(
        xx, yy, coefficients, pupil_radius
    )
    return phase


@jaxtyped(typechecker=beartype)
def spherical_aberration(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    amplitude: ScalarFloat,
    pupil_radius: ScalarFloat,
) -> Float[Array, " H W"]:
    """Generate primary spherical aberration (Z11 in Noll notation).

    Parameters
    ----------
    xx : Float[Array, " H W"]
        X coordinate grid in meters
    yy : Float[Array, " H W"]
        Y coordinate grid in meters
    amplitude : ScalarFloat
        Spherical aberration amplitude in waves
    pupil_radius : ScalarFloat
        Pupil radius in meters

    Returns
    -------
    phase : Float[Array, " H W"]
        Spherical aberration phase map in radians
    """
    coefficients: Float[Array, " 11"] = jnp.zeros(11)
    coefficients = coefficients.at[10].set(amplitude)
    phase: Float[Array, " H W"] = generate_aberration_noll(
        xx, yy, coefficients, pupil_radius
    )
    return phase


@jaxtyped(typechecker=beartype)
def trefoil(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    amplitude_0: ScalarFloat,
    amplitude_30: ScalarFloat,
    pupil_radius: ScalarFloat,
) -> Float[Array, " H W"]:
    """Generate trefoil aberration (Z9 and Z10 in Noll notation).

    Parameters
    ----------
    xx : Float[Array, " H W"]
        X coordinate grid in meters
    yy : Float[Array, " H W"]
        Y coordinate grid in meters
    amplitude_0 : ScalarFloat
        Vertical trefoil amplitude in waves (Z10)
    amplitude_30 : ScalarFloat
        Oblique trefoil amplitude in waves (Z9)
    pupil_radius : ScalarFloat
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
    coefficients: Float[Array, " 10"] = jnp.zeros(10)
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
    pupil_radius: ScalarFloat,
) -> OpticalWavefront:
    """Apply Zernike aberrations to an optical wavefront.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront
    coefficients : Float[Array, " N"]
        Noll-indexed Zernike coefficients in waves (index i = Noll index i+1)
    pupil_radius : ScalarFloat
        Pupil radius in meters

    Returns
    -------
    wavefront_out : OpticalWavefront
        Aberrated wavefront
    """
    h: int
    w: int
    h, w = incoming.field.shape[:2]
    x: Float[Array, " W"] = jnp.arange(-w // 2, w // 2) * incoming.dx
    y: Float[Array, " H"] = jnp.arange(-h // 2, h // 2) * incoming.dx
    xx: Float[Array, " H W"]
    yy: Float[Array, " H W"]
    xx, yy = jnp.meshgrid(x, y)
    phase: Float[Array, " H W"] = generate_aberration_noll(
        xx, yy, coefficients, pupil_radius
    )
    field_out: Float[Array, " H W"] = add_phase_screen(incoming.field, phase)
    wavefront_out: OpticalWavefront = make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return wavefront_out
