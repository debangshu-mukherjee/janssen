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
compute_phase_from_coeffs : function
    Compute phase map from Zernike coefficients with start index
phase_rms : function
    Compute RMS of phase within the unit pupil
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
    Sign convention: j even -> m >= 0 (cosine), j odd -> m <= 0 (sine).

    The radial order n is found from the cumulative count relation:
    n(n+1)/2 < j <= (n+1)(n+2)/2.

    Within each row n, the position k (0-indexed) determines |m|.
    For n even: |m| follows pattern 0,2,2,4,4,...
    For n odd: |m| follows pattern 1,1,3,3,5,5,...
    """
    n_float: Float[Array, " "] = (-1 + jnp.sqrt(1 + 8 * j)) / 2
    n: int = int(jnp.ceil(n_float)) - 1

    j_start: int = n * (n + 1) // 2 + 1
    k: int = j - j_start

    m_abs_even_n: int = 2 * ((k + 1) // 2)
    m_abs_odd_n: int = 2 * (k // 2) + 1
    m_abs: int = int(jnp.where(n % 2 == 0, m_abs_even_n, m_abs_odd_n))

    m_positive: int = m_abs
    m_negative: int = -m_abs
    m_with_sign: int = int(jnp.where(j % 2 == 0, m_positive, m_negative))
    m: int = int(jnp.where(m_abs == 0, 0, m_with_sign))

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
    Sign convention: j even -> m >= 0 (cosine), j odd -> m <= 0 (sine).

    The first Noll index for row n is j_base = n(n+1)/2 + 1.

    For m=0, the position k within the row is 0.
    For m!=0, find the pair of k values for the given |m|, then select
    based on the sign of m and the parity requirement.

    For n even: |m| values are 0,2,4,...; group index g = |m|/2;
    k_first = 2g-1 for g>0, else 0.
    For n odd: |m| values are 1,3,5,...; group index g = (|m|-1)/2;
    k_first = 2g.

    The final k is chosen such that m > 0 yields an even j,
    and m < 0 yields an odd j.
    """
    j_base: int = n * (n + 1) // 2 + 1
    m_abs: int = abs(m)

    g_even_n: int = m_abs // 2
    k_first_even_n: int = int(jnp.where(g_even_n > 0, 2 * g_even_n - 1, 0))
    g_odd_n: int = (m_abs - 1) // 2
    k_first_odd_n: int = 2 * g_odd_n
    k_first: int = int(jnp.where(n % 2 == 0, k_first_even_n, k_first_odd_n))

    j_first: int = j_base + k_first
    j_first_is_even: int = 1 - (j_first % 2)
    k_for_pos: int = int(jnp.where(j_first_is_even, k_first, k_first + 1))
    k_for_neg: int = int(jnp.where(j_first_is_even, k_first + 1, k_first))
    k: int = int(
        jnp.where(m_abs == 0, 0, jnp.where(m > 0, k_for_pos, k_for_neg))
    )

    j: int = j_base + k
    return j


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
    n, m = noll_to_nm(j)
    result: Float[Array, " *batch"] = zernike_nm(rho, theta, n, m, normalize)
    return result


def _zernike_radial_traced(
    rho: Float[Array, " *batch"],
    n: Int[Array, " "],
    m_abs: Int[Array, " "],
    max_n: int = 20,
) -> Float[Array, " *batch"]:
    """Traced-compatible radial Zernike polynomial.

    Supports traced n and m values by using fixed maximum loop bounds.
    This is necessary for use inside jax.lax.scan where n and m are traced.

    Parameters
    ----------
    rho : Float[Array, " *batch"]
        Normalized radial coordinate (0 to 1)
    n : Int[Array, " "]
        Radial order (traced JAX array)
    m_abs : Int[Array, " "]
        Absolute value of azimuthal frequency (traced JAX array)
    max_n : int, optional
        Maximum radial order to support, by default 20. The loop iterates
        max_n // 2 + 1 times regardless of actual n value.

    Returns
    -------
    Float[Array, " *batch"]
        Radial polynomial R_n^|m|(rho)

    Notes
    -----
    Validity check: (n - m_abs) must be even for valid Zernike polynomials.
    Invalid combinations return zeros.

    The number of terms in the sum is (n - m_abs) // 2 + 1.
    Terms where s >= num_terms are masked out during accumulation.

    Uses jax.scipy.special.gammaln for stable factorial computation with
    traced values.
    """
    valid = ((n - m_abs) % 2) == 0
    num_terms = (n - m_abs) // 2 + 1

    def body_fn(
        s: Int[Array, " "], carry: Float[Array, " *batch"]
    ) -> Float[Array, " *batch"]:
        sign = (-1.0) ** s
        num = jnp.exp(jax.scipy.special.gammaln(n - s + 1))
        denom_s = jnp.exp(jax.scipy.special.gammaln(s + 1))
        denom_n_plus = jnp.exp(
            jax.scipy.special.gammaln((n + m_abs) // 2 - s + 1)
        )
        denom_n_minus = jnp.exp(
            jax.scipy.special.gammaln((n - m_abs) // 2 - s + 1)
        )
        denom = denom_s * denom_n_plus * denom_n_minus
        coeff = sign * num / denom
        power_term = rho ** (n - 2 * s)
        term = coeff * power_term
        term = jnp.where(s < num_terms, term, 0.0)
        return carry + term

    result = jax.lax.fori_loop(0, max_n // 2 + 1, body_fn, jnp.zeros_like(rho))
    return jnp.where(valid, result, jnp.zeros_like(rho))


def _zernike_polynomial_traced(
    rho: Float[Array, " *batch"],
    theta: Float[Array, " *batch"],
    n: Int[Array, " "],
    m: Int[Array, " "],
    normalize: bool = True,
    max_n: int = 20,
) -> Float[Array, " *batch"]:
    """Traced-compatible Zernike polynomial.

    Supports traced n and m values for use inside jax.lax.scan.

    Parameters
    ----------
    rho : Float[Array, " *batch"]
        Normalized radial coordinate (0 to 1)
    theta : Float[Array, " *batch"]
        Azimuthal angle in radians
    n : Int[Array, " "]
        Radial order (traced JAX array)
    m : Int[Array, " "]
        Azimuthal frequency (traced JAX array, signed)
    normalize : bool, optional
        Whether to normalize for unit RMS over unit circle, by default True
    max_n : int, optional
        Maximum radial order to support, by default 20

    Returns
    -------
    Float[Array, " *batch"]
        Zernike polynomial Z_n^m(rho, theta)

    Notes
    -----
    Angular part uses cosine for m > 0, sine for m < 0, and 1 for m = 0.
    Normalization factor is sqrt(n+1) for m=0 and sqrt(2*(n+1)) for m≠0.
    Returns zero outside the unit circle (rho > 1).
    """
    m_abs = jnp.abs(m)
    r = _zernike_radial_traced(rho, n, m_abs, max_n)

    angular_cos = jnp.cos(m_abs * theta)
    angular_sin = jnp.sin(m_abs * theta)
    angular_ones = jnp.ones_like(theta)
    angular = jnp.where(
        m > 0, angular_cos, jnp.where(m < 0, angular_sin, angular_ones)
    )

    norm_m0 = jnp.sqrt(n + 1.0)
    norm_m_nonzero = jnp.sqrt(2.0 * (n + 1.0))
    norm = jnp.where(
        normalize, jnp.where(m == 0, norm_m0, norm_m_nonzero), 1.0
    )

    mask = rho <= 1.0
    return norm * r * angular * mask


@jaxtyped(typechecker=beartype)
def generate_aberration_nm(
    xx: Float[Array, " H W"],
    yy: Float[Array, " H W"],
    n_indices: Int[Array, " N"],
    m_indices: Int[Array, " N"],
    coefficients: Float[Array, " N"],
    pupil_radius: ScalarFloat,
) -> Float[Array, " H W"]:
    """Generate aberration from (n,m) indices and coefficients.

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
    Uses jax.lax.scan for efficient accumulation with traced-compatible
    Zernike polynomial computation.
    """
    rho: Float[Array, " H W"] = jnp.sqrt(xx**2 + yy**2) / pupil_radius
    theta: Float[Array, " H W"] = jnp.arctan2(yy, xx)

    def scan_fn(
        phase_acc: Float[Array, " H W"],
        inputs: Tuple[Int[Array, " "], Int[Array, " "], Float[Array, " "]],
    ) -> Tuple[Float[Array, " H W"], None]:
        n, m, coeff = inputs
        z: Float[Array, " H W"] = _zernike_polynomial_traced(
            rho, theta, n, m, normalize=True
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
    Sign convention: j even -> m >= 0 (cosine), j odd -> m <= 0 (sine).

    The radial order n is computed from n(n+1)/2 < j <= (n+1)(n+2)/2.
    The position k within row n determines |m|, which follows the pattern:
    0,2,2,4,4,... for n even and 1,1,3,3,5,5,... for n odd.
    """
    num_coeffs: int = coefficients.shape[0]
    j_indices: Int[Array, " nn"] = jnp.arange(
        1, num_coeffs + 1, dtype=jnp.int32
    )

    n_float: Float[Array, " nn"] = (-1 + jnp.sqrt(1 + 8 * j_indices)) / 2
    n_indices: Int[Array, " nn"] = (jnp.ceil(n_float) - 1).astype(jnp.int32)

    j_start: Int[Array, " nn"] = n_indices * (n_indices + 1) // 2 + 1
    k: Int[Array, " nn"] = j_indices - j_start

    m_abs_even_n: Int[Array, " nn"] = 2 * ((k + 1) // 2)
    m_abs_odd_n: Int[Array, " nn"] = 2 * (k // 2) + 1
    m_abs: Int[Array, " nn"] = jnp.where(
        n_indices % 2 == 0, m_abs_even_n, m_abs_odd_n
    )

    m_positive: Int[Array, " nn"] = m_abs
    m_negative: Int[Array, " nn"] = -m_abs
    m_with_sign: Int[Array, " nn"] = jnp.where(
        j_indices % 2 == 0, m_positive, m_negative
    )
    m_indices: Int[Array, " nn"] = jnp.where(m_abs == 0, 0, m_with_sign)

    phase: Float[Array, " hh ww"] = generate_aberration_nm(
        xx, yy, n_indices, m_indices, coefficients, pupil_radius
    )
    return phase


@jaxtyped(typechecker=beartype)
def compute_phase_from_coeffs(
    rho: Float[Array, " *batch"],
    theta: Float[Array, " *batch"],
    coefficients: Float[Array, " N"],
    start_noll: int = 4,
) -> Float[Array, " *batch"]:
    """Compute phase map from Zernike coefficients.

    Generates a phase aberration map by summing normalized Zernike polynomials
    weighted by the provided coefficients. The coefficients are mapped to
    consecutive Noll indices starting from `start_noll`.

    Parameters
    ----------
    rho : Float[Array, " *batch"]
        Normalized radial coordinate (0 to 1)
    theta : Float[Array, " *batch"]
        Azimuthal angle in radians
    coefficients : Float[Array, " N"]
        Zernike coefficients in waves. Element i corresponds to
        Noll index (start_noll + i).
    start_noll : int, optional
        Starting Noll index for the coefficients, by default 4 (defocus).
        Common choices: 1 (piston), 4 (defocus, skipping tip/tilt).

    Returns
    -------
    Float[Array, " *batch"]
        Phase map in radians

    Notes
    -----
    The phase is computed as:
        phase = 2 * pi * sum_i(coefficients[i] * Z_{start_noll + i})

    where Z_j is the normalized Zernike polynomial for Noll index j.
    The output is in radians, with coefficients interpreted as waves.

    Examples
    --------
    >>> # Compute phase for defocus through spherical aberration (j=4 to j=11)
    >>> coeffs = jnp.array([0.5, 0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.3])
    >>> phase = compute_phase_from_coeffs(rho, theta, coeffs, start_noll=4)
    """
    num_coeffs: int = coefficients.shape[0]
    noll_indices: Int[Array, " N"] = jnp.arange(
        start_noll, start_noll + num_coeffs, dtype=jnp.int32
    )

    # Convert Noll indices to (n, m) pairs using vectorized operations
    n_float: Float[Array, " N"] = (-1 + jnp.sqrt(1 + 8 * noll_indices)) / 2
    n_indices: Int[Array, " N"] = (jnp.ceil(n_float) - 1).astype(jnp.int32)

    j_start: Int[Array, " N"] = n_indices * (n_indices + 1) // 2 + 1
    k: Int[Array, " N"] = noll_indices - j_start

    m_abs_even_n: Int[Array, " N"] = 2 * ((k + 1) // 2)
    m_abs_odd_n: Int[Array, " N"] = 2 * (k // 2) + 1
    m_abs: Int[Array, " N"] = jnp.where(
        n_indices % 2 == 0, m_abs_even_n, m_abs_odd_n
    )

    m_positive: Int[Array, " N"] = m_abs
    m_negative: Int[Array, " N"] = -m_abs
    m_with_sign: Int[Array, " N"] = jnp.where(
        noll_indices % 2 == 0, m_positive, m_negative
    )
    m_indices: Int[Array, " N"] = jnp.where(m_abs == 0, 0, m_with_sign)

    # Accumulate phase using scan
    def scan_fn(
        phase_acc: Float[Array, " *batch"],
        inputs: Tuple[Int[Array, " "], Int[Array, " "], Float[Array, " "]],
    ) -> Tuple[Float[Array, " *batch"], None]:
        n, m, coeff = inputs
        z: Float[Array, " *batch"] = _zernike_polynomial_traced(
            rho, theta, n, m, normalize=True
        )
        updated_phase: Float[Array, " *batch"] = phase_acc + coeff * z
        return updated_phase, None

    initial_phase: Float[Array, " *batch"] = jnp.zeros_like(rho)
    inputs: Tuple[Int[Array, " N"], Int[Array, " N"], Float[Array, " N"]] = (
        n_indices,
        m_indices,
        coefficients,
    )
    phase: Float[Array, " *batch"]
    phase, _ = jax.lax.scan(scan_fn, initial_phase, inputs)

    phase_radians: Float[Array, " *batch"] = 2 * jnp.pi * phase
    return phase_radians


@jaxtyped(typechecker=beartype)
def phase_rms(
    rho: Float[Array, " *batch"],
    theta: Float[Array, " *batch"],
    coefficients: Float[Array, " N"],
    start_noll: int = 4,
) -> Float[Array, " "]:
    """Compute RMS of phase within the unit pupil.

    Calculates the root-mean-square of the phase aberration within the
    region where rho <= 1.0 (the unit pupil).

    Parameters
    ----------
    rho : Float[Array, " *batch"]
        Normalized radial coordinate (0 to 1)
    theta : Float[Array, " *batch"]
        Azimuthal angle in radians
    coefficients : Float[Array, " N"]
        Zernike coefficients in waves. Element i corresponds to
        Noll index (start_noll + i).
    start_noll : int, optional
        Starting Noll index for the coefficients, by default 4 (defocus).

    Returns
    -------
    Float[Array, " "]
        RMS phase value in radians

    Notes
    -----
    The RMS is computed as:
        RMS = sqrt(mean((phase - mean(phase))^2))

    where the mean is taken only over pixels within the unit pupil (rho <= 1).
    The piston (mean phase) is subtracted before computing RMS.

    Examples
    --------
    >>> # Compute RMS for a set of aberration coefficients
    >>> coeffs = jnp.array([0.5, 0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.3])
    >>> rms = phase_rms(rho, theta, coeffs, start_noll=4)
    """
    phase: Float[Array, " *batch"] = compute_phase_from_coeffs(
        rho, theta, coefficients, start_noll
    )
    mask: Float[Array, " *batch"] = rho <= 1.0
    phase_in_pupil: Float[Array, " *batch"] = jnp.where(mask, phase, 0.0)
    n_pixels: Float[Array, " "] = jnp.sum(mask)
    mean_phase: Float[Array, " "] = jnp.sum(phase_in_pupil) / n_pixels
    variance: Float[Array, " "] = (
        jnp.sum(jnp.where(mask, (phase - mean_phase) ** 2, 0.0)) / n_pixels
    )
    rms: Float[Array, " "] = jnp.sqrt(variance)
    return rms


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
