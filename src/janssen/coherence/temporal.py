"""Temporal/spectral coherence functions.

Extended Summary
----------------
This module provides functions for modeling temporal coherence in optical
fields. Temporal coherence describes the correlation of the field with
itself at different times, which is related to the spectral bandwidth.

The **coherence time** tau_c is inversely related to spectral bandwidth:
    tau_c ~ 1 / Delta_nu = lambda^2 / (c * Delta_lambda)

The **coherence length** is:
    L_c = c * tau_c = lambda^2 / Delta_lambda

For polychromatic light, different wavelength components propagate with
different phase velocities, leading to temporal decoherence.

Routine Listings
----------------
gaussian_spectrum : function
    Generate Gaussian spectral distribution
lorentzian_spectrum : function
    Generate Lorentzian spectral distribution (natural lineshape)
rectangular_spectrum : function
    Generate rectangular (flat-top) spectral distribution
blackbody_spectrum : function
    Generate blackbody (Planck) spectral distribution
coherence_length : function
    Calculate coherence length L_c = lambda^2 / Delta_lambda
coherence_time : function
    Calculate coherence time tau_c = lambda^2 / (c * Delta_lambda)
bandwidth_from_coherence_length : function
    Calculate bandwidth from coherence length

Notes
-----
The temporal coherence function gamma(tau) is the Fourier transform of
the power spectrum S(nu). For example:
- Gaussian spectrum -> Gaussian coherence function
- Lorentzian spectrum -> Exponential coherence function
- Rectangular spectrum -> Sinc coherence function

References
----------
1. Goodman, J. W. "Statistical Optics" (2015)
2. Born, M. & Wolf, E. "Principles of Optics" Chapter 10
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float, jaxtyped

from janssen.utils import ScalarFloat, ScalarInteger

C_LIGHT: float = 299792458.0


@jaxtyped(typechecker=beartype)
def gaussian_spectrum(
    center_wavelength: ScalarFloat,
    bandwidth_fwhm: ScalarFloat,
    num_wavelengths: ScalarInteger,
    wavelength_range: Optional[Tuple[ScalarFloat, ScalarFloat]] = None,
) -> Tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Generate Gaussian spectral distribution.

    Parameters
    ----------
    center_wavelength : ScalarFloat
        Center wavelength in meters.
    bandwidth_fwhm : ScalarFloat
        Full width at half maximum (FWHM) of the spectrum in meters.
    num_wavelengths : ScalarInteger
        Number of wavelength sample points.
    wavelength_range : Optional[Tuple[ScalarFloat, ScalarFloat]]
        (min, max) wavelength range in meters. If None, uses
        center +/- 3*sigma where sigma = FWHM / 2.355.

    Returns
    -------
    wavelengths : Float[Array, " n"]
        Wavelength sample points in meters.
    weights : Float[Array, " n"]
        Normalized spectral weights S(lambda). Sum equals 1.

    Notes
    -----
    The Gaussian spectrum is:
        S(lambda) = exp(-(lambda - lambda_0)^2 / (2*sigma^2))

    where sigma = FWHM / (2*sqrt(2*ln(2))) ~ FWHM / 2.355.

    A Gaussian spectrum produces a Gaussian temporal coherence function,
    which is smooth and well-behaved for numerical computation.

    Examples
    --------
    >>> wl, weights = gaussian_spectrum(633e-9, 10e-9, 11)
    >>> jnp.sum(weights)  # Should be 1.0
    """
    lam0: Float[Array, " "] = jnp.asarray(center_wavelength, dtype=jnp.float64)
    fwhm: Float[Array, " "] = jnp.asarray(bandwidth_fwhm, dtype=jnp.float64)
    n: int = int(num_wavelengths)
    fwhm_to_sigma: float = 2.0 * jnp.sqrt(2.0 * jnp.log(2.0))
    sigma: Float[Array, " "] = fwhm / fwhm_to_sigma
    if wavelength_range is None:
        lam_min: Float[Array, " "] = lam0 - 3.0 * sigma
        lam_max: Float[Array, " "] = lam0 + 3.0 * sigma
    else:
        lam_min = jnp.asarray(wavelength_range[0], dtype=jnp.float64)
        lam_max = jnp.asarray(wavelength_range[1], dtype=jnp.float64)
    wavelengths: Float[Array, " n"] = jnp.linspace(lam_min, lam_max, n)
    gaussian_weights: Float[Array, " n"] = jnp.exp(
        -((wavelengths - lam0) ** 2) / (2.0 * sigma**2)
    )
    weights: Float[Array, " n"] = gaussian_weights / jnp.sum(gaussian_weights)
    return wavelengths, weights


@jaxtyped(typechecker=beartype)
def lorentzian_spectrum(
    center_wavelength: ScalarFloat,
    bandwidth_fwhm: ScalarFloat,
    num_wavelengths: ScalarInteger,
    wavelength_range: Optional[Tuple[ScalarFloat, ScalarFloat]] = None,
) -> Tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Generate Lorentzian spectral distribution (natural lineshape).

    The Lorentzian lineshape arises from natural (radiative) broadening
    of atomic transitions. It has broader tails than a Gaussian.

    Parameters
    ----------
    center_wavelength : ScalarFloat
        Center wavelength in meters.
    bandwidth_fwhm : ScalarFloat
        Full width at half maximum (FWHM) of the spectrum in meters.
    num_wavelengths : ScalarInteger
        Number of wavelength sample points.
    wavelength_range : Optional[Tuple[ScalarFloat, ScalarFloat]]
        (min, max) wavelength range in meters. If None, uses
        center +/- 5*FWHM to capture the broad tails.

    Returns
    -------
    wavelengths : Float[Array, " n"]
        Wavelength sample points in meters.
    weights : Float[Array, " n"]
        Normalized spectral weights S(lambda). Sum equals 1.

    Notes
    -----
    The Lorentzian spectrum is:
        S(lambda) = (gamma/2)^2 / ((lambda - lambda_0)^2 + (gamma/2)^2)

    where gamma = FWHM.

    A Lorentzian spectrum produces an exponentially decaying temporal
    coherence function: gamma(tau) ~ exp(-|tau|/tau_c).
    """
    lam0: Float[Array, " "] = jnp.asarray(center_wavelength, dtype=jnp.float64)
    gamma: Float[Array, " "] = jnp.asarray(bandwidth_fwhm, dtype=jnp.float64)
    n: int = int(num_wavelengths)
    if wavelength_range is None:
        lam_min: Float[Array, " "] = lam0 - 5.0 * gamma
        lam_max: Float[Array, " "] = lam0 + 5.0 * gamma
    else:
        lam_min = jnp.asarray(wavelength_range[0], dtype=jnp.float64)
        lam_max = jnp.asarray(wavelength_range[1], dtype=jnp.float64)
    wavelengths: Float[Array, " n"] = jnp.linspace(lam_min, lam_max, n)
    half_gamma: Float[Array, " "] = gamma / 2.0
    lorentzian_weights: Float[Array, " n"] = half_gamma**2 / (
        (wavelengths - lam0) ** 2 + half_gamma**2
    )
    weights: Float[Array, " n"] = lorentzian_weights / jnp.sum(
        lorentzian_weights
    )
    return wavelengths, weights


@jaxtyped(typechecker=beartype)
def rectangular_spectrum(
    center_wavelength: ScalarFloat,
    bandwidth: ScalarFloat,
    num_wavelengths: ScalarInteger,
) -> Tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Generate rectangular (flat-top) spectral distribution.

    Parameters
    ----------
    center_wavelength : ScalarFloat
        Center wavelength in meters.
    bandwidth : ScalarFloat
        Full width of the rectangular spectrum in meters.
    num_wavelengths : ScalarInteger
        Number of wavelength sample points.

    Returns
    -------
    wavelengths : Float[Array, " n"]
        Wavelength sample points in meters.
    weights : Float[Array, " n"]
        Uniform spectral weights. Sum equals 1.

    Notes
    -----
    A rectangular spectrum produces a sinc temporal coherence function:
        gamma(tau) ~ sinc(Delta_nu * tau)

    This is useful for modeling filtered broadband sources.
    """
    lam0: Float[Array, " "] = jnp.asarray(center_wavelength, dtype=jnp.float64)
    delta_lam: Float[Array, " "] = jnp.asarray(bandwidth, dtype=jnp.float64)
    n: int = int(num_wavelengths)
    lam_min: Float[Array, " "] = lam0 - delta_lam / 2.0
    lam_max: Float[Array, " "] = lam0 + delta_lam / 2.0
    wavelengths: Float[Array, " n"] = jnp.linspace(lam_min, lam_max, n)
    weights: Float[Array, " n"] = jnp.ones(n, dtype=jnp.float64) / n
    return wavelengths, weights


@jaxtyped(typechecker=beartype)
def blackbody_spectrum(
    temperature: ScalarFloat,
    wavelength_range: Tuple[ScalarFloat, ScalarFloat],
    num_wavelengths: ScalarInteger,
) -> Tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Generate blackbody (Planck) spectral distribution.

    Parameters
    ----------
    temperature : ScalarFloat
        Temperature of the blackbody in Kelvin.
    wavelength_range : Tuple[ScalarFloat, ScalarFloat]
        (min, max) wavelength range in meters.
    num_wavelengths : ScalarInteger
        Number of wavelength sample points.

    Returns
    -------
    wavelengths : Float[Array, " n"]
        Wavelength sample points in meters.
    weights : Float[Array, " n"]
        Normalized spectral weights following Planck distribution.
        Sum equals 1.

    Notes
    -----
    The Planck distribution is:
        B(lambda, T) = (2*h*c^2 / lambda^5) / (exp(h*c/(lambda*k*T)) - 1)

    where h is Planck's constant, c is speed of light, k is Boltzmann's
    constant.

    For a 5800 K blackbody (Sun), the peak is near 500 nm.
    """
    planck_constant: float = 6.62607015e-34
    boltzmann_constant: float = 1.380649e-23
    c: float = C_LIGHT
    temp: Float[Array, " "] = jnp.asarray(temperature, dtype=jnp.float64)
    lam_min: Float[Array, " "] = jnp.asarray(
        wavelength_range[0], dtype=jnp.float64
    )
    lam_max: Float[Array, " "] = jnp.asarray(
        wavelength_range[1], dtype=jnp.float64
    )
    n: int = int(num_wavelengths)
    wavelengths: Float[Array, " n"] = jnp.linspace(lam_min, lam_max, n)
    planck_exponent: Float[Array, " n"] = (
        planck_constant * c / (wavelengths * boltzmann_constant * temp)
    )
    planck_exponent_clipped: Float[Array, " n"] = jnp.clip(
        planck_exponent, -700, 700
    )
    planck_weights: Float[Array, " n"] = (
        2.0 * planck_constant * c**2 / wavelengths**5
    ) / (jnp.exp(planck_exponent_clipped) - 1.0)
    weights: Float[Array, " n"] = planck_weights / jnp.sum(planck_weights)
    return wavelengths, weights


@jaxtyped(typechecker=beartype)
def coherence_length(
    center_wavelength: ScalarFloat,
    bandwidth: ScalarFloat,
) -> Float[Array, " "]:
    """Calculate coherence length L_c = lambda^2 / Delta_lambda.

    The coherence length is the optical path difference at which the
    fringe visibility drops to ~1/e for a Gaussian spectrum.

    Parameters
    ----------
    center_wavelength : ScalarFloat
        Center wavelength in meters.
    bandwidth : ScalarFloat
        Spectral bandwidth (FWHM) in meters.

    Returns
    -------
    l_c : Float[Array, " "]
        Coherence length in meters.

    Notes
    -----
    For a Gaussian spectrum with FWHM bandwidth Delta_lambda:
        L_c = (2*ln(2)/pi) * lambda^2 / Delta_lambda
            ~ 0.44 * lambda^2 / Delta_lambda

    For a Lorentzian spectrum:
        L_c = lambda^2 / (pi * Delta_lambda)

    This function uses the Gaussian formula.

    Examples
    --------
    >>> # HeNe laser with 1 GHz linewidth (very narrow)
    >>> # Delta_lambda = lambda^2 * Delta_nu / c ~ 1e-15 m
    >>> l_c = coherence_length(633e-9, 1e-15)  # ~ 400 m

    >>> # White LED with 50 nm FWHM
    >>> l_c = coherence_length(550e-9, 50e-9)  # ~ 6 um
    """
    lam: Float[Array, " "] = jnp.asarray(center_wavelength, dtype=jnp.float64)
    delta_lam: Float[Array, " "] = jnp.asarray(bandwidth, dtype=jnp.float64)
    gaussian_coherence_factor: float = 2.0 * jnp.log(2.0) / jnp.pi
    l_c: Float[Array, " "] = gaussian_coherence_factor * lam**2 / delta_lam
    return l_c


@jaxtyped(typechecker=beartype)
def coherence_time(
    center_wavelength: ScalarFloat,
    bandwidth: ScalarFloat,
) -> Float[Array, " "]:
    """Calculate coherence time tau_c = lambda^2 / (c * Delta_lambda).

    The coherence time is the time delay at which the temporal coherence
    function drops to ~1/e for a Gaussian spectrum.

    Parameters
    ----------
    center_wavelength : ScalarFloat
        Center wavelength in meters.
    bandwidth : ScalarFloat
        Spectral bandwidth (FWHM) in meters.

    Returns
    -------
    tau_c : Float[Array, " "]
        Coherence time in seconds.

    Notes
    -----
    tau_c = L_c / c, where L_c is the coherence length and c is the
    speed of light.

    For a HeNe laser with 1 GHz linewidth: tau_c ~ 1 ns
    For white light with 100 nm bandwidth: tau_c ~ 10 fs
    """
    l_c: Float[Array, " "] = coherence_length(center_wavelength, bandwidth)
    tau_c: Float[Array, " "] = l_c / C_LIGHT
    return tau_c


@jaxtyped(typechecker=beartype)
def bandwidth_from_coherence_length(
    center_wavelength: ScalarFloat,
    coh_length: ScalarFloat,
) -> Float[Array, " "]:
    """Calculate bandwidth from coherence length.

    Inverse of coherence_length function.

    Parameters
    ----------
    center_wavelength : ScalarFloat
        Center wavelength in meters.
    coh_length : ScalarFloat
        Coherence length in meters.

    Returns
    -------
    bandwidth : Float[Array, " "]
        Spectral bandwidth (FWHM) in meters.
    """
    lam: Float[Array, " "] = jnp.asarray(center_wavelength, dtype=jnp.float64)
    l_c: Float[Array, " "] = jnp.asarray(coh_length, dtype=jnp.float64)
    gaussian_coherence_factor: float = 2.0 * jnp.log(2.0) / jnp.pi
    bandwidth: Float[Array, " "] = gaussian_coherence_factor * lam**2 / l_c
    return bandwidth


@jaxtyped(typechecker=beartype)
def spectral_phase_from_dispersion(
    wavelengths: Float[Array, " n"],
    center_wavelength: ScalarFloat,
    gdd: ScalarFloat = 0.0,
    tod: ScalarFloat = 0.0,
) -> Float[Array, " n"]:
    """Calculate spectral phase from group delay dispersion.

    Models the effect of material dispersion on ultrashort pulses.

    Parameters
    ----------
    wavelengths : Float[Array, " n"]
        Wavelength sample points in meters.
    center_wavelength : ScalarFloat
        Center wavelength in meters.
    gdd : ScalarFloat
        Group delay dispersion in s^2 (fs^2 = 1e-30 s^2).
    tod : ScalarFloat
        Third-order dispersion in s^3.

    Returns
    -------
    phase : Float[Array, " n"]
        Spectral phase in radians for each wavelength.

    Notes
    -----
    The spectral phase is expanded as:
        phi(omega) = phi_0 + phi_1*(omega-omega_0) + phi_2*(omega-omega_0)^2/2
                   + phi_3*(omega-omega_0)^3/6 + ...

    where phi_2 = GDD (group delay dispersion) and phi_3 = TOD.

    GDD causes pulse broadening. Typical values:
    - Fused silica: ~36 fs^2/mm at 800 nm
    - BK7 glass: ~45 fs^2/mm at 800 nm
    """
    lam0: Float[Array, " "] = jnp.asarray(center_wavelength, dtype=jnp.float64)
    gdd_val: Float[Array, " "] = jnp.asarray(gdd, dtype=jnp.float64)
    tod_val: Float[Array, " "] = jnp.asarray(tod, dtype=jnp.float64)
    omega: Float[Array, " n"] = 2.0 * jnp.pi * C_LIGHT / wavelengths
    omega0: Float[Array, " "] = 2.0 * jnp.pi * C_LIGHT / lam0
    delta_omega: Float[Array, " n"] = omega - omega0
    phase: Float[Array, " n"] = (
        0.5 * gdd_val * delta_omega**2 + (1.0 / 6.0) * tod_val * delta_omega**3
    )
    return phase
