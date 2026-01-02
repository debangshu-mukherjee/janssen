"""Partially coherent source models.

Extended Summary
----------------
This module provides models for common partially coherent light sources
encountered in optical experiments. Each source model combines spatial
and/or temporal partial coherence characteristics.

Real light sources are never perfectly coherent:
- Lasers have finite linewidth (temporal) and may have mode structure (spatial)
- LEDs are spatially incoherent and have broad spectra
- Thermal sources are fully incoherent both spatially and temporally
- Synchrotron sources have anisotropic spatial coherence

Routine Listings
----------------
led_source : function
    Model an LED source with spatial and temporal partial coherence
thermal_source : function
    Model a thermal (blackbody) source
synchrotron_source : function
    Model a synchrotron X-ray source with anisotropic coherence
laser_with_mode_noise : function
    Model a laser with imperfect mode purity
multimode_fiber_output : function
    Model the output of a multimode optical fiber

Notes
-----
Source models return CoherentModeSet and/or spectral information that
can be used for propagation through optical systems.

References
----------
1. Goodman, J. W. "Statistical Optics" (2015)
2. Mandel, L. & Wolf, E. "Optical Coherence and Quantum Optics" (1995)
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from janssen.utils import (
    CoherentModeSet,
    ScalarFloat,
    ScalarInteger,
    make_coherent_mode_set,
)

from .modes import gaussian_schell_model_modes, hermite_gaussian_modes
from .temporal import gaussian_spectrum


@jaxtyped(typechecker=beartype)
def led_source(
    center_wavelength: ScalarFloat,
    bandwidth_fwhm: ScalarFloat,
    spatial_coherence_width: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Tuple[ScalarInteger, ScalarInteger],
    num_spatial_modes: ScalarInteger = 10,
    num_spectral_samples: ScalarInteger = 11,
) -> Tuple[CoherentModeSet, Float[Array, " n"], Float[Array, " n"]]:
    """Model an LED source with both spatial and temporal partial coherence.

    LEDs are spatially incoherent (extended source) and temporally
    partially coherent (broad spectrum, typically 20-50 nm FWHM).

    Parameters
    ----------
    center_wavelength : ScalarFloat
        Center wavelength in meters (e.g., 530e-9 for green LED).
    bandwidth_fwhm : ScalarFloat
        Spectral bandwidth (FWHM) in meters (typically 20-50 nm).
    spatial_coherence_width : ScalarFloat
        Spatial coherence width in meters. For an LED die of size D
        at distance z, this is approximately lambda * z / D.
    dx : ScalarFloat
        Pixel spacing in meters.
    grid_size : Tuple[ScalarInteger, ScalarInteger]
        (height, width) of the grid in pixels.
    num_spatial_modes : ScalarInteger
        Number of spatial coherent modes. Default is 10.
    num_spectral_samples : ScalarInteger
        Number of wavelength samples. Default is 11.

    Returns
    -------
    mode_set : CoherentModeSet
        Spatial coherent mode set at center wavelength.
    wavelengths : Float[Array, " n"]
        Wavelength sample points in meters.
    spectral_weights : Float[Array, " n"]
        Normalized spectral weights.

    Notes
    -----
    For complete LED simulation, propagate the mode_set at each wavelength
    and compute weighted intensity sum.

    Typical LED parameters:
    - White LED: 400-700 nm, very broad
    - Red LED: 630 nm, ~20 nm FWHM
    - Green LED: 530 nm, ~30 nm FWHM
    - Blue LED: 470 nm, ~25 nm FWHM

    Spatial coherence depends on viewing distance and LED die size:
    - sigma_c ~ lambda * z / D_die
    - Typical LED die: 0.3-1 mm
    - At 10 cm distance: sigma_c ~ 50-200 um
    """
    # Spatial coherence: use Gaussian Schell-model
    # Beam width should be larger than coherence width for LED
    beam_width: Float[Array, " "] = jnp.asarray(
        spatial_coherence_width * 3.0, dtype=jnp.float64
    )

    mode_set: CoherentModeSet = gaussian_schell_model_modes(
        wavelength=center_wavelength,
        dx=dx,
        grid_size=grid_size,
        beam_width=beam_width,
        coherence_width=spatial_coherence_width,
        num_modes=num_spatial_modes,
    )

    # Temporal/spectral: Gaussian spectrum
    wavelengths: Float[Array, " n"]
    spectral_weights: Float[Array, " n"]
    wavelengths, spectral_weights = gaussian_spectrum(
        center_wavelength=center_wavelength,
        bandwidth_fwhm=bandwidth_fwhm,
        num_wavelengths=num_spectral_samples,
    )

    return mode_set, wavelengths, spectral_weights


@jaxtyped(typechecker=beartype)
def thermal_source(
    temperature: ScalarFloat,
    source_diameter: ScalarFloat,
    propagation_distance: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Tuple[ScalarInteger, ScalarInteger],
    wavelength_range: Tuple[ScalarFloat, ScalarFloat],
    center_wavelength: Optional[ScalarFloat] = None,
    num_modes: ScalarInteger = 20,
    num_spectral_samples: ScalarInteger = 21,
) -> Tuple[CoherentModeSet, Float[Array, " n"], Float[Array, " n"]]:
    """Model a thermal (blackbody) source.

    Thermal sources are fully incoherent at the source but develop
    partial spatial coherence after propagation (van Cittert-Zernike).

    Parameters
    ----------
    temperature : ScalarFloat
        Temperature in Kelvin.
    source_diameter : ScalarFloat
        Diameter of the thermal source in meters.
    propagation_distance : ScalarFloat
        Distance from source to observation plane in meters.
    dx : ScalarFloat
        Pixel spacing in meters.
    grid_size : Tuple[ScalarInteger, ScalarInteger]
        (height, width) of the grid in pixels.
    wavelength_range : Tuple[ScalarFloat, ScalarFloat]
        (min, max) wavelength range in meters.
    center_wavelength : Optional[ScalarFloat]
        Center wavelength for mode calculation. If None, uses
        Wien's displacement law: lambda_peak = 2.898e-3 / T.
    num_modes : ScalarInteger
        Number of spatial coherent modes. Default is 20.
    num_spectral_samples : ScalarInteger
        Number of wavelength samples. Default is 21.

    Returns
    -------
    mode_set : CoherentModeSet
        Spatial coherent mode set.
    wavelengths : Float[Array, " n"]
        Wavelength sample points in meters.
    spectral_weights : Float[Array, " n"]
        Blackbody spectral weights.

    Notes
    -----
    Wien's displacement law gives the peak wavelength:
        lambda_peak = 2.898e-3 / T (meters * Kelvin)

    Examples
    --------
    - Sun (5800 K): peak at 500 nm
    - Incandescent bulb (2800 K): peak at 1035 nm
    - Human body (310 K): peak at 9.3 um
    """
    from .temporal import blackbody_spectrum

    # Coherence width from van Cittert-Zernike
    if center_wavelength is None:
        # Wien's law
        center_wl: Float[Array, " "] = 2.898e-3 / jnp.asarray(
            temperature, dtype=jnp.float64
        )
    else:
        center_wl = jnp.asarray(center_wavelength, dtype=jnp.float64)

    # Coherence width
    coherence_width: Float[Array, " "] = (
        0.44
        * center_wl
        * jnp.asarray(propagation_distance, dtype=jnp.float64)
        / jnp.asarray(source_diameter, dtype=jnp.float64)
    )

    # Extended beam width (thermal source is typically large)
    beam_width: Float[Array, " "] = coherence_width * 5.0

    # Spatial modes
    mode_set: CoherentModeSet = gaussian_schell_model_modes(
        wavelength=center_wl,
        dx=dx,
        grid_size=grid_size,
        beam_width=beam_width,
        coherence_width=coherence_width,
        num_modes=num_modes,
    )

    # Blackbody spectrum
    wavelengths: Float[Array, " n"]
    spectral_weights: Float[Array, " n"]
    wavelengths, spectral_weights = blackbody_spectrum(
        temperature=temperature,
        wavelength_range=wavelength_range,
        num_wavelengths=num_spectral_samples,
    )

    return mode_set, wavelengths, spectral_weights


@jaxtyped(typechecker=beartype)
def synchrotron_source(
    center_wavelength: ScalarFloat,
    horizontal_coherence: ScalarFloat,
    vertical_coherence: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Tuple[ScalarInteger, ScalarInteger],
    num_modes_h: ScalarInteger = 5,
    num_modes_v: ScalarInteger = 5,
) -> CoherentModeSet:
    """Model a synchrotron X-ray source with anisotropic coherence.

    Synchrotron sources have different coherence properties in the
    horizontal (orbit plane) and vertical directions due to the
    different source sizes.

    Parameters
    ----------
    center_wavelength : ScalarFloat
        Wavelength in meters (e.g., 1e-10 for 1 Angstrom X-rays).
    horizontal_coherence : ScalarFloat
        Horizontal coherence width in meters.
    vertical_coherence : ScalarFloat
        Vertical coherence width in meters.
    dx : ScalarFloat
        Pixel spacing in meters.
    grid_size : Tuple[ScalarInteger, ScalarInteger]
        (height, width) of the grid in pixels.
    num_modes_h : ScalarInteger
        Number of horizontal modes. Default is 5.
    num_modes_v : ScalarInteger
        Number of vertical modes. Default is 5.

    Returns
    -------
    mode_set : CoherentModeSet
        Anisotropic coherent mode set.

    Notes
    -----
    Synchrotron coherence is typically:
    - Horizontal: larger source size, smaller coherence
    - Vertical: smaller source size, larger coherence

    For modern diffraction-limited storage rings:
    - Coherent fraction can exceed 10%
    - Source sizes approach the diffraction limit

    The modes are products of 1D Hermite-Gaussians with different
    widths in h and v directions.
    """
    hh: int = int(grid_size[0])
    ww: int = int(grid_size[1])

    # Mode widths (geometric mean approach)
    sigma_h: Float[Array, " "] = jnp.asarray(
        horizontal_coherence, dtype=jnp.float64
    )
    sigma_v: Float[Array, " "] = jnp.asarray(
        vertical_coherence, dtype=jnp.float64
    )

    # Create coordinate grids
    y: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * dx
    x: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)

    # Normalized coordinates for each direction
    x_norm: Float[Array, " hh ww"] = xx * jnp.sqrt(2.0) / sigma_h
    y_norm: Float[Array, " hh ww"] = yy * jnp.sqrt(2.0) / sigma_v

    # Gaussian envelopes
    gaussian: Float[Array, " hh ww"] = jnp.exp(
        -(xx**2) / sigma_h**2 - (yy**2) / sigma_v**2
    )

    def _hermite_polynomial(
        order: int, x: Float[Array, "..."]
    ) -> Float[Array, "..."]:
        """Compute physicist's Hermite polynomial."""
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

    # Generate 2D modes as products of 1D modes
    n_h: int = int(num_modes_h)
    n_v: int = int(num_modes_v)
    total_modes: int = n_h * n_v

    modes_list = []
    weights_list = []

    for ih in range(n_h):
        h_h = _hermite_polynomial(ih, x_norm)
        weight_h = jnp.exp(-ih / 2.0)  # Thermal-like distribution

        for iv in range(n_v):
            h_v = _hermite_polynomial(iv, y_norm)
            weight_v = jnp.exp(-iv / 2.0)

            mode = h_h * h_v * gaussian

            # Normalize
            energy = jnp.sum(jnp.abs(mode) ** 2)
            mode = mode / jnp.sqrt(energy + 1e-20)

            modes_list.append(mode)
            weights_list.append(weight_h * weight_v)

    modes: Complex[Array, " num_modes hh ww"] = jnp.stack(
        modes_list, axis=0
    ).astype(jnp.complex128)
    weights: Float[Array, " num_modes"] = jnp.array(weights_list)
    weights = weights / jnp.sum(weights)

    return make_coherent_mode_set(
        modes=modes,
        weights=weights,
        wavelength=center_wavelength,
        dx=dx,
        z_position=0.0,
        polarization=False,
    )


@jaxtyped(typechecker=beartype)
def laser_with_mode_noise(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Tuple[ScalarInteger, ScalarInteger],
    beam_waist: ScalarFloat,
    mode_purity: ScalarFloat,
    num_modes: ScalarInteger = 5,
) -> CoherentModeSet:
    """Model a laser with imperfect mode purity (partial spatial coherence).

    Real lasers may have contributions from higher-order transverse modes
    due to cavity imperfections, thermal effects, or multimode operation.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength in meters.
    dx : ScalarFloat
        Pixel spacing in meters.
    grid_size : Tuple[ScalarInteger, ScalarInteger]
        (height, width) of the grid in pixels.
    beam_waist : ScalarFloat
        1/e^2 intensity radius (waist) of the fundamental mode in meters.
    mode_purity : ScalarFloat
        Mode purity factor between 0 and 1.
        1.0 = perfect TEM00 (fully coherent)
        0.0 = equal power in all modes (highly incoherent)
    num_modes : ScalarInteger
        Number of transverse modes to include. Default is 5.

    Returns
    -------
    mode_set : CoherentModeSet
        Laser mode set with specified purity.

    Notes
    -----
    The mode weights are:
        w_0 = mode_purity
        w_n = (1 - mode_purity) * exp(-n) / Z  (for n > 0)

    where Z is a normalization constant.

    Mode purity is related to M^2 beam quality factor:
        M^2 = sum_n (2n + 1) * w_n

    For perfect TEM00: M^2 = 1
    For mixed modes: M^2 > 1
    """
    n_modes: int = int(num_modes)
    purity: Float[Array, " "] = jnp.asarray(mode_purity, dtype=jnp.float64)
    purity = jnp.clip(purity, 0.0, 1.0)

    # Generate Hermite-Gaussian modes
    hg_modes: CoherentModeSet = hermite_gaussian_modes(
        wavelength=wavelength,
        dx=dx,
        grid_size=grid_size,
        beam_waist=beam_waist,
        max_order=int(jnp.sqrt(n_modes)) + 1,  # Approximate order needed
    )

    # Take first n_modes
    modes: Complex[Array, " num_modes hh ww"] = hg_modes.modes[:n_modes]

    # Custom weights based on mode purity
    n_arr: Float[Array, " num_modes"] = jnp.arange(n_modes, dtype=jnp.float64)

    # Weight for TEM00
    w0: Float[Array, " "] = purity

    # Weights for higher modes (exponentially decreasing)
    higher_mode_weights: Float[Array, " num_modes"] = jnp.where(
        n_arr == 0,
        0.0,
        jnp.exp(-n_arr),
    )

    # Normalize higher mode weights and scale by (1 - purity)
    higher_mode_sum: Float[Array, " "] = jnp.sum(higher_mode_weights)
    higher_mode_weights = (
        (1.0 - purity) * higher_mode_weights / (higher_mode_sum + 1e-20)
    )

    # Combine
    weights: Float[Array, " num_modes"] = jnp.where(
        n_arr == 0,
        w0,
        higher_mode_weights,
    )

    return make_coherent_mode_set(
        modes=modes,
        weights=weights,
        wavelength=wavelength,
        dx=dx,
        z_position=0.0,
        polarization=False,
        normalize_weights=True,
    )


@jaxtyped(typechecker=beartype)
def multimode_fiber_output(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Tuple[ScalarInteger, ScalarInteger],
    fiber_core_radius: ScalarFloat,
    num_modes: ScalarInteger = 10,
    mode_distribution: str = "uniform",
) -> CoherentModeSet:
    """Model the output of a multimode optical fiber.

    Multimode fibers support many propagating modes, leading to
    spatially incoherent output (speckle patterns).

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength in meters.
    dx : ScalarFloat
        Pixel spacing in meters.
    grid_size : Tuple[ScalarInteger, ScalarInteger]
        (height, width) of the grid in pixels.
    fiber_core_radius : ScalarFloat
        Radius of the fiber core in meters.
    num_modes : ScalarInteger
        Number of fiber modes to include. Default is 10.
    mode_distribution : str
        Mode power distribution: "uniform" or "thermal".
        Default is "uniform".

    Returns
    -------
    mode_set : CoherentModeSet
        Fiber output mode set.

    Notes
    -----
    The number of modes in a step-index fiber is approximately:
        N ~ V^2 / 2

    where V = (2*pi/lambda) * a * NA is the V-number.

    For a typical 50 um core, 0.2 NA fiber at 633 nm:
        V ~ 100, N ~ 5000 modes

    In practice, we use a smaller number of modes to represent
    the dominant contributions.
    """
    hh: int = int(grid_size[0])
    ww: int = int(grid_size[1])
    n_modes: int = int(num_modes)
    a: Float[Array, " "] = jnp.asarray(fiber_core_radius, dtype=jnp.float64)

    # Create coordinate grids
    y: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * dx
    x: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)
    r: Float[Array, " hh ww"] = jnp.sqrt(xx**2 + yy**2)
    phi: Float[Array, " hh ww"] = jnp.arctan2(yy, xx)

    # Fiber core mask
    core_mask: Float[Array, " hh ww"] = jnp.where(r <= a, 1.0, 0.0)

    # Generate LP modes (simplified: using cos/sin variations)
    # LP_lm modes have l azimuthal nodes and m radial nodes
    modes_list = []

    mode_idx = 0
    for l in range(int(jnp.sqrt(n_modes)) + 2):
        for m in range(1, int(jnp.sqrt(n_modes)) + 2):
            if mode_idx >= n_modes:
                break

            # Radial part: simplified Bessel-like profile
            # Using Gaussian-Laguerre as approximation
            rho = r / a
            radial = jnp.exp(-(rho**2) / 2.0) * (rho ** abs(l))

            # Azimuthal part
            if l == 0:
                azimuthal = jnp.ones_like(phi)
            else:
                # Include both cos and sin variants
                azimuthal = jnp.cos(l * phi)

            mode = radial * azimuthal * core_mask

            # Normalize
            energy = jnp.sum(jnp.abs(mode) ** 2)
            if energy > 1e-20:
                mode = mode / jnp.sqrt(energy)
                modes_list.append(mode)
                mode_idx += 1

            if mode_idx >= n_modes:
                break

    # Pad with zeros if we don't have enough modes
    while len(modes_list) < n_modes:
        modes_list.append(jnp.zeros((hh, ww)))

    modes: Complex[Array, " num_modes hh ww"] = jnp.stack(
        modes_list[:n_modes], axis=0
    ).astype(jnp.complex128)

    # Mode weights
    if mode_distribution == "uniform":
        weights: Float[Array, " num_modes"] = jnp.ones(n_modes) / n_modes
    elif mode_distribution == "thermal":
        n_arr = jnp.arange(n_modes, dtype=jnp.float64)
        weights = jnp.exp(-n_arr / 3.0)
        weights = weights / jnp.sum(weights)
    else:
        raise ValueError(f"Unknown mode distribution: {mode_distribution}")

    return make_coherent_mode_set(
        modes=modes,
        weights=weights,
        wavelength=wavelength,
        dx=dx,
        z_position=0.0,
        polarization=False,
    )
