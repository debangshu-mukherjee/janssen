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
    Model an LED source with spatial and temporal partial coherence.
thermal_source : function
    Model a thermal (blackbody) source.
synchrotron_source : function
    Model a synchrotron X-ray source with anisotropic coherence.
laser_with_mode_noise : function
    Model a laser with imperfect mode purity.
multimode_fiber_output : function
    Model the output of a multimode optical fiber.
_synchrotron_source_impl : function, internal (pure JAX)
    JIT-compiled synchrotron source mode generation.
_multimode_fiber_output_impl : function, internal (pure JAX)
    JIT-compiled multimode fiber mode generation.

Notes
-----
Source models return CoherentModeSet and/or spectral information that
can be used for propagation through optical systems.

References
----------
1. Goodman, J. W. "Statistical Optics" (2015)
2. Mandel, L. & Wolf, E. "Optical Coherence and Quantum Optics" (1995)
"""

from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from janssen.types import (
    CoherentModeSet,
    ScalarFloat,
    ScalarInteger,
    make_coherent_mode_set,
)

from .modes import (
    gaussian_schell_model_modes,
    hermite_gaussian_modes,
    thermal_mode_weights,
)
from .temporal import blackbody_spectrum, gaussian_spectrum


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
    if center_wavelength is None:
        center_wl: Float[Array, " "] = 2.898e-3 / jnp.asarray(
            temperature, dtype=jnp.float64
        )
    else:
        center_wl = jnp.asarray(center_wavelength, dtype=jnp.float64)

    coherence_width: Float[Array, " "] = (
        0.44
        * center_wl
        * jnp.asarray(propagation_distance, dtype=jnp.float64)
        / jnp.asarray(source_diameter, dtype=jnp.float64)
    )

    beam_width: Float[Array, " "] = coherence_width * 5.0

    mode_set: CoherentModeSet = gaussian_schell_model_modes(
        wavelength=center_wl,
        dx=dx,
        grid_size=grid_size,
        beam_width=beam_width,
        coherence_width=coherence_width,
        num_modes=num_modes,
    )

    wavelengths: Float[Array, " n"]
    spectral_weights: Float[Array, " n"]
    wavelengths, spectral_weights = blackbody_spectrum(
        temperature=temperature,
        wavelength_range=wavelength_range,
        num_wavelengths=num_spectral_samples,
    )

    return mode_set, wavelengths, spectral_weights


def _hermite_polynomial(
    order: Float[Array, " "], x: Float[Array, " hh ww"]
) -> Float[Array, " hh ww"]:
    """Compute physicist's Hermite polynomial using fori_loop.

    Parameters
    ----------
    order : Float[Array, " "]
        Polynomial order (will be converted to int for loop bound).
    x : Float[Array, " hh ww"]
        Input array.

    Returns
    -------
    Float[Array, " hh ww"]
        Hermite polynomial H_n(x).
    """

    def _body_fn_int(
        k: int,
        carry: Tuple[Float[Array, " hh ww"], Float[Array, " hh ww"]],
    ) -> Tuple[Float[Array, " hh ww"], Float[Array, " hh ww"]]:
        h_prev2, h_prev1 = carry
        h_curr: Float[Array, " hh ww"] = (
            2.0 * x * h_prev1 - 2.0 * (k - 1) * h_prev2
        )
        return h_prev1, h_curr

    h0: Float[Array, " hh ww"] = jnp.ones_like(x)
    h1: Float[Array, " hh ww"] = 2.0 * x

    _, h_n = jax.lax.fori_loop(
        2, order.astype(jnp.int32) + 1, _body_fn_int, (h0, h1)
    )

    result: Float[Array, " hh ww"] = jnp.where(
        order == 0, h0, jnp.where(order == 1, h1, h_n)
    )
    return result


def _generate_synchrotron_mode_indices(
    n_h: int, n_v: int
) -> Float[Array, " num_modes 2"]:
    """Generate (ih, iv) mode indices for synchrotron source.

    Parameters
    ----------
    n_h : int
        Number of horizontal modes.
    n_v : int
        Number of vertical modes.

    Returns
    -------
    Float[Array, " num_modes 2"]
        Array of (ih, iv) mode index pairs.
    """
    mode_indices_list = []
    for ih in range(n_h):
        for iv in range(n_v):
            mode_indices_list.append((ih, iv))
    mode_indices: Float[Array, " num_modes 2"] = jnp.array(
        mode_indices_list, dtype=jnp.float64
    )
    return mode_indices


@partial(jax.jit, static_argnums=(3, 4))
def _synchrotron_source_impl(
    dx: Float[Array, " "],
    horizontal_coherence: Float[Array, " "],
    vertical_coherence: Float[Array, " "],
    hh: int,
    ww: int,
    mode_indices: Float[Array, " num_modes 2"],
) -> Tuple[Complex[Array, " num_modes hh ww"], Float[Array, " num_modes"]]:
    """JIT-compiled synchrotron source mode generation.

    Pure JAX implementation with static grid dimensions for efficient
    JIT compilation. Use this directly in pure JAX workflows.

    Parameters
    ----------
    dx : Float[Array, " "]
        Pixel spacing in meters.
    horizontal_coherence : Float[Array, " "]
        Horizontal coherence width in meters.
    vertical_coherence : Float[Array, " "]
        Vertical coherence width in meters.
    hh : int
        Grid height in pixels (static).
    ww : int
        Grid width in pixels (static).
    mode_indices : Float[Array, " num_modes 2"]
        Array of (n_h, n_v) mode index pairs.

    Returns
    -------
    modes : Complex[Array, " num_modes hh ww"]
        Normalized anisotropic Gaussian mode fields.
    weights : Float[Array, " num_modes"]
        Mode weights based on coherence anisotropy.
    """
    sigma_h: Float[Array, " "] = horizontal_coherence
    sigma_v: Float[Array, " "] = vertical_coherence

    y: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * dx
    x: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)

    x_norm: Float[Array, " hh ww"] = xx * jnp.sqrt(2.0) / sigma_h
    y_norm: Float[Array, " hh ww"] = yy * jnp.sqrt(2.0) / sigma_v

    gaussian: Float[Array, " hh ww"] = jnp.exp(
        -(xx**2) / sigma_h**2 - (yy**2) / sigma_v**2
    )

    def _compute_single_mode_int(
        indices: Float[Array, " 2"],
    ) -> Tuple[Complex[Array, " hh ww"], Float[Array, " "]]:
        ih: Float[Array, " "] = indices[0]
        iv: Float[Array, " "] = indices[1]

        h_h: Float[Array, " hh ww"] = _hermite_polynomial(ih, x_norm)
        h_v: Float[Array, " hh ww"] = _hermite_polynomial(iv, y_norm)
        thermal_weight_h: Float[Array, " "] = jnp.exp(-ih / 2.0)
        thermal_weight_v: Float[Array, " "] = jnp.exp(-iv / 2.0)

        mode: Float[Array, " hh ww"] = h_h * h_v * gaussian
        mode_energy: Float[Array, " "] = jnp.sum(jnp.abs(mode) ** 2)
        mode_normalized: Complex[Array, " hh ww"] = (
            mode / jnp.sqrt(mode_energy + 1e-20)
        ).astype(jnp.complex128)

        weight: Float[Array, " "] = thermal_weight_h * thermal_weight_v
        return mode_normalized, weight

    modes: Complex[Array, " num_modes hh ww"]
    weights: Float[Array, " num_modes"]
    modes, weights = jax.vmap(_compute_single_mode_int)(mode_indices)

    weights = weights / jnp.sum(weights)

    return modes, weights


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
    n_h: int = int(num_modes_h)
    n_v: int = int(num_modes_v)

    dx_arr: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    h_coh_arr: Float[Array, " "] = jnp.asarray(
        horizontal_coherence, dtype=jnp.float64
    )
    v_coh_arr: Float[Array, " "] = jnp.asarray(
        vertical_coherence, dtype=jnp.float64
    )

    mode_indices: Float[Array, " num_modes 2"] = (
        _generate_synchrotron_mode_indices(n_h, n_v)
    )

    modes, weights = _synchrotron_source_impl(
        dx_arr, h_coh_arr, v_coh_arr, hh, ww, mode_indices
    )

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

    max_order: int = int(jnp.sqrt(n_modes)) + 1
    hg_num_modes: int = (max_order + 1) * (max_order + 2) // 2
    hg_weights: Float[Array, " hg_num_modes"] = thermal_mode_weights(
        hg_num_modes
    )

    hg_modes: CoherentModeSet = hermite_gaussian_modes(
        wavelength=wavelength,
        dx=dx,
        grid_size=grid_size,
        beam_waist=beam_waist,
        max_order=max_order,
        mode_weights=hg_weights,
    )

    modes: Complex[Array, " num_modes hh ww"] = hg_modes.modes[:n_modes]

    n_arr: Float[Array, " num_modes"] = jnp.arange(n_modes, dtype=jnp.float64)

    tem00_weight: Float[Array, " "] = purity

    higher_mode_weights: Float[Array, " num_modes"] = jnp.where(
        n_arr == 0,
        0.0,
        jnp.exp(-n_arr),
    )

    higher_mode_sum: Float[Array, " "] = jnp.sum(higher_mode_weights)
    higher_mode_weights = (
        (1.0 - purity) * higher_mode_weights / (higher_mode_sum + 1e-20)
    )

    weights: Float[Array, " num_modes"] = jnp.where(
        n_arr == 0,
        tem00_weight,
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


def _generate_fiber_mode_indices(n_modes: int) -> Float[Array, " num_modes 2"]:
    """Generate (azimuthal_idx, radial_idx) indices for fiber modes.

    Parameters
    ----------
    n_modes : int
        Number of modes to generate.

    Returns
    -------
    Float[Array, " num_modes 2"]
        Array of (azimuthal_idx, radial_idx) index pairs.
    """
    mode_indices_list = []
    max_order = int(n_modes**0.5) + 2
    for azimuthal_idx in range(max_order):
        for radial_idx in range(1, max_order):
            if len(mode_indices_list) >= n_modes:
                break
            mode_indices_list.append((azimuthal_idx, radial_idx))
        if len(mode_indices_list) >= n_modes:
            break
    mode_indices: Float[Array, " num_modes 2"] = jnp.array(
        mode_indices_list[:n_modes], dtype=jnp.float64
    )
    return mode_indices


@partial(jax.jit, static_argnums=(2, 3))
def _multimode_fiber_output_impl(
    dx: Float[Array, " "],
    fiber_core_radius: Float[Array, " "],
    hh: int,
    ww: int,
    mode_indices: Float[Array, " num_modes 2"],
) -> Complex[Array, " num_modes hh ww"]:
    """JIT-compiled multimode fiber mode generation.

    Pure JAX implementation with static grid dimensions for efficient
    JIT compilation. Use this directly in pure JAX workflows.

    Parameters
    ----------
    dx : Float[Array, " "]
        Pixel spacing in meters.
    fiber_core_radius : Float[Array, " "]
        Fiber core radius in meters.
    hh : int
        Grid height in pixels (static).
    ww : int
        Grid width in pixels (static).
    mode_indices : Float[Array, " num_modes 2"]
        Array of (azimuthal_idx, radial_idx) index pairs.

    Returns
    -------
    modes : Complex[Array, " num_modes hh ww"]
        Normalized LP-like fiber mode fields.
    """
    a: Float[Array, " "] = fiber_core_radius

    y: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * dx
    x: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)
    r: Float[Array, " hh ww"] = jnp.sqrt(xx**2 + yy**2)
    phi: Float[Array, " hh ww"] = jnp.arctan2(yy, xx)

    core_mask: Float[Array, " hh ww"] = jnp.where(r <= a, 1.0, 0.0)
    rho: Float[Array, " hh ww"] = r / a

    def _compute_single_mode_int(
        indices: Float[Array, " 2"],
    ) -> Complex[Array, " hh ww"]:
        azimuthal_idx: Float[Array, " "] = indices[0]

        radial_profile: Float[Array, " hh ww"] = jnp.exp(-(rho**2) / 2.0) * (
            rho ** jnp.abs(azimuthal_idx)
        )
        azimuthal_profile: Float[Array, " hh ww"] = jnp.where(
            azimuthal_idx == 0,
            jnp.ones_like(phi),
            jnp.cos(azimuthal_idx * phi),
        )

        mode: Float[Array, " hh ww"] = (
            radial_profile * azimuthal_profile * core_mask
        )
        mode_energy: Float[Array, " "] = jnp.sum(jnp.abs(mode) ** 2)
        mode_normalized: Complex[Array, " hh ww"] = (
            mode / jnp.sqrt(mode_energy + 1e-20)
        ).astype(jnp.complex128)
        return mode_normalized

    modes: Complex[Array, " num_modes hh ww"] = jax.vmap(
        _compute_single_mode_int
    )(mode_indices)
    return modes


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

    dx_arr: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    a_arr: Float[Array, " "] = jnp.asarray(
        fiber_core_radius, dtype=jnp.float64
    )

    mode_indices: Float[Array, " num_modes 2"] = _generate_fiber_mode_indices(
        n_modes
    )

    modes = _multimode_fiber_output_impl(dx_arr, a_arr, hh, ww, mode_indices)

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
