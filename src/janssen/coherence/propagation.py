"""Propagation of partially coherent fields.

Extended Summary
----------------
This module provides functions for propagating partially coherent optical
fields represented as coherent mode sets or polychromatic wavefronts.

The key principle is that coherent modes propagate independently:
- Each mode is propagated using standard coherent propagation
- The final intensity is the incoherent sum of propagated mode intensities

For polychromatic fields, each wavelength component propagates with its
own phase velocity, leading to chromatic dispersion effects.

Routine Listings
----------------
propagate_coherent_modes : function
    Propagate all coherent modes through free space
propagate_polychromatic : function
    Propagate polychromatic wavefront with chromatic dispersion
apply_element_to_modes : function
    Apply an optical element to all coherent modes
intensity_from_modes : function
    Calculate total intensity from coherent mode sum
intensity_from_polychromatic : function
    Calculate total intensity from polychromatic wavefront

Notes
-----
Propagation is performed using vmap for efficient parallel computation
over all modes or wavelengths. This naturally leverages JAX's vectorization
capabilities.

References
----------
1. Mandel, L. & Wolf, E. "Optical Coherence and Quantum Optics" (1995)
2. Thibault, P. & Menzel, A. Nature 494, 68-71 (2013) - ptychography
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Optional
from jaxtyping import Array, Complex, Float, jaxtyped

from janssen.prop.free_space_prop import angular_spectrum_prop, fresnel_prop
from janssen.utils import (
    CoherentModeSet,
    OpticalWavefront,
    PolychromaticWavefront,
    ScalarFloat,
    ScalarNumeric,
    make_coherent_mode_set,
    make_optical_wavefront,
    make_polychromatic_wavefront,
)


@jaxtyped(typechecker=beartype)
def propagate_coherent_modes(
    mode_set: CoherentModeSet,
    distance: ScalarNumeric,
    method: str = "angular_spectrum",
    refractive_index: Optional[ScalarNumeric] = 1.0,
) -> CoherentModeSet:
    """Propagate all coherent modes through free space.

    Each mode is propagated independently, preserving the modal structure.
    Uses vmap for efficient parallel propagation of all modes.

    Parameters
    ----------
    mode_set : CoherentModeSet
        Input coherent mode set.
    distance : ScalarNumeric
        Propagation distance in meters.
    method : str
        Propagation method: "angular_spectrum" or "fresnel".
        Default is "angular_spectrum".
    refractive_index : Optional[ScalarNumeric]
        Refractive index of the medium. Default is 1.0 (vacuum).

    Returns
    -------
    propagated : CoherentModeSet
        Propagated coherent mode set with updated z_position.

    Notes
    -----
    The propagation preserves modal orthogonality in free space.
    The weights remain unchanged since each mode propagates
    independently without coupling.

    For propagation through aberrated systems or scattering media,
    modes may couple and a new eigendecomposition may be needed.
    """
    modes: Complex[Array, " num_modes hh ww"] = mode_set.modes
    if method == "angular_spectrum":
        prop_fn = angular_spectrum_prop
    elif method == "fresnel":
        prop_fn = fresnel_prop
    else:
        raise ValueError(f"Unknown propagation method: {method}")

    def _propagate_single_mode_int(
        mode: Complex[Array, " hh ww"],
    ) -> Complex[Array, " hh ww"]:
        """Propagate single mode through free space."""
        temporary_wavefront = make_optical_wavefront(
            field=mode,
            wavelength=mode_set.wavelength,
            dx=mode_set.dx,
            z_position=mode_set.z_position,
            polarization=False,
        )

        propagated_wavefront = prop_fn(
            temporary_wavefront,
            z_move=distance,
            refractive_index=refractive_index,
        )
        return propagated_wavefront.field

    propagated_modes: Complex[Array, " num_modes hh ww"] = jax.vmap(
        _propagate_single_mode_int
    )(modes)
    new_z: Float[Array, " "] = mode_set.z_position + jnp.asarray(
        distance, dtype=jnp.float64
    ) * jnp.asarray(refractive_index, dtype=jnp.float64)
    return make_coherent_mode_set(
        modes=propagated_modes,
        weights=mode_set.weights,
        wavelength=mode_set.wavelength,
        dx=mode_set.dx,
        z_position=new_z,
        polarization=mode_set.polarization,
        normalize_weights=False,
    )


@jaxtyped(typechecker=beartype)
def propagate_polychromatic(
    wavefront: PolychromaticWavefront,
    distance: ScalarNumeric,
    method: str = "angular_spectrum",
    refractive_index: Optional[ScalarNumeric] = 1.0,
) -> PolychromaticWavefront:
    """Propagate polychromatic wavefront accounting for chromatic dispersion.

    Each wavelength component propagates with its own phase velocity,
    leading to wavelength-dependent phase accumulation.

    Parameters
    ----------
    wavefront : PolychromaticWavefront
        Input polychromatic wavefront.
    distance : ScalarNumeric
        Propagation distance in meters.
    method : str
        Propagation method: "angular_spectrum" or "fresnel".
        Default is "angular_spectrum".
    refractive_index : Optional[ScalarNumeric]
        Refractive index of the medium (assumed constant across spectrum).
        Default is 1.0. For dispersive media, use material_prop.

    Returns
    -------
    propagated : PolychromaticWavefront
        Propagated polychromatic wavefront.

    Notes
    -----
    For accurate chromatic simulation in dispersive media, the refractive
    index should vary with wavelength. This function uses a constant
    refractive index for simplicity.

    The phase velocity varies as c/n, so different wavelengths accumulate
    different phases over the same distance. This leads to:
    - Chromatic focus shift in imaging
    - Pulse broadening in ultrafast optics
    - Coherence effects in interferometry
    """
    fields: Complex[Array, " num_wavelengths hh ww"] = wavefront.fields
    wavelengths: Float[Array, " num_wavelengths"] = wavefront.wavelengths
    if method == "angular_spectrum":
        prop_fn = angular_spectrum_prop
    elif method == "fresnel":
        prop_fn = fresnel_prop
    else:
        raise ValueError(f"Unknown propagation method: {method}")

    def _propagate_at_wavelength_int(
        field: Complex[Array, " hh ww"],
        wl: Float[Array, " "],
    ) -> Complex[Array, " hh ww"]:
        """Propagate field at specific wavelength."""
        wf = make_optical_wavefront(
            field=field,
            wavelength=wl,
            dx=wavefront.dx,
            z_position=wavefront.z_position,
            polarization=False,
        )
        propagated_wf = prop_fn(
            wf,
            z_move=distance,
            refractive_index=refractive_index,
        )
        return propagated_wf.field

    propagated_fields: Complex[Array, " num_wavelengths hh ww"] = jax.vmap(
        _propagate_at_wavelength_int
    )(fields, wavelengths)
    new_z: Float[Array, " "] = wavefront.z_position + jnp.asarray(
        distance, dtype=jnp.float64
    ) * jnp.asarray(refractive_index, dtype=jnp.float64)
    return make_polychromatic_wavefront(
        fields=propagated_fields,
        wavelengths=wavelengths,
        spectral_weights=wavefront.spectral_weights,
        dx=wavefront.dx,
        z_position=new_z,
        polarization=wavefront.polarization,
        normalize_weights=False,
    )


@jaxtyped(typechecker=beartype)
def apply_element_to_modes(
    mode_set: CoherentModeSet,
    element_fn: Callable[[OpticalWavefront], OpticalWavefront],
) -> CoherentModeSet:
    """Apply an optical element function to all coherent modes.

    Enables partial coherence propagation through arbitrary optical
    systems defined by element functions.

    Parameters
    ----------
    mode_set : CoherentModeSet
        Input coherent mode set.
    element_fn : Callable[[OpticalWavefront], OpticalWavefront]
        Function that takes an OpticalWavefront and returns a
        transformed OpticalWavefront. Examples: lens_propagation,
        aperture functions, phase masks.

    Returns
    -------
    transformed : CoherentModeSet
        Transformed coherent mode set.

    Notes
    -----
    The element function should be pure (no side effects) and
    JAX-compatible for use with vmap and grad.

    Common element functions include:
    - lens_propagation: Apply thin lens phase
    - circular_aperture: Apply aperture transmission
    - zernike_aberration: Add wavefront aberration
    """
    modes: Complex[Array, " num_modes hh ww"] = mode_set.modes

    def _apply_to_single_mode_int(
        mode: Complex[Array, " hh ww"],
    ) -> Complex[Array, " hh ww"]:
        """Apply optical element to single mode."""
        wavefront = make_optical_wavefront(
            field=mode,
            wavelength=mode_set.wavelength,
            dx=mode_set.dx,
            z_position=mode_set.z_position,
            polarization=False,
        )
        transformed_wf = element_fn(wavefront)
        return transformed_wf.field

    transformed_modes: Complex[Array, " num_modes hh ww"] = jax.vmap(
        _apply_to_single_mode_int
    )(modes)
    sample_wf = make_optical_wavefront(
        field=modes[0],
        wavelength=mode_set.wavelength,
        dx=mode_set.dx,
        z_position=mode_set.z_position,
        polarization=False,
    )
    transformed_sample = element_fn(sample_wf)
    return make_coherent_mode_set(
        modes=transformed_modes,
        weights=mode_set.weights,
        wavelength=mode_set.wavelength,
        dx=transformed_sample.dx,
        z_position=transformed_sample.z_position,
        polarization=mode_set.polarization,
        normalize_weights=False,
    )


@jaxtyped(typechecker=beartype)
def intensity_from_modes(
    mode_set: CoherentModeSet,
) -> Float[Array, " hh ww"]:
    """Calculate total intensity from coherent mode sum.

    Computes the incoherent sum of mode intensities:
        I(r) = sum_n weights[n] * |modes[n](r)|^2

    Parameters
    ----------
    mode_set : CoherentModeSet
        Input coherent mode set.

    Returns
    -------
    intensity : Float[Array, " hh ww"]
        Total intensity distribution.

    Notes
    -----
    This is the observable intensity for a partially coherent field.
    The phases of different modes do not interfere because they
    represent statistically independent contributions.
    """
    modes: Complex[Array, " num_modes hh ww"] = mode_set.modes
    weights: Float[Array, " num_modes"] = mode_set.weights

    if mode_set.polarization:
        mode_intensities: Float[Array, " num_modes hh ww"] = jnp.sum(
            jnp.abs(modes) ** 2, axis=-1
        )
    else:
        mode_intensities = jnp.abs(modes) ** 2

    intensity: Float[Array, " hh ww"] = jnp.sum(
        weights[:, jnp.newaxis, jnp.newaxis] * mode_intensities, axis=0
    )
    return intensity


@jaxtyped(typechecker=beartype)
def intensity_from_polychromatic(
    wavefront: PolychromaticWavefront,
) -> Float[Array, " hh ww"]:
    """Calculate total intensity from polychromatic wavefront.

    Computes the spectrally-weighted incoherent sum:
        I(r) = sum_lambda S(lambda) * |E(r, lambda)|^2

    Parameters
    ----------
    wavefront : PolychromaticWavefront
        Input polychromatic wavefront.

    Returns
    -------
    intensity : Float[Array, " hh ww"]
        Total intensity distribution.

    Notes
    -----
    Different wavelength components do not interfere because they
    oscillate at different frequencies. The intensity is always
    the incoherent sum over wavelengths.
    """
    fields: Complex[Array, " num_wavelengths hh ww"] = wavefront.fields
    spectral_weights: Float[Array, " num_wavelengths"] = (
        wavefront.spectral_weights
    )

    if wavefront.polarization:
        field_intensities: Float[Array, " num_wavelengths hh ww"] = jnp.sum(
            jnp.abs(fields) ** 2, axis=-1
        )
    else:
        field_intensities = jnp.abs(fields) ** 2

    intensity: Float[Array, " hh ww"] = jnp.sum(
        spectral_weights[:, jnp.newaxis, jnp.newaxis] * field_intensities,
        axis=0,
    )
    return intensity


@jaxtyped(typechecker=beartype)
def propagate_and_focus_modes(
    mode_set: CoherentModeSet,
    focal_length: ScalarFloat,
    propagation_distance: ScalarFloat,
    method: str = "angular_spectrum",
) -> CoherentModeSet:
    """Propagate modes through a thin lens and to focus.

    Convenience function that applies a thin lens phase and propagates
    to the focal plane.

    Parameters
    ----------
    mode_set : CoherentModeSet
        Input coherent mode set.
    focal_length : ScalarFloat
        Focal length of the lens in meters.
    propagation_distance : ScalarFloat
        Distance from lens to observation plane in meters.
        For perfect focus, set equal to focal_length.
    method : str
        Propagation method. Default is "angular_spectrum".

    Returns
    -------
    focused : CoherentModeSet
        Focused coherent mode set.

    Notes
    -----
    The thin lens phase is:
        phi(r) = -k * r^2 / (2 * f)

    where k = 2*pi/lambda and f is the focal length.
    """
    modes: Complex[Array, " num_modes hh ww"] = mode_set.modes
    hh: int = modes.shape[1]
    ww: int = modes.shape[2]
    y: Float[Array, " hh"] = (jnp.arange(hh) - hh / 2) * mode_set.dx
    x: Float[Array, " ww"] = (jnp.arange(ww) - ww / 2) * mode_set.dx
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)
    r2: Float[Array, " hh ww"] = xx**2 + yy**2
    wavenumber: Float[Array, " "] = 2.0 * jnp.pi / mode_set.wavelength
    focal_length_arr: Float[Array, " "] = jnp.asarray(
        focal_length, dtype=jnp.float64
    )
    thin_lens_phase: Float[Array, " hh ww"] = (
        -wavenumber * r2 / (2.0 * focal_length_arr)
    )
    lens_transmission: Complex[Array, " hh ww"] = jnp.exp(1j * thin_lens_phase)
    modes_after_lens: Complex[Array, " num_modes hh ww"] = (
        modes * lens_transmission[jnp.newaxis, :, :]
    )
    mode_set_after_lens = make_coherent_mode_set(
        modes=modes_after_lens,
        weights=mode_set.weights,
        wavelength=mode_set.wavelength,
        dx=mode_set.dx,
        z_position=mode_set.z_position,
        polarization=mode_set.polarization,
        normalize_weights=False,
    )
    return propagate_coherent_modes(
        mode_set_after_lens,
        distance=propagation_distance,
        method=method,
    )
