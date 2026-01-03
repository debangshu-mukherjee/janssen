"""Partial coherence module for optical simulations.

Extended Summary
----------------
This module provides comprehensive support for partially coherent optical
fields, including both spatial coherence (extended sources, van Cittert-Zernike
theorem) and temporal coherence (finite bandwidth, chromatic effects).

The core approach is **coherent mode decomposition**: any partially coherent
field can be represented as a weighted sum of orthogonal coherent modes
(Mercer's theorem). This enables efficient simulation by propagating a
finite number of modes and summing intensities.

Routine Listings
----------------
:func:`gaussian_coherence_kernel`
    Gaussian spatial coherence kernel.
:func:`jinc_coherence_kernel`
    Jinc kernel from circular incoherent source.
:func:`rectangular_coherence_kernel`
    Sinc kernel from rectangular source.
:func:`coherence_width_from_source`
    Coherence width via van Cittert-Zernike theorem.
:func:`complex_degree_of_coherence`
    Normalized coherence from mutual intensity.
:func:`gaussian_spectrum`
    Gaussian spectral distribution.
:func:`lorentzian_spectrum`
    Lorentzian (natural) lineshape.
:func:`rectangular_spectrum`
    Flat-top spectral distribution.
:func:`blackbody_spectrum`
    Planck spectral distribution.
:func:`coherence_length`
    Calculate coherence length from bandwidth.
:func:`coherence_time`
    Calculate coherence time from bandwidth.
:func:`bandwidth_from_coherence_length`
    Invert coherence length to bandwidth.
:func:`spectral_phase_from_dispersion`
    Spectral phase from material dispersion.
:func:`hermite_gaussian_modes`
    Generate Hermite-Gaussian mode set.
:func:`gaussian_schell_model_modes`
    Modes for Gaussian Schell-model source.
:func:`eigenmode_decomposition`
    Decompose mutual intensity into modes.
:func:`effective_mode_count`
    Calculate participation ratio of modes.
:func:`modes_from_wavefront`
    Wrap coherent field as single-mode set.
:func:`mutual_intensity_from_modes`
    Reconstruct mutual intensity from modes.
:func:`propagate_coherent_modes`
    Propagate all modes through free space.
:func:`propagate_polychromatic`
    Propagate polychromatic wavefront.
:func:`apply_element_to_modes`
    Apply optical element to all modes.
:func:`intensity_from_modes`
    Total intensity from mode sum.
:func:`intensity_from_polychromatic`
    Total intensity from spectral sum.
:func:`propagate_and_focus_modes`
    Lens focusing and propagation of modes.
:func:`led_source`
    LED with spatial and temporal partial coherence.
:func:`thermal_source`
    Thermal (blackbody) source.
:func:`synchrotron_source`
    Synchrotron X-ray source (anisotropic).
:func:`laser_with_mode_noise`
    Laser with imperfect mode purity.
:func:`multimode_fiber_output`
    Multimode fiber output.

Notes
-----
The memory-efficient coherent mode representation stores O(M × N²) data
for M modes on an N×N grid, compared to O(N⁴) for the full mutual
intensity. For typical simulations, 5-20 modes capture most of the
field energy.

The total intensity from coherent modes is:
    I(r) = Σₙ wₙ |φₙ(r)|²

where wₙ are the modal weights and φₙ are the modes.
"""

from .modes import (
    effective_mode_count,
    eigenmode_decomposition,
    gaussian_schell_model_modes,
    hermite_gaussian_modes,
    modes_from_wavefront,
    mutual_intensity_from_modes,
)

from .propagation import (
    apply_element_to_modes,
    intensity_from_modes,
    intensity_from_polychromatic,
    propagate_and_focus_modes,
    propagate_coherent_modes,
    propagate_polychromatic,
)

from .sources import (
    laser_with_mode_noise,
    led_source,
    multimode_fiber_output,
    synchrotron_source,
    thermal_source,
)
from .spatial import (
    coherence_width_from_source,
    complex_degree_of_coherence,
    gaussian_coherence_kernel,
    jinc_coherence_kernel,
    rectangular_coherence_kernel,
)

from .temporal import (
    bandwidth_from_coherence_length,
    blackbody_spectrum,
    coherence_length,
    coherence_time,
    gaussian_spectrum,
    lorentzian_spectrum,
    rectangular_spectrum,
    spectral_phase_from_dispersion,
)

__all__: list[str] = [
    "gaussian_coherence_kernel",
    "jinc_coherence_kernel",
    "rectangular_coherence_kernel",
    "coherence_width_from_source",
    "complex_degree_of_coherence",
    "gaussian_spectrum",
    "lorentzian_spectrum",
    "rectangular_spectrum",
    "blackbody_spectrum",
    "coherence_length",
    "coherence_time",
    "bandwidth_from_coherence_length",
    "spectral_phase_from_dispersion",
    "hermite_gaussian_modes",
    "gaussian_schell_model_modes",
    "eigenmode_decomposition",
    "effective_mode_count",
    "modes_from_wavefront",
    "mutual_intensity_from_modes",
    "propagate_coherent_modes",
    "propagate_polychromatic",
    "apply_element_to_modes",
    "intensity_from_modes",
    "intensity_from_polychromatic",
    "propagate_and_focus_modes",
    "led_source",
    "thermal_source",
    "synchrotron_source",
    "laser_with_mode_noise",
    "multimode_fiber_output",
]
