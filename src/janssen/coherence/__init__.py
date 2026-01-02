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

Submodules
----------
spatial
    Spatial coherence functions (kernels, van Cittert-Zernike)
temporal
    Temporal/spectral coherence functions (spectra, coherence length)
modes
    Coherent mode generation and decomposition
propagation
    Propagation of partially coherent fields
sources
    Partially coherent source models (LED, thermal, synchrotron)

Key Data Structures
-------------------
CoherentModeSet : PyTree
    Coherent mode representation: modes, weights, wavelength, dx
PolychromaticWavefront : PyTree
    Polychromatic field representation for temporal coherence
MutualIntensity : PyTree
    Full J(r1, r2) representation (O(N^4) memory - use sparingly)

These types are defined in janssen.utils.coherence_types.

Routine Listings
----------------
Spatial Coherence:
    gaussian_coherence_kernel : Gaussian spatial coherence kernel
    jinc_coherence_kernel : Jinc kernel from circular incoherent source
    rectangular_coherence_kernel : Sinc kernel from rectangular source
    coherence_width_from_source : Coherence width (van Cittert-Zernike)
    complex_degree_of_coherence : Normalized coherence from mutual intensity

Temporal Coherence:
    gaussian_spectrum : Gaussian spectral distribution
    lorentzian_spectrum : Lorentzian (natural) lineshape
    rectangular_spectrum : Flat-top spectral distribution
    blackbody_spectrum : Planck spectral distribution
    coherence_length : Calculate L_c = lambda^2 / Delta_lambda
    coherence_time : Calculate tau_c = lambda^2 / (c * Delta_lambda)

Mode Generation:
    hermite_gaussian_modes : Generate Hermite-Gaussian mode set
    gaussian_schell_model_modes : Modes for Gaussian Schell-model source
    eigenmode_decomposition : Decompose mutual intensity into modes
    effective_mode_count : Calculate participation ratio
    modes_from_wavefront : Wrap coherent field as single-mode set

Propagation:
    propagate_coherent_modes : Propagate all modes through free space
    propagate_polychromatic : Propagate polychromatic wavefront
    apply_element_to_modes : Apply optical element to all modes
    intensity_from_modes : Total intensity from mode sum
    intensity_from_polychromatic : Total intensity from spectral sum
    propagate_and_focus_modes : Convenience: lens + propagation

Source Models:
    led_source : LED with spatial and temporal partial coherence
    thermal_source : Thermal (blackbody) source
    synchrotron_source : Synchrotron X-ray source (anisotropic)
    laser_with_mode_noise : Laser with imperfect mode purity
    multimode_fiber_output : Multimode fiber output

Notes
-----
The memory-efficient coherent mode representation stores O(M × N²) data
for M modes on an N×N grid, compared to O(N⁴) for the full mutual
intensity. For typical simulations, 5-20 modes capture most of the
field energy.

The total intensity from coherent modes is:
    I(r) = Σₙ wₙ |φₙ(r)|²

where wₙ are the modal weights and φₙ are the modes.

Example Usage
-------------
>>> from janssen.coherence import led_source, propagate_coherent_modes
>>> from janssen.coherence import intensity_from_modes
>>>
>>> # Create LED source
>>> modes, wavelengths, weights = led_source(
...     center_wavelength=530e-9,
...     bandwidth_fwhm=30e-9,
...     spatial_coherence_width=50e-6,
...     dx=1e-6,
...     grid_size=(256, 256),
... )
>>>
>>> # Propagate through optical system
>>> propagated = propagate_coherent_modes(modes, distance=0.1)
>>>
>>> # Compute observable intensity
>>> I = intensity_from_modes(propagated)

References
----------
1. Mandel, L. & Wolf, E. "Optical Coherence and Quantum Optics" (1995)
2. Goodman, J. W. "Statistical Optics" (2015)
3. Born, M. & Wolf, E. "Principles of Optics" Chapter 10
4. Starikov, A. & Wolf, E. "Coherent-mode representation of Gaussian
   Schell-model sources" JOSA A (1982)
5. Thibault, P. & Menzel, A. "Reconstructing state mixtures from
   diffraction measurements" Nature (2013)
"""

# Spatial coherence functions
from .spatial import (
    coherence_width_from_source,
    complex_degree_of_coherence,
    gaussian_coherence_kernel,
    jinc_coherence_kernel,
    rectangular_coherence_kernel,
)

# Temporal coherence functions
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

# Mode generation and decomposition
from .modes import (
    effective_mode_count,
    eigenmode_decomposition,
    gaussian_schell_model_modes,
    hermite_gaussian_modes,
    modes_from_wavefront,
    mutual_intensity_from_modes,
)

# Propagation functions
from .propagation import (
    apply_element_to_modes,
    intensity_from_modes,
    intensity_from_polychromatic,
    propagate_and_focus_modes,
    propagate_coherent_modes,
    propagate_polychromatic,
)

# Source models
from .sources import (
    laser_with_mode_noise,
    led_source,
    multimode_fiber_output,
    synchrotron_source,
    thermal_source,
)

__all__: list[str] = [
    # Spatial coherence
    "gaussian_coherence_kernel",
    "jinc_coherence_kernel",
    "rectangular_coherence_kernel",
    "coherence_width_from_source",
    "complex_degree_of_coherence",
    # Temporal coherence
    "gaussian_spectrum",
    "lorentzian_spectrum",
    "rectangular_spectrum",
    "blackbody_spectrum",
    "coherence_length",
    "coherence_time",
    "bandwidth_from_coherence_length",
    "spectral_phase_from_dispersion",
    # Mode generation
    "hermite_gaussian_modes",
    "gaussian_schell_model_modes",
    "eigenmode_decomposition",
    "effective_mode_count",
    "modes_from_wavefront",
    "mutual_intensity_from_modes",
    # Propagation
    "propagate_coherent_modes",
    "propagate_polychromatic",
    "apply_element_to_modes",
    "intensity_from_modes",
    "intensity_from_polychromatic",
    "propagate_and_focus_modes",
    # Sources
    "led_source",
    "thermal_source",
    "synchrotron_source",
    "laser_with_mode_noise",
    "multimode_fiber_output",
]
