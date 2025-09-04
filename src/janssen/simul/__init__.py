"""Differentiable optical simulation toolkit.

Extended Summary
----------------
Comprehensive optical simulation framework for modeling light propagation
through various optical elements. All components are differentiable and
optimized for JAX transformations, enabling gradient-based optimization
of optical systems.

Submodules
----------
apertures
    Aperture functions for optical microscopy
elements
    Optical element transformations
microscope
    Microscopy simulation pipelines
helper
    Helper functions for optical propagation

Routine Listings
----------------
annular_aperture
    Create an annular (ring-shaped) aperture
circular_aperture
    Create a circular aperture
gaussian_apodizer
    Apply Gaussian apodization to a field
gaussian_apodizer_elliptical
    Apply elliptical Gaussian apodization
rectangular_aperture
    Create a rectangular aperture
supergaussian_apodizer
    Apply super-Gaussian apodization
supergaussian_apodizer_elliptical
    Apply elliptical super-Gaussian apodization
variable_transmission_aperture
    Create aperture with variable transmission
amplitude_grating_binary
    Create binary amplitude grating
apply_phase_mask
    Apply a phase mask to a field
apply_phase_mask_fn
    Apply a phase mask function
beam_splitter
    Model beam splitter operation
half_waveplate
    Half-wave plate transformation
mirror_reflection
    Model mirror reflection
nd_filter
    Neutral density filter
phase_grating_blazed_elliptical
    Elliptical blazed phase grating
phase_grating_sawtooth
    Sawtooth phase grating
phase_grating_sine
    Sinusoidal phase grating
polarizer_jones
    Jones matrix for polarizer
prism_phase_ramp
    Phase ramp from prism
quarter_waveplate
    Quarter-wave plate transformation
waveplate_jones
    General waveplate Jones matrix
add_phase_screen
    Add phase screen to field
create_spatial_grid
    Create computational spatial grid
field_intensity
    Calculate field intensity
normalize_field
    Normalize optical field
scale_pixel
    Scale pixel size in field
linear_interaction
    Linear light-matter interaction
simple_diffractogram
    Generate diffraction pattern
simple_microscope
    Simple microscope forward model

Notes
-----
All simulation functions support automatic differentiation and can be
composed to model complex optical systems. The toolkit is optimized for
both forward simulation and inverse problems in optics.
"""

from .apertures import (
    annular_aperture,
    circular_aperture,
    gaussian_apodizer,
    gaussian_apodizer_elliptical,
    rectangular_aperture,
    supergaussian_apodizer,
    supergaussian_apodizer_elliptical,
    variable_transmission_aperture,
)
from .elements import (
    amplitude_grating_binary,
    apply_phase_mask,
    apply_phase_mask_fn,
    beam_splitter,
    half_waveplate,
    mirror_reflection,
    nd_filter,
    phase_grating_blazed_elliptical,
    phase_grating_sawtooth,
    phase_grating_sine,
    polarizer_jones,
    prism_phase_ramp,
    quarter_waveplate,
    waveplate_jones,
)
from .helper import (
    add_phase_screen,
    create_spatial_grid,
    field_intensity,
    normalize_field,
    scale_pixel,
)
from .microscope import (
    linear_interaction,
    simple_diffractogram,
    simple_microscope,
)

__all__: list[str] = [
    "annular_aperture",
    "circular_aperture",
    "gaussian_apodizer",
    "gaussian_apodizer_elliptical",
    "rectangular_aperture",
    "supergaussian_apodizer",
    "supergaussian_apodizer_elliptical",
    "variable_transmission_aperture",
    "amplitude_grating_binary",
    "apply_phase_mask",
    "apply_phase_mask_fn",
    "beam_splitter",
    "half_waveplate",
    "mirror_reflection",
    "nd_filter",
    "phase_grating_blazed_elliptical",
    "phase_grating_sawtooth",
    "phase_grating_sine",
    "polarizer_jones",
    "prism_phase_ramp",
    "quarter_waveplate",
    "waveplate_jones",
    "add_phase_screen",
    "create_spatial_grid",
    "field_intensity",
    "normalize_field",
    "scale_pixel",
    "linear_interaction",
    "simple_diffractogram",
    "simple_microscope",
    "lens_propagation",
    "simple_diffractogram",
    "simple_microscope",
]
