"""
Module: janssen.simul.
----------------------
Differentiable optical simulation toolkit.

This package implements various optical components and propagation models
with JAX for automatic differentiation and acceleration. All functions
are fully differentiable and JIT-compilable.

Submodules
----------
apertures
    Aperture functions for creating and manipulating optical wavefronts.
elements
    Common optical elements beyond lenses and basic apertures.
helper
    Utility functions for creating grids, phase manipulation, and field 
    calculations.
lens_optics
    Optical propagation functions including angular spectrum, Fresnel, 
    and Fraunhofer methods.
lenses`:
    Models for various lens types and their optical properties
microscope
    Forward propagation of light through optical elements.
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
