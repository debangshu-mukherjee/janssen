"""Differentiable optical simulation toolkit.

Extended Summary
----------------
Comprehensive optical simulation framework for modeling light
propagation
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
zernike
    Zernike polynomial functions for optical aberration modeling

Routine Listings
----------------
annular_aperture : function
    Create an annular (ring-shaped) aperture
circular_aperture : function
    Create a circular aperture
gaussian_apodizer : function
    Apply Gaussian apodization to a field
gaussian_apodizer_elliptical : function
    Apply elliptical Gaussian apodization
rectangular_aperture : function
    Create a rectangular aperture
supergaussian_apodizer : function
    Apply super-Gaussian apodization
supergaussian_apodizer_elliptical : function
    Apply elliptical super-Gaussian apodization
variable_transmission_aperture : function
    Create aperture with variable transmission
amplitude_grating_binary : function
    Create binary amplitude grating
apply_phase_mask : function
    Apply a phase mask to a field
apply_phase_mask_fn : function
    Apply a phase mask function
beam_splitter : function
    Model beam splitter operation
half_waveplate : function
    Half-wave plate transformation
mirror_reflection : function
    Model mirror reflection
nd_filter : function
    Neutral density filter
phase_grating_blazed_elliptical : function
    Elliptical blazed phase grating
phase_grating_sawtooth : function
    Sawtooth phase grating
phase_grating_sine : function
    Sinusoidal phase grating
polarizer_jones : function
    Jones matrix for polarizer
prism_phase_ramp : function
    Phase ramp from prism
quarter_waveplate : function
    Quarter-wave plate transformation
waveplate_jones : function
    General waveplate Jones matrix
add_phase_screen : function
    Add phase screen to field
create_spatial_grid : function
    Create computational spatial grid
field_intensity : function
    Calculate field intensity
normalize_field : function
    Normalize optical field
scale_pixel : function
    Scale pixel size in field
linear_interaction : function
    Linear light-matter interaction
simple_diffractogram : function
    Generate diffraction pattern
simple_microscope : function
    Simple microscope forward model
zernike_polynomial : function
    Generate a single Zernike polynomial
zernike_radial : function
    Radial component of Zernike polynomial
factorial : function
    JAX-compatible factorial computation
noll_to_nm : function
    Convert Noll index to (n, m) indices
nm_to_noll : function
    Convert (n, m) indices to Noll index
generate_aberration : function
    Generate aberration phase map from Zernike coefficients
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
from .zernike import (
    apply_aberration,
    astigmatism,
    coma,
    defocus,
    factorial,
    generate_aberration,
    nm_to_noll,
    noll_to_nm,
    spherical_aberration,
    trefoil,
    zernike_polynomial,
    zernike_radial,
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
    "zernike_polynomial",
    "zernike_radial",
    "factorial",
    "noll_to_nm",
    "nm_to_noll",
    "generate_aberration",
    "defocus",
    "astigmatism",
    "coma",
    "spherical_aberration",
    "trefoil",
    "apply_aberration",
]
