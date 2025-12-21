"""Differentiable optical simulation toolkit.

Extended Summary
----------------
Comprehensive optical simulation framework for modeling light
propagation through various optical elements. All components are
differentiable and optimized for JAX transformations, enabling gradient-based
optimization of optical systems.

Submodules
----------
apertures
    Aperture functions for optical microscopy
bessel
    Bessel functions for JAX
elements
    Optical element transformations
helper
    Helper functions for optical propagation
zernike
    Zernike polynomial functions for optical aberration modeling

Routine Listings
----------------
add_phase_screen : function
    Add phase screen to field
amplitude_grating_binary : function
    Create binary amplitude grating
annular_aperture : function
    Create an annular (ring-shaped) aperture
apply_aberration : function
    Apply aberration to optical wavefront
apply_phase_mask : function
    Apply a phase mask to a field
apply_phase_mask_fn : function
    Apply a phase mask function
astigmatism : function
    Generate astigmatism aberration (Z5, Z6)
beam_splitter : function
    Model beam splitter operation
bessel_iv_series : function
    Compute I_v(x) using series expansion for Bessel function.
bessel_j0 : function
    Compute J_0(x), regular Bessel function of the first kind, order 0.
bessel_jn : function
    Compute J_n(x), regular Bessel function of the first kind, order n.
bessel_k0_series : function
    Compute K_0(x) using series expansion.
bessel_kn_recurrence : function
    Compute K_n(x) using recurrence relation.
bessel_kv : function
    Compute K_v(x), modified Bessel function of the second kind.
bessel_kv_small_integer : function
    Compute K_v(x) for small x and integer v.
bessel_kv_small_non_integer : function
    Compute K_v(x) for small x and non-integer v.
circular_aperture : function
    Create a circular aperture
coma : function
    Generate coma aberration (Z7, Z8)
create_spatial_grid : function
    Create computational spatial grid
defocus : function
    Generate defocus aberration (Z4)
factorial : function
    JAX-compatible factorial computation
field_intensity : function
    Calculate field intensity
gaussian_apodizer : function
    Apply Gaussian apodization to a field
gaussian_apodizer_elliptical : function
    Apply elliptical Gaussian apodization
generate_aberration_nm : function
    Generate aberration phase map from Zernike coefficients (n, m)
generate_aberration_noll : function
    Generate aberration phase map from Zernike coefficients (Noll)
compute_phase_from_coeffs : function
    Compute phase map from Zernike coefficients with configurable starting index
phase_rms : function
    Compute RMS of phase within the unit pupil
half_waveplate : function
    Half-wave plate transformation
mirror_reflection : function
    Model mirror reflection
nd_filter : function
    Neutral density filter
nm_to_noll : function
    Convert (n, m) indices to Noll index
noll_to_nm : function
    Convert Noll index to (n, m) indices
normalize_field : function
    Normalize optical field
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
rectangular_aperture : function
    Create a rectangular aperture
scale_pixel : function
    Scale pixel size in field
spherical_aberration : function
    Generate spherical aberration (Z11)
supergaussian_apodizer : function
    Apply super-Gaussian apodization
supergaussian_apodizer_elliptical : function
    Apply elliptical super-Gaussian apodization
trefoil : function
    Generate trefoil aberration (Z9, Z10)
variable_transmission_aperture : function
    Create aperture with variable transmission
waveplate_jones : function
    General waveplate Jones matrix
zernike_polynomial : function
    Generate a single Zernike polynomial
zernike_radial : function
    Radial component of Zernike polynomial

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
from .bessel import (
    bessel_iv_series,
    bessel_j0,
    bessel_jn,
    bessel_k0_series,
    bessel_kn_recurrence,
    bessel_kv,
    bessel_kv_small_integer,
    bessel_kv_small_non_integer,
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
    sellmeier,
)
from .zernike import (
    apply_aberration,
    astigmatism,
    coma,
    compute_phase_from_coeffs,
    defocus,
    factorial,
    generate_aberration_nm,
    generate_aberration_noll,
    nm_to_noll,
    noll_to_nm,
    phase_rms,
    spherical_aberration,
    trefoil,
    zernike_polynomial,
    zernike_radial,
)

__all__: list[str] = [
    "add_phase_screen",
    "amplitude_grating_binary",
    "annular_aperture",
    "apply_aberration",
    "apply_phase_mask",
    "apply_phase_mask_fn",
    "astigmatism",
    "beam_splitter",
    "bessel_iv_series",
    "bessel_j0",
    "bessel_jn",
    "bessel_k0_series",
    "bessel_kn_recurrence",
    "bessel_kv",
    "bessel_kv_small_integer",
    "bessel_kv_small_non_integer",
    "circular_aperture",
    "coma",
    "compute_phase_from_coeffs",
    "create_spatial_grid",
    "defocus",
    "factorial",
    "field_intensity",
    "gaussian_apodizer",
    "gaussian_apodizer_elliptical",
    "generate_aberration_nm",
    "generate_aberration_noll",
    "half_waveplate",
    "mirror_reflection",
    "nd_filter",
    "nm_to_noll",
    "noll_to_nm",
    "normalize_field",
    "phase_grating_blazed_elliptical",
    "phase_grating_sawtooth",
    "phase_grating_sine",
    "phase_rms",
    "polarizer_jones",
    "prism_phase_ramp",
    "quarter_waveplate",
    "rectangular_aperture",
    "scale_pixel",
    "sellmeier",
    "spherical_aberration",
    "supergaussian_apodizer",
    "supergaussian_apodizer_elliptical",
    "trefoil",
    "variable_transmission_aperture",
    "waveplate_jones",
    "zernike_polynomial",
    "zernike_radial",
]
