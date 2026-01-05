"""Differentiable optical simulation toolkit.

Extended Summary
----------------
Comprehensive optical simulation framework for modeling light
propagation through various optical elements. All components are
differentiable and optimized for JAX transformations, enabling gradient-based
optimization of optical systems.

Routine Listings
----------------
:func:`circular_aperture`
    Create a circular aperture.
:func:`rectangular_aperture`
    Create a rectangular aperture.
:func:`annular_aperture`
    Create an annular (ring-shaped) aperture.
:func:`gaussian_apodizer`
    Apply Gaussian apodization to a field.
:func:`gaussian_apodizer_elliptical`
    Apply elliptical Gaussian apodization.
:func:`supergaussian_apodizer`
    Apply super-Gaussian apodization.
:func:`supergaussian_apodizer_elliptical`
    Apply elliptical super-Gaussian apodization.
:func:`variable_transmission_aperture`
    Create aperture with variable transmission.
:func:`apply_phase_mask`
    Apply a phase mask to a field.
:func:`apply_phase_mask_fn`
    Apply a phase mask function.
:func:`amplitude_grating_binary`
    Create binary amplitude grating.
:func:`phase_grating_sine`
    Sinusoidal phase grating.
:func:`phase_grating_sawtooth`
    Sawtooth phase grating.
:func:`phase_grating_blazed_elliptical`
    Elliptical blazed phase grating.
:func:`beam_splitter`
    Model beam splitter operation.
:func:`mirror_reflection`
    Model mirror reflection.
:func:`nd_filter`
    Neutral density filter.
:func:`prism_phase_ramp`
    Phase ramp from prism.
:func:`polarizer_jones`
    Jones matrix for polarizer.
:func:`half_waveplate`
    Half-wave plate transformation.
:func:`quarter_waveplate`
    Quarter-wave plate transformation.
:func:`waveplate_jones`
    General waveplate Jones matrix.
:func:`add_phase_screen`
    Add phase screen to field.
:func:`create_spatial_grid`
    Create computational spatial grid.
:func:`field_intensity`
    Calculate field intensity.
:func:`normalize_field`
    Normalize optical field.
:func:`scale_pixel`
    Scale pixel size in field.
:func:`sellmeier`
    Sellmeier equation for refractive index.
:func:`zernike_polynomial`
    Generate a single Zernike polynomial.
:func:`zernike_radial`
    Radial component of Zernike polynomial.
:func:`noll_to_nm`
    Convert Noll index to (n, m) indices.
:func:`nm_to_noll`
    Convert (n, m) indices to Noll index.
:func:`factorial`
    JAX-compatible factorial computation.
:func:`defocus`
    Generate defocus aberration (Z4).
:func:`astigmatism`
    Generate astigmatism aberration (Z5, Z6).
:func:`coma`
    Generate coma aberration (Z7, Z8).
:func:`trefoil`
    Generate trefoil aberration (Z9, Z10).
:func:`spherical_aberration`
    Generate spherical aberration (Z11).
:func:`apply_aberration`
    Apply aberration to optical wavefront.
:func:`generate_aberration_nm`
    Generate aberration phase map from (n, m) coefficients.
:func:`generate_aberration_noll`
    Generate aberration phase map from Noll coefficients.
:func:`compute_phase_from_coeffs`
    Compute phase map from Zernike coefficients.
:func:`phase_rms`
    Compute RMS of phase within the unit pupil.

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
