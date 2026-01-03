"""Optical beam and material models for testing and validation.

Extended Summary
----------------
Models for generating datasets for testing and validation.

Routine Listings
----------------
:func:`plane_wave`
    Creates a uniform plane wave with optional tilt.
:func:`gaussian_beam`
    Creates a Gaussian beam from complex beam parameter q.
:func:`collimated_gaussian`
    Creates a collimated Gaussian beam with flat phase.
:func:`converging_gaussian`
    Creates a Gaussian beam converging to a focus.
:func:`diverging_gaussian`
    Creates a Gaussian beam diverging from a virtual source.
:func:`bessel_beam`
    Creates a Bessel beam with specified cone angle.
:func:`hermite_gaussian`
    Creates Hermite-Gaussian modes.
:func:`laguerre_gaussian`
    Creates Laguerre-Gaussian modes.
:func:`sinusoidal_wave`
    Creates a sinusoidal interference pattern.
:func:`propagate_beam`
    Generates a beam at multiple z positions as a PropagatingWavefront.
:func:`radially_polarized_beam`
    Generate a radially polarized beam.
:func:`azimuthally_polarized_beam`
    Generate an azimuthally polarized beam.
:func:`linear_polarized_beam`
    Generate a linearly polarized beam with arbitrary angle.
:func:`x_polarized_beam`
    Generate an x-polarized beam.
:func:`y_polarized_beam`
    Generate a y-polarized beam.
:func:`circular_polarized_beam`
    Generate a circularly polarized beam.
:func:`generalized_cylindrical_vector_beam`
    Generate a generalized cylindrical vector beam.
:func:`uniform_material`
    Creates a uniform 3D material with constant refractive index.
:func:`layered_material`
    Creates alternating layers of materials.
:func:`gradient_index_material`
    Creates a gradient-index (GRIN) material with radial profile.
:func:`spherical_inclusion`
    Creates a material with spherical inclusion.
:func:`biological_cell`
    Creates a biological cell model with nucleus.
:func:`generate_usaf_pattern`
    Generates USAF 1951 resolution test pattern.
:func:`calculate_usaf_group_range`
    Calculates the viable USAF group range for given parameters.
:func:`get_bar_width_pixels`
    Calculate bar width in pixels for a given group and element.
:func:`create_bar_triplet`
    Creates 3 parallel bars (horizontal or vertical).
:func:`create_element_pattern`
    Creates a single element pattern (horizontal + vertical bars).
:func:`create_group_pattern`
    Creates a group pattern with multiple elements.

Notes
-----
All functions are JAX-compatible and support automatic differentiation.
"""

from .beams import (
    bessel_beam,
    collimated_gaussian,
    converging_gaussian,
    diverging_gaussian,
    gaussian_beam,
    hermite_gaussian,
    laguerre_gaussian,
    plane_wave,
    propagate_beam,
    sinusoidal_wave,
)
from .material_models import (
    biological_cell,
    gradient_index_material,
    layered_material,
    spherical_inclusion,
    uniform_material,
)
from .polar_beams import (
    azimuthally_polarized_beam,
    circular_polarized_beam,
    generalized_cylindrical_vector_beam,
    linear_polarized_beam,
    radially_polarized_beam,
    x_polarized_beam,
    y_polarized_beam,
)
from .usaf_pattern import (
    calculate_usaf_group_range,
    create_bar_triplet,
    create_element_pattern,
    create_group_pattern,
    generate_usaf_pattern,
    get_bar_width_pixels,
)

__all__: list[str] = [
    "azimuthally_polarized_beam",
    "bessel_beam",
    "biological_cell",
    "calculate_usaf_group_range",
    "circular_polarized_beam",
    "collimated_gaussian",
    "converging_gaussian",
    "create_bar_triplet",
    "create_element_pattern",
    "create_group_pattern",
    "diverging_gaussian",
    "gaussian_beam",
    "generalized_cylindrical_vector_beam",
    "generate_usaf_pattern",
    "get_bar_width_pixels",
    "gradient_index_material",
    "hermite_gaussian",
    "layered_material",
    "laguerre_gaussian",
    "linear_polarized_beam",
    "plane_wave",
    "propagate_beam",
    "radially_polarized_beam",
    "sinusoidal_wave",
    "spherical_inclusion",
    "uniform_material",
    "x_polarized_beam",
    "y_polarized_beam",
]
