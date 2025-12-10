"""Lens implementations and optical calculations.

Extended Summary
----------------
Models for generating datasets for testing and validation.

Submodules
----------
beams
    Beam generation functions
material_models
    Material models for optical simulations
polar_beams
    Polarized beam generators for vector optics
usaf_pattern
    USAF test pattern generation

Routine Listings
----------------
bessel_beam : function
    Creates a Bessel beam with specified cone angle
biological_cell : function
    Creates a biological cell model with nucleus
calculate_usaf_group_range : function
    Calculates the viable USAF group range for given parameters
collimated_gaussian : function
    Creates a collimated Gaussian beam with flat phase
converging_gaussian : function
    Creates a Gaussian beam converging to a focus
create_bar_triplet : function
    Creates 3 parallel bars (horizontal or vertical)
create_element_pattern : function
    Creates a single element pattern (horizontal + vertical bars)
create_group_pattern : function
    Creates a group pattern with multiple elements
diverging_gaussian : function
    Creates a Gaussian beam diverging from a virtual source
gaussian_beam : function
    Creates a Gaussian beam from complex beam parameter q
generate_usaf_pattern : function
    Generates USAF 1951 resolution test pattern
get_bar_width_pixels : function
    Calculate bar width in pixels for a given group and element
gradient_index_material : function
    Creates a gradient-index (GRIN) material with radial profile
hermite_gaussian : function
    Creates Hermite-Gaussian modes
layered_material : function
    Creates alternating layers of materials
laguerre_gaussian : function
    Creates Laguerre-Gaussian modes
plane_wave : function
    Creates a uniform plane wave with optional tilt
propagate_beam : function
    Generates a beam at multiple z positions as a PropagatingWavefront
radially_polarized_beam : function
    Generate a radially polarized beam
azimuthally_polarized_beam : function
    Generate an azimuthally polarized beam
linear_polarized_beam : function
    Generate a linearly polarized beam with arbitrary angle
x_polarized_beam : function
    Generate an x-polarized beam
y_polarized_beam : function
    Generate a y-polarized beam
circular_polarized_beam : function
    Generate a circularly polarized beam
generalized_cylindrical_vector_beam : function
    Generate a generalized cylindrical vector beam
sinusoidal_wave : function
    Creates a sinusoidal interference pattern
spherical_inclusion : function
    Creates a material with spherical inclusion
uniform_material : function
    Creates a uniform 3D material with constant refractive index


Notes
-----
All propagation functions are JAX-compatible and support automatic
differentiation. The lens functions can model both ideal and realistic
optical elements with aberrations.
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
