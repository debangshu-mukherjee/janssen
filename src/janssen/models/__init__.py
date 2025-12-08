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
usaf_pattern
    USAF test pattern generation

Routine Listings
----------------
bessel_beam : function
    Creates a Bessel beam with specified cone angle
biological_cell : function
    Creates a biological cell model with nucleus
collimated_gaussian : function
    Creates a collimated Gaussian beam with flat phase
converging_gaussian : function
    Creates a Gaussian beam converging to a focus
create_bar_triplet : function
    Creates 3 parallel bars (horizontal or vertical)
create_element : function
    Creates a single element (horizontal + vertical bars)
diverging_gaussian : function
    Creates a Gaussian beam diverging from a virtual source
gaussian_beam : function
    Creates a Gaussian beam from complex beam parameter q
generate_usaf_pattern : function
    Generates USAF 1951 resolution test pattern
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
from .usaf_pattern import (
    create_bar_triplet,
    create_element,
    generate_usaf_pattern,
)

__all__: list[str] = [
    "bessel_beam",
    "biological_cell",
    "collimated_gaussian",
    "converging_gaussian",
    "create_bar_triplet",
    "create_element",
    "diverging_gaussian",
    "gaussian_beam",
    "generate_usaf_pattern",
    "gradient_index_material",
    "hermite_gaussian",
    "layered_material",
    "laguerre_gaussian",
    "plane_wave",
    "propagate_beam",
    "sinusoidal_wave",
    "spherical_inclusion",
    "uniform_material",
]
