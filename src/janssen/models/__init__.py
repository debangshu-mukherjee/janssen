"""Lens implementations and optical calculations.

Extended Summary
----------------
Models for generating datasets for testing and validation.

Submodules
----------
material_models
    Material models for optical simulations
usaf_pattern
    USAF test pattern generation

Routine Listings
----------------
biological_cell : function
    Creates a biological cell model with nucleus
create_bar_triplet : function
    Creates 3 parallel bars (horizontal or vertical)
create_element : function
    Creates a single element (horizontal + vertical bars)
generate_usaf_pattern : function
    Generates USAF 1951 resolution test pattern
gradient_index_material : function
    Creates a gradient-index (GRIN) material with radial profile
layered_material : function
    Creates alternating layers of materials
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
    "biological_cell",
    "create_bar_triplet",
    "create_element",
    "generate_usaf_pattern",
    "gradient_index_material",
    "layered_material",
    "spherical_inclusion",
    "uniform_material",
]
