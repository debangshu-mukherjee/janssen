"""Lens implementations and optical calculations.

Extended Summary
----------------
Models for generating datasets for testing and validation.

Submodules
----------
usaf_pattern
    USAF test pattern generation

Routine Listings
----------------
create_bar_triplet : function
    Creates 3 parallel bars (horizontal or vertical)
create_element : function
    Creates a single element (horizontal + vertical bars)
generate_usaf_pattern : function
    Generates USAF 1951 resolution test pattern


Notes
-----
All propagation functions are JAX-compatible and support automatic
differentiation. The lens functions can model both ideal and realistic
optical elements with aberrations.
"""

from .usaf_pattern import (
    create_bar_triplet,
    create_element,
    generate_usaf_pattern,
)

__all__: list[str] = [
    "create_bar_triplet",
    "create_element",
    "generate_usaf_pattern",
]
