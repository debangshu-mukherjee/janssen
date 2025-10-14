"""Lens implementations and optical calculations.

Extended Summary
----------------
Comprehensive lens modeling for simulating optical elements. Includes
implementations of common lens types and their optical properties.
For propagation algorithms, see the janssen.prop submodule.

Submodules
----------
lens_elements
    Lens elements for optical simulations

Routine Listings
----------------
create_lens_phase : function
    Create phase profile for a lens based on its parameters.
double_concave_lens : function
    Create parameters for a double concave lens.
double_convex_lens : function
    Create parameters for a double convex lens.
lens_focal_length : function
    Calculate focal length from lens parameters.
lens_thickness_profile : function
    Calculate thickness profile of a lens.
meniscus_lens : function
    Create parameters for a meniscus lens.
plano_concave_lens : function
    Create parameters for a plano-concave lens.
plano_convex_lens : function
    Create parameters for a plano-convex lens.
propagate_through_lens : function
    Propagate optical wavefront through a lens.

Notes
-----
All lens functions are JAX-compatible and support automatic
differentiation. The lens functions can model both ideal and realistic
optical elements with aberrations.

For propagation methods (angular_spectrum_prop, fresnel_prop, etc.),
see janssen.prop.
"""

from .lens_elements import (
    create_lens_phase,
    double_concave_lens,
    double_convex_lens,
    lens_focal_length,
    lens_thickness_profile,
    meniscus_lens,
    plano_concave_lens,
    plano_convex_lens,
    propagate_through_lens,
)

__all__: list[str] = [
    "create_lens_phase",
    "double_concave_lens",
    "double_convex_lens",
    "lens_focal_length",
    "lens_thickness_profile",
    "meniscus_lens",
    "plano_concave_lens",
    "plano_convex_lens",
    "propagate_through_lens",
]
