"""Lens implementations and optical calculations.

Extended Summary
----------------
Comprehensive lens modeling and optical propagation algorithms for
simulating light propagation through various optical elements. Includes
implementations of common lens types and propagation methods based on
wave optics.

Submodules
----------
lens_elements
    Lens elements for optical simulations
lens_prop
    Lens propagation functions

Routine Listings
----------------
create_lens_phase : function
    Create phase profile for a lens based on its parameters
double_concave_lens : function
    Create parameters for a double concave lens
double_convex_lens : function
    Create parameters for a double convex lens
lens_focal_length : function
    Calculate focal length from lens parameters
lens_thickness_profile : function
    Calculate thickness profile of a lens
meniscus_lens : function
    Create parameters for a meniscus lens
plano_concave_lens : function
    Create parameters for a plano-concave lens
plano_convex_lens : function
    Create parameters for a plano-convex lens
propagate_through_lens : function
    Propagate optical wavefront through a lens
angular_spectrum_prop : function
    Angular spectrum propagation method
digital_zoom : function
    Digital zoom transformation for optical fields
fraunhofer_prop : function
    Fraunhofer (far-field) propagation
fresnel_prop : function
    Fresnel (near-field) propagation
lens_propagation : function
    General lens-based propagation
optical_zoom : function
    Optical zoom transformation
correct_propagator : function
    Automatically selects the most appropriate propagation method

Notes
-----
All propagation functions are JAX-compatible and support automatic
differentiation. The lens functions can model both ideal and realistic
optical elements with aberrations.
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
from .lens_prop import (
    angular_spectrum_prop,
    correct_propagator,
    digital_zoom,
    fraunhofer_prop,
    fresnel_prop,
    lens_propagation,
    optical_zoom,
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
    "angular_spectrum_prop",
    "correct_propagator",
    "digital_zoom",
    "fraunhofer_prop",
    "fresnel_prop",
    "optical_zoom",
    "lens_propagation",
]
