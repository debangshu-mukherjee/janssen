"""Lens implementations and optical calculations.

Extended Summary
----------------
Comprehensive lens modeling and optical propagation algorithms for simulating
light propagation through various optical elements. Includes implementations
of common lens types and propagation methods based on wave optics.

Submodules
----------
lens_elements
    Lens elements for optical simulations
lens_prop
    Lens propagation functions

Routine Listings
----------------
create_lens_phase
    Create phase profile for a lens based on its parameters
double_concave_lens
    Create parameters for a double concave lens
double_convex_lens
    Create parameters for a double convex lens
lens_focal_length
    Calculate focal length from lens parameters
lens_thickness_profile
    Calculate thickness profile of a lens
meniscus_lens
    Create parameters for a meniscus lens
plano_concave_lens
    Create parameters for a plano-concave lens
plano_convex_lens
    Create parameters for a plano-convex lens
propagate_through_lens
    Propagate optical wavefront through a lens
angular_spectrum_prop
    Angular spectrum propagation method
digital_zoom
    Digital zoom transformation for optical fields
fraunhofer_prop
    Fraunhofer (far-field) propagation
fresnel_prop
    Fresnel (near-field) propagation
lens_propagation
    General lens-based propagation
optical_zoom
    Optical zoom transformation

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
    "digital_zoom",
    "fraunhofer_prop",
    "fresnel_prop",
    "optical_zoom",
    "lens_propagation",
]
