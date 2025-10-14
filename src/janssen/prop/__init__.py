"""Propagation methods for optical wavefronts.

Extended Summary
----------------
Various propagation algorithms for simulating optical field propagation
through different media and materials. Includes lens-based propagation,
material-based propagation, and general free-space propagation methods.

Submodules
----------
free_space_prop
    Free-space propagation functions using scalar diffraction theory
material_prop
    Material propagation functions for sliced material models (coming soon)

Routine Listings
----------------
angular_spectrum_prop : function
    Angular spectrum propagation method (no paraxial approximation).
correct_propagator : function
    Automatically selects the most appropriate propagation method.
digital_zoom : function
    Digital zoom transformation for optical fields.
fraunhofer_prop : function
    Fraunhofer (far-field) propagation.
fresnel_prop : function
    Fresnel (near-field) propagation.
lens_propagation : function
    Propagate optical wavefront through a lens.
optical_zoom : function
    Optical zoom transformation.

Notes
-----
All propagation functions are JAX-compatible and support automatic
differentiation. The module is designed to be extensible for new
propagation methods.
"""

from .free_space_prop import (
    angular_spectrum_prop,
    correct_propagator,
    digital_zoom,
    fraunhofer_prop,
    fresnel_prop,
    lens_propagation,
    optical_zoom,
)

__all__: list[str] = [
    "angular_spectrum_prop",
    "correct_propagator",
    "digital_zoom",
    "fraunhofer_prop",
    "fresnel_prop",
    "lens_propagation",
    "optical_zoom",
]
