"""Lens implementations and optical calculations.

Extended Summary
----------------
Comprehensive lens modeling for simulating optical elements. Includes
implementations of common lens types and their optical properties.
For propagation algorithms, see the janssen.prop submodule.

Routine Listings
----------------
:func:`create_lens_phase`
    Create phase profile for a lens based on its parameters.
:func:`lens_focal_length`
    Calculate focal length from lens parameters.
:func:`lens_thickness_profile`
    Calculate thickness profile of a lens.
:func:`propagate_through_lens`
    Propagate optical wavefront through a lens.
:func:`plano_convex_lens`
    Create parameters for a plano-convex lens.
:func:`plano_concave_lens`
    Create parameters for a plano-concave lens.
:func:`double_convex_lens`
    Create parameters for a double convex lens.
:func:`double_concave_lens`
    Create parameters for a double concave lens.
:func:`meniscus_lens`
    Create parameters for a meniscus lens.

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
