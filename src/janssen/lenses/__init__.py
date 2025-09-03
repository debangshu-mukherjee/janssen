"""
Module: janssen.lenses
----------------------
Differentiable optical lenses.

This package implements various lenses and optical propagators.

Submodules
----------
len_elements
    Optics model for simulation of optical lenses.
len_prop
    Lens propagation functions.
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
