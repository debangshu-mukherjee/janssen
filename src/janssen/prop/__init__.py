"""Propagation methods for optical wavefronts.

Extended Summary
----------------
Various propagation algorithms for simulating optical field propagation
through different media and materials. Includes lens-based propagation,
material-based propagation, free-space propagation, and high-NA vector
focusing methods.

Submodules
----------
free_space_prop
    Free-space propagation functions using scalar diffraction theory
material_prop
    Material propagation functions for sliced material models
vector_focusing
    High-NA vector focusing using Richards-Wolf diffraction integrals

Routine Listings
----------------
angular_spectrum_prop : function
    Angular spectrum propagation method (no paraxial approximation).
aplanatic_apodization : function
    Apply sqrt(cos(theta)) apodization for aplanatic lens systems.
compute_focal_volume : function
    Compute 3D focal volume at multiple z planes.
correct_propagator : function
    Automatically selects the most appropriate propagation method.
debye_wolf_focus : function
    Compute focal field using Debye-Wolf formulation.
digital_zoom : function
    Digital zoom transformation for optical fields.
fraunhofer_prop : function
    Fraunhofer (far-field) propagation.
fresnel_prop : function
    Fresnel (near-field) propagation.
high_na_focus : function
    Compute focal field using Richards-Wolf vector diffraction integrals.
lens_propagation : function
    Propagate optical wavefront through a lens.
multislice_propagation : function
    Propagate optical wavefront through a 3D material.
optical_path_length : function
    Compute the optical path length through a material.
optical_zoom : function
    Optical zoom transformation.
scalar_focus_for_comparison : function
    Compute scalar focal field for comparison with vector result.
total_transmit : function
    Compute the total transmission through a material.

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
from .material_prop import (
    multislice_propagation,
    optical_path_length,
    total_transmit,
)
from .vector_focusing import (
    aplanatic_apodization,
    compute_focal_volume,
    debye_wolf_focus,
    high_na_focus,
    scalar_focus_for_comparison,
)

__all__: list[str] = [
    "angular_spectrum_prop",
    "aplanatic_apodization",
    "compute_focal_volume",
    "correct_propagator",
    "debye_wolf_focus",
    "digital_zoom",
    "fraunhofer_prop",
    "fresnel_prop",
    "high_na_focus",
    "lens_propagation",
    "multislice_propagation",
    "optical_path_length",
    "optical_zoom",
    "scalar_focus_for_comparison",
    "total_transmit",
]
