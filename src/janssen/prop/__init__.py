"""Propagation methods for optical wavefronts.

Extended Summary
----------------
Various propagation algorithms for simulating optical field propagation
through different media and materials. Includes lens-based propagation,
material-based propagation, free-space propagation, and high-NA vector
focusing methods.

Routine Listings
----------------
:func:`angular_spectrum_prop`
    Angular spectrum propagation method (no paraxial approximation).
:func:`aplanatic_apodization`
    Apply sqrt(cos(theta)) apodization for aplanatic lens systems.
:func:`compute_focal_volume`
    Compute 3D focal volume at multiple z planes.
:func:`correct_propagator`
    Automatically selects the most appropriate propagation method.
:func:`debye_wolf_focus`
    Compute focal field using Debye-Wolf formulation.
:func:`digital_zoom`
    Digital zoom transformation for optical fields.
:func:`fraunhofer_prop`
    Fraunhofer (far-field) propagation.
:func:`fraunhofer_prop_scaled`
    Fraunhofer propagation with output at specified pixel size.
:func:`fresnel_prop`
    Fresnel (near-field) propagation.
:func:`high_na_focus`
    Compute focal field using Richards-Wolf vector diffraction integrals.
:func:`lens_propagation`
    Propagate optical wavefront through a lens.
:func:`multislice_propagation`
    Propagate optical wavefront through a 3D material.
:func:`optical_path_length`
    Compute the optical path length through a material.
:func:`optical_zoom`
    Optical zoom transformation.
:func:`scalar_focus_for_comparison`
    Compute scalar focal field for comparison with vector result.
:func:`total_transmit`
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
    fraunhofer_prop_scaled,
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
    "fraunhofer_prop_scaled",
    "fresnel_prop",
    "high_na_focus",
    "lens_propagation",
    "multislice_propagation",
    "optical_path_length",
    "optical_zoom",
    "scalar_focus_for_comparison",
    "total_transmit",
]
