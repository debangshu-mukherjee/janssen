"""Microscope implementations and forward models.

Extended Summary
----------------
Complete forward models for optical microscopy including diffraction
patterns, light-sample interactions, and multi-position imaging.

Routine Listings
----------------
:func:`simple_microscope`
    Calculates 3D diffractograms at all pixel positions in parallel.
:func:`simple_diffractogram`
    Calculates the diffractogram using a simple model.
:func:`diffractogram_noscale`
    Calculates the diffractogram without scaling camera pixel size.
:func:`linear_interaction`
    Propagates optical wavefront through sample using linear interaction.

Notes
-----
These functions provide complete forward models for optical microscopy
and are designed for use in inverse problems and ptychography reconstruction.
All functions are JAX-compatible and support automatic differentiation.
"""

from .simple_microscopes import (
    diffractogram_noscale,
    linear_interaction,
    simple_diffractogram,
    simple_microscope,
)

__all__: list[str] = [
    "diffractogram_noscale",
    "linear_interaction",
    "simple_diffractogram",
    "simple_microscope",
]
