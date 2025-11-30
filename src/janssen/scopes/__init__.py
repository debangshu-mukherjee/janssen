"""Microscope implementations and forward models.

Extended Summary
----------------
Complete forward models for optical microscopy including diffraction
patterns, light-sample interactions, and multi-position imaging.

Submodules
----------
simple_microscopes
    Simple microscope forward models for optical microscopy

Routine Listings
----------------
diffractogram_noscale : function
    Calculates the diffractogram without scaling camera pixel size
linear_interaction : function
    Propagates an optical wavefront through a sample using linear interaction
simple_diffractogram : function
    Calculates the diffractogram using a simple model
simple_microscope : function
    Calculates 3D diffractograms at all pixel positions in parallel

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
