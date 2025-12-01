"""Plotting utilities for optical data visualization.

Extended Summary
----------------
Functions for visualizing optical wavefronts, diffraction patterns,
and other data structures from the janssen package.

Submodules
----------
wavefront
    Wavefront visualization functions

Routine Listings
----------------
plot_amplitude : function
    Plot the amplitude of an optical wavefront
plot_complex_wavefront : function
    Plot a complex optical wavefront using HSV color mapping
plot_intensity : function
    Plot the intensity of an optical wavefront
plot_phase : function
    Plot the phase of an optical wavefront using HSV color mapping

Notes
-----
These plotting functions are designed for data visualization only and
do not require JAX compatibility. They accept PyTree data structures
from the janssen package.
"""

from .wavefront import (
    plot_amplitude,
    plot_complex_wavefront,
    plot_intensity,
    plot_phase,
)

__all__: list[str] = [
    "plot_amplitude",
    "plot_complex_wavefront",
    "plot_intensity",
    "plot_phase",
]
