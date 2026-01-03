"""Plotting utilities for optical data visualization.

Extended Summary
----------------
Functions for visualizing optical wavefronts, diffraction patterns,
and other data structures from the janssen package.

Routine Listings
----------------
:func:`plot_intensity`
    Plot the intensity of an optical wavefront.
:func:`plot_amplitude`
    Plot the amplitude of an optical wavefront.
:func:`plot_phase`
    Plot the phase of an optical wavefront using HSV color mapping.
:func:`plot_complex_wavefront`
    Plot a complex optical wavefront using HSV color mapping.

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
