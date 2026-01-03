"""Inversion algorithms for phase retrieval and ptychography.

Extended Summary
----------------
Comprehensive algorithms for phase retrieval and ptychographic
reconstruction using differentiable programming techniques. Includes
various optimization strategies and loss functions for reconstructing
complex-valued fields.

Routine Listings
----------------
:func:`epie_optical`
    Extended PIE algorithm for optical ptychography.
:func:`single_pie_iteration`
    Single iteration of PIE algorithm.
:func:`single_pie_sequential`
    Sequential PIE implementation for multiple positions.
:func:`single_pie_vmap`
    Vectorized PIE implementation using vmap.
:func:`simple_microscope_epie`
    Ptychography reconstruction using extended PIE algorithm.
:func:`simple_microscope_ptychography`
    Resumable ptychography reconstruction using gradient-based optimization.
:func:`create_loss_function`
    Factory function for creating various loss functions.
:func:`compute_fov_and_positions`
    Compute FOV size and normalized positions from experimental data.
:func:`init_simple_microscope`
    Initialize reconstruction by inverting simple microscope forward model.
:func:`init_simple_epie`
    Initialize ePIE reconstruction.

Notes
-----
All functions are JAX-compatible and support automatic differentiation.
The algorithms can be composed with JIT compilation for improved
performance.
"""

from .engine import (
    epie_optical,
    single_pie_iteration,
    single_pie_sequential,
    single_pie_vmap,
)
from .initialization import (
    compute_fov_and_positions,
    init_simple_epie,
    init_simple_microscope,
)
from .loss_functions import create_loss_function
from .ptychography import (
    simple_microscope_epie,
    simple_microscope_ptychography,
)

__all__: list[str] = [
    "compute_fov_and_positions",
    "create_loss_function",
    "epie_optical",
    "init_simple_epie",
    "init_simple_microscope",
    "simple_microscope_epie",
    "simple_microscope_ptychography",
    "single_pie_iteration",
    "single_pie_sequential",
    "single_pie_vmap",
]
