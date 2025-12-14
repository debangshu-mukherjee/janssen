"""Inversion algorithms for phase retrieval and ptychography.

Extended Summary
----------------
Comprehensive algorithms for phase retrieval and ptychographic
reconstruction
using differentiable programming techniques. Includes various
optimization
strategies and loss functions for reconstructing complex-valued fields.

Submodules
----------
engine
    Reconstruction engine
ptychography
    Ptychographic algorithms
loss_functions
    Loss function definitions
initialization
    Sample initialization strategies

Routine Listings
----------------
compute_fov_and_positions : function
    Compute FOV size and normalized positions from experimental data
create_loss_function : function
    Factory function for creating various loss functions
epie_optical : function
    Extended PIE algorithm for optical ptychography
init_simple_microscope : function
    Initialize reconstruction by inverting the simple microscope forward model
simple_microscope_ptychography : function
    Resumable ptychography reconstruction algorithm
single_pie_iteration : function
    Single iteration of PIE algorithm
single_pie_sequential : function
    Sequential PIE implementation for multiple positions
single_pie_vmap : function
    Vectorized PIE implementation using vmap

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
    init_simple_microscope,
)
from .loss_functions import create_loss_function
from .ptychography import simple_microscope_ptychography

__all__: list[str] = [
    "compute_fov_and_positions",
    "create_loss_function",
    "epie_optical",
    "init_simple_microscope",
    "simple_microscope_ptychography",
    "single_pie_iteration",
    "single_pie_sequential",
    "single_pie_vmap",
]
