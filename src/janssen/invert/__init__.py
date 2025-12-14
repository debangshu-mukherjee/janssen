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

Routine Listings
----------------
create_loss_function : function
    Factory function for creating various loss functions
epie_optical : function
    Extended PIE algorithm for optical ptychography
simple_microscope_ptychography : function
    Main ptychography reconstruction algorithm using PtychographyParams
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
from .loss_functions import create_loss_function
from .ptychography import simple_microscope_ptychography

__all__: list[str] = [
    "create_loss_function",
    "epie_optical",
    "simple_microscope_ptychography",
    "single_pie_iteration",
    "single_pie_sequential",
    "single_pie_vmap",
]
