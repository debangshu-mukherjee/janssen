"""Inversion algorithms for phase retrieval and ptychography.

Extended Summary
----------------
Comprehensive algorithms for phase retrieval and ptychographic reconstruction
using differentiable programming techniques. Includes various optimization
strategies and loss functions for reconstructing complex-valued fields.

Submodules
----------
engine
    Reconstruction engine
ptychography
    Ptychographic algorithms
optimizers
    Optimization routines
loss_functions
    Loss function definitions

Routine Listings
----------------
create_loss_function : function
    Factory function for creating various loss functions
simple_microscope_ptychography
    Main ptychography reconstruction algorithm for optical microscopy
epie_optical : function
    Extended PIE algorithm for optical ptychography
single_pie_iteration : function
    Single iteration of PIE algorithm
single_pie_sequential : function
    Sequential PIE implementation for multiple positions
single_pie_vmap
    Vectorized PIE implementation using vmap
init_adam : function
    Initialize Adam optimizer state
init_adagrad : function
    Initialize Adagrad optimizer state
init_rmsprop : function
    Initialize RMSprop optimizer state

Notes
-----
All functions are JAX-compatible and support automatic differentiation.
The algorithms can be composed with JIT compilation for improved performance.
"""

from .engine import (
    epie_optical,
    single_pie_iteration,
    single_pie_sequential,
    single_pie_vmap,
)
from .loss_functions import create_loss_function
from .optimizers import init_adagrad, init_adam, init_rmsprop
from .ptychography import simple_microscope_ptychography

__all__: list[str] = [
    "create_loss_function",
    "simple_microscope_ptychography",
    "epie_optical",
    "single_pie_iteration",
    "single_pie_sequential",
    "single_pie_vmap",
    "init_adam",
    "init_adagrad",
    "init_rmsprop",
]
