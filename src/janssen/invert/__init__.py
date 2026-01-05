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
:class:`MixedStatePtychoData`
    PyTree for mixed-state ptychography reconstruction state.
:func:`make_mixed_state_ptycho_data`
    Factory function for MixedStatePtychoData creation.
:func:`mixed_state_forward`
    Compute predicted diffraction patterns for all positions.
:func:`mixed_state_forward_single_position`
    Forward model for one scan position with mixed-state illumination.
:func:`mixed_state_loss`
    Compute reconstruction loss for mixed-state ptychography.
:func:`coherence_parameterized_loss`
    Loss function with coherence width as optimizable parameter.
:func:`mixed_state_gradient_step`
    Single gradient descent step for mixed-state reconstruction.
:func:`mixed_state_reconstruct`
    Run mixed-state ptychography reconstruction.

Notes
-----
All functions are JAX-compatible and support automatic differentiation.
The algorithms can be composed with JIT compilation for improved
performance.
"""

from janssen.types import MixedStatePtychoData, make_mixed_state_ptycho_data

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
from .mixed_state import (
    coherence_parameterized_loss,
    mixed_state_forward,
    mixed_state_forward_single_position,
    mixed_state_gradient_step,
    mixed_state_loss,
    mixed_state_reconstruct,
)
from .ptychography import (
    simple_microscope_epie,
    simple_microscope_ptychography,
)

__all__: list[str] = [
    "MixedStatePtychoData",
    "coherence_parameterized_loss",
    "compute_fov_and_positions",
    "create_loss_function",
    "epie_optical",
    "init_simple_epie",
    "init_simple_microscope",
    "make_mixed_state_ptycho_data",
    "mixed_state_forward",
    "mixed_state_forward_single_position",
    "mixed_state_gradient_step",
    "mixed_state_loss",
    "mixed_state_reconstruct",
    "simple_microscope_epie",
    "simple_microscope_ptychography",
    "single_pie_iteration",
    "single_pie_sequential",
    "single_pie_vmap",
]
