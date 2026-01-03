"""Common utility functions used throughout the code.

Extended Summary
----------------
Core utilities for the janssen package including type definitions,
factory functions, and decorators for type checking and validation.
Provides the foundation for type-safe JAX programming with PyTrees.

Routine Listings
----------------
:func:`create_mesh`
    Creates a device mesh for data parallelism across available devices.
:func:`get_device_count`
    Gets the number of available JAX devices.
:func:`shard_batch`
    Shards array data across the batch dimension for parallel processing.
:func:`make_optical_wavefront`
    Factory function for OpticalWavefront creation.
:func:`make_propagating_wavefront`
    Factory function for PropagatingWavefront creation.
:func:`optical2propagating`
    Creates a PropagatingWavefront from a tuple of OpticalWavefronts.
:func:`make_coherent_mode_set`
    Factory function for CoherentModeSet creation.
:func:`make_polychromatic_wavefront`
    Factory function for PolychromaticWavefront creation.
:func:`make_mutual_intensity`
    Factory function for MutualIntensity creation.
:func:`make_diffractogram`
    Factory function for Diffractogram creation.
:func:`make_grid_params`
    Factory function for GridParams creation.
:func:`make_lens_params`
    Factory function for LensParams creation.
:func:`make_microscope_data`
    Factory function for MicroscopeData creation.
:func:`make_sample_function`
    Factory function for SampleFunction creation.
:func:`make_sliced_material_function`
    Factory function for SlicedMaterialFunction creation.
:func:`make_optimizer_state`
    Factory function for OptimizerState creation.
:func:`make_ptychography_params`
    Factory function for PtychographyParams creation.
:func:`make_ptychography_reconstruction`
    Factory function for PtychographyReconstruction creation.
:func:`make_epie_data`
    Factory function for EpieData creation.
:func:`make_epie_params`
    Factory function for EpieParams creation.
:func:`make_vector_wavefront_3d`
    Factory function for VectorWavefront3D creation.
:func:`jones_to_vector3d`
    Convert Jones field to 3-component vector field.
:func:`vector3d_to_jones`
    Extract transverse components as Jones field.
:func:`fourier_shift`
    FFT-based sub-pixel shifting of 2D fields.
:func:`wirtinger_grad`
    Compute the Wirtinger gradient of a complex-valued function.
:class:`OpticalWavefront`
    PyTree for optical wavefront representation.
:class:`PropagatingWavefront`
    PyTree for propagating optical wavefront representation.
:class:`CoherentModeSet`
    PyTree for coherent mode decomposition of partially coherent fields.
:class:`PolychromaticWavefront`
    PyTree for polychromatic/broadband field representation.
:class:`MutualIntensity`
    PyTree for full mutual intensity J(r1, r2) representation.
:class:`Diffractogram`
    PyTree for storing diffraction patterns.
:class:`GridParams`
    PyTree for computational grid parameters.
:class:`LensParams`
    PyTree for lens optical parameters.
:class:`MicroscopeData`
    PyTree for microscopy data.
:class:`SampleFunction`
    PyTree for sample representation.
:class:`SlicedMaterialFunction`
    PyTree for 3D sliced material with complex refractive index.
:class:`OptimizerState`
    PyTree for optimizer state tracking.
:class:`PtychographyParams`
    PyTree for ptychography reconstruction parameters.
:class:`PtychographyReconstruction`
    PyTree for ptychography reconstruction results.
:class:`EpieData`
    PyTree for ePIE algorithm data.
:class:`EpieParams`
    PyTree for ePIE algorithm parameters.
:class:`VectorWavefront3D`
    PyTree for full 3-component vector electric field.

Notes
-----
Always use factory functions for creating PyTree instances to ensure
proper type checking and validation. All PyTrees are registered with
JAX and support automatic differentiation.
"""

from .coherence_types import (
    CoherentModeSet,
    MutualIntensity,
    PolychromaticWavefront,
    make_coherent_mode_set,
    make_mutual_intensity,
    make_polychromatic_wavefront,
)
from .distributed import (
    create_mesh,
    get_device_count,
    shard_batch,
)
from .factory import (
    make_diffractogram,
    make_epie_data,
    make_epie_params,
    make_grid_params,
    make_lens_params,
    make_microscope_data,
    make_optical_wavefront,
    make_optimizer_state,
    make_propagating_wavefront,
    make_ptychography_params,
    make_ptychography_reconstruction,
    make_sample_function,
    make_sliced_material_function,
    optical2propagating,
)
from .math import fourier_shift, wirtinger_grad
from .types import (
    Diffractogram,
    EpieData,
    EpieParams,
    GridParams,
    LensParams,
    MicroscopeData,
    NonJaxNumber,
    OpticalWavefront,
    OptimizerState,
    PropagatingWavefront,
    PtychographyParams,
    PtychographyReconstruction,
    SampleFunction,
    ScalarBool,
    ScalarComplex,
    ScalarFloat,
    ScalarInteger,
    ScalarNumeric,
    SlicedMaterialFunction,
)
from .vector_types import (
    VectorWavefront3D,
    jones_to_vector3d,
    make_vector_wavefront_3d,
    vector3d_to_jones,
)

__all__: list[str] = [
    # Distributed utilities
    "create_mesh",
    "get_device_count",
    "shard_batch",
    # Factory functions
    "make_coherent_mode_set",
    "make_diffractogram",
    "make_epie_data",
    "make_epie_params",
    "make_grid_params",
    "make_lens_params",
    "make_microscope_data",
    "make_mutual_intensity",
    "make_optical_wavefront",
    "make_optimizer_state",
    "make_polychromatic_wavefront",
    "make_propagating_wavefront",
    "make_ptychography_params",
    "make_ptychography_reconstruction",
    "make_sample_function",
    "make_sliced_material_function",
    "make_vector_wavefront_3d",
    "optical2propagating",
    # Vector type utilities
    "jones_to_vector3d",
    "vector3d_to_jones",
    # Math utilities
    "fourier_shift",
    "wirtinger_grad",
    # Coherence types
    "CoherentModeSet",
    "MutualIntensity",
    "PolychromaticWavefront",
    # Core types
    "Diffractogram",
    "EpieData",
    "EpieParams",
    "GridParams",
    "LensParams",
    "MicroscopeData",
    "NonJaxNumber",
    "OpticalWavefront",
    "OptimizerState",
    "PropagatingWavefront",
    "PtychographyParams",
    "PtychographyReconstruction",
    "SampleFunction",
    "ScalarBool",
    "ScalarComplex",
    "ScalarFloat",
    "ScalarInteger",
    "ScalarNumeric",
    "SlicedMaterialFunction",
    "VectorWavefront3D",
]
