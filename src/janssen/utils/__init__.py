"""Common utility functions used throughout the code.

Extended Summary
----------------
Core utilities for the janssen package including type definitions,
factory functions, and decorators for type checking and validation.
Provides the foundation for type-safe JAX programming with PyTrees.

Submodules
----------
distributed
    Multi-device utilities for scalable optical computing
factory
    Factory functions for creating data structures
types
    Type definitions and PyTrees

Routine Listings
----------------
create_mesh : function
    Creates a device mesh for data parallelism across available devices
get_device_count : function
    Gets the number of available JAX devices
shard_batch : function
    Shards array data across the batch dimension for parallel processing
Diffractogram : PyTree
    PyTree for storing diffraction patterns
GridParams : PyTree
    PyTree for computational grid parameters
LensParams : PyTree
    PyTree for lens optical parameters
MicroscopeData : PyTree
    PyTree for microscopy data
OpticalWavefront : PyTree
    PyTree for optical wavefront representation
OptimizerState : PyTree
    PyTree for optimizer state tracking
PtychographyParams : PyTree
    PyTree for ptychography reconstruction parameters
SampleFunction : PyTree
    PyTree for sample representation
SlicedMaterialFunction : PyTree
    PyTree for 3D sliced material with complex refractive index
make_diffractogram : function
    Factory function for Diffractogram creation
make_grid_params : function
    Factory function for GridParams creation
make_lens_params : function
    Factory function for LensParams creation
make_microscope_data : function
    Factory function for MicroscopeData creation
make_optical_wavefront : function
    Factory function for OpticalWavefront creation
make_optimizer_state : function
    Factory function for OptimizerState creation
make_ptychography_params : function
    Factory function for PtychographyParams creation
make_sample_function : function
    Factory function for SampleFunction creation
make_sliced_material_function : function
    Factory function for SlicedMaterialFunction creation
non_jax_number : TypeAlias
    Type alias for Python numeric types
scalar_bool : TypeAlias
    Type alias for scalar boolean values
scalar_complex : TypeAlias
    Type alias for scalar complex values
scalar_float : TypeAlias
    Type alias for scalar float values
scalar_integer : TypeAlias
    Type alias for scalar integer values
scalar_numeric : TypeAlias
    Type alias for any scalar numeric value

Notes
-----
Always use factory functions for creating PyTree instances to ensure
proper type checking and validation. All PyTrees are registered with
JAX and support automatic differentiation.
"""

from .distributed import (
    create_mesh,
    get_device_count,
    shard_batch,
)
from .factory import (
    make_diffractogram,
    make_grid_params,
    make_lens_params,
    make_microscope_data,
    make_optical_wavefront,
    make_optimizer_state,
    make_ptychography_params,
    make_sample_function,
    make_sliced_material_function,
)
from .types import (
    Diffractogram,
    GridParams,
    LensParams,
    MicroscopeData,
    OpticalWavefront,
    OptimizerState,
    PtychographyParams,
    SampleFunction,
    SlicedMaterialFunction,
    non_jax_number,
    scalar_bool,
    scalar_complex,
    scalar_float,
    scalar_integer,
    scalar_numeric,
)

__all__: list[str] = [
    "Diffractogram",
    "GridParams",
    "LensParams",
    "MicroscopeData",
    "OpticalWavefront",
    "OptimizerState",
    "PtychographyParams",
    "SampleFunction",
    "SlicedMaterialFunction",
    "create_mesh",
    "make_diffractogram",
    "make_grid_params",
    "make_lens_params",
    "make_microscope_data",
    "make_optical_wavefront",
    "make_optimizer_state",
    "make_ptychography_params",
    "make_sample_function",
    "make_sliced_material_function",
    "non_jax_number",
    "scalar_bool",
    "scalar_complex",
    "scalar_float",
    "scalar_integer",
    "scalar_numeric",
    "shard_batch",
    "get_device_count",
]
