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
NonJaxNumber : TypeAlias
    Type alias for Python numeric types
OpticalWavefront : PyTree
    PyTree for optical wavefront representation
OptimizerState : PyTree
    PyTree for optimizer state tracking
PropagatingWavefront : PyTree
    PyTree for propagating optical wavefront representation
PtychographyParams : PyTree
    PyTree for ptychography reconstruction parameters
SampleFunction : PyTree
    PyTree for sample representation
ScalarBool : TypeAlias
    Type alias for scalar boolean values
ScalarComplex : TypeAlias
    Type alias for scalar complex values
ScalarFloat : TypeAlias
    Type alias for scalar float values
ScalarInteger : TypeAlias
    Type alias for scalar integer values
ScalarNumeric : TypeAlias
    Type alias for any scalar numeric value
SlicedMaterialFunction : PyTree
    PyTree for 3D sliced material with complex refractive index

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
    NonJaxNumber,
    OpticalWavefront,
    OptimizerState,
    PropagatingWavefront,
    PtychographyParams,
    SampleFunction,
    ScalarBool,
    ScalarComplex,
    ScalarFloat,
    ScalarInteger,
    ScalarNumeric,
    SlicedMaterialFunction,
)

__all__: list[str] = [
    "create_mesh",
    "get_device_count",
    "make_diffractogram",
    "make_grid_params",
    "make_lens_params",
    "make_microscope_data",
    "make_optical_wavefront",
    "make_optimizer_state",
    "make_ptychography_params",
    "make_sample_function",
    "make_sliced_material_function",
    "shard_batch",
    "Diffractogram",
    "GridParams",
    "LensParams",
    "MicroscopeData",
    "NonJaxNumber",
    "OpticalWavefront",
    "OptimizerState",
    "PropagatingWavefront",
    "PtychographyParams",
    "SampleFunction",
    "ScalarBool",
    "ScalarComplex",
    "ScalarFloat",
    "ScalarInteger",
    "ScalarNumeric",
    "SlicedMaterialFunction",
]
