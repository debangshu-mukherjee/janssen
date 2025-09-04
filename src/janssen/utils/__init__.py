"""Common utility functions used throughout the code.

Extended Summary
----------------
Core utilities for the janssen package including type definitions,
factory functions, and decorators for type checking and validation.
Provides the foundation for type-safe JAX programming with PyTrees.

Submodules
----------
factory
    Factory functions for creating data structures
types
    Type definitions and PyTrees

Routine Listings
----------------
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
SampleFunction : PyTree
    PyTree for sample representation
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
make_sample_function : function
    Factory function for SampleFunction creation
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

from .factory import (
    make_diffractogram,
    make_grid_params,
    make_lens_params,
    make_microscope_data,
    make_optical_wavefront,
    make_optimizer_state,
    make_sample_function,
)
from .types import (
    Diffractogram,
    GridParams,
    LensParams,
    MicroscopeData,
    OpticalWavefront,
    OptimizerState,
    SampleFunction,
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
    "SampleFunction",
    "make_diffractogram",
    "make_grid_params",
    "make_lens_params",
    "make_microscope_data",
    "make_optical_wavefront",
    "make_optimizer_state",
    "make_sample_function",
    "non_jax_number",
    "scalar_bool",
    "scalar_complex",
    "scalar_float",
    "scalar_integer",
    "scalar_numeric",
]
