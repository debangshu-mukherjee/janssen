"""Common utility functions used throughout the code.

Extended Summary
----------------
Core utilities for the janssen package including type definitions,
factory functions, and decorators for type checking and validation.
Provides the foundation for type-safe JAX programming with PyTrees.

Submodules
----------
decorators
    Decorators for type checking and JAX transformations
factory
    Factory functions for creating data structures
types
    Type definitions and PyTrees

Routine Listings
----------------
beartype
    Runtime type checker decorator
jaxtyped
    JAX array shape and dtype type checker
Diffractogram
    PyTree for storing diffraction patterns
GridParams
    PyTree for computational grid parameters
LensParams
    PyTree for lens optical parameters
MicroscopeData
    PyTree for microscopy data
OpticalWavefront
    PyTree for optical wavefront representation
OptimizerState
    PyTree for optimizer state tracking
SampleFunction
    PyTree for sample representation
make_diffractogram
    Factory function for Diffractogram creation
make_grid_params
    Factory function for GridParams creation
make_lens_params
    Factory function for LensParams creation
make_microscope_data
    Factory function for MicroscopeData creation
make_optical_wavefront
    Factory function for OpticalWavefront creation
make_optimizer_state
    Factory function for OptimizerState creation
make_sample_function
    Factory function for SampleFunction creation
non_jax_number
    Type alias for Python numeric types
scalar_bool
    Type alias for scalar boolean values
scalar_complex
    Type alias for scalar complex values
scalar_float
    Type alias for scalar float values
scalar_integer
    Type alias for scalar integer values
scalar_numeric
    Type alias for any scalar numeric value

Notes
-----
Always use factory functions for creating PyTree instances to ensure
proper type checking and validation. All PyTrees are registered with
JAX and support automatic differentiation.
"""

from .decorators import beartype, jaxtyped
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
    "beartype",
    "jaxtyped",
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
