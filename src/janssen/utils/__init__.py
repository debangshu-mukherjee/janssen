"""
Module: janssen.utils.
----------------------

Common utility functions used throughout the code.

Submodules
----------
decorators
    Decorators for type checking and JAX transformations
factory
    Factory functions for creating PyTrees with data validation, 
    without the need to instantiate the classes directly. 
    The PyTrees themselves are in types.py
types
    Data structures and type definitions for common use
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
