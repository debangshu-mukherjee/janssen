"""
Module: janssen.common
------------------------
Common utility functions used throughout the code.

Submodules
----------
- `decorators`:
    Decorators for type checking and JAX transformations
- `types`:
    Data structures and type definitions for common use
"""

from .decorators import beartype, jaxtyped
from .types import (Diffractogram, GridParams, LensParams, MicroscopeData,
                    OpticalWavefront, SampleFunction, make_diffractogram,
                    make_grid_params, make_lens_params, make_microscope_data,
                    make_optical_wavefront, make_sample_function,
                    non_jax_number, scalar_complex, scalar_float,
                    scalar_integer, scalar_numeric)

__all__: list[str] = [
    "beartype",
    "jaxtyped",
    "Diffractogram",
    "GridParams",
    "LensParams",
    "MicroscopeData",
    "OpticalWavefront",
    "SampleFunction",
    "make_diffractogram",
    "make_grid_params",
    "make_lens_params",
    "make_microscope_data",
    "make_optical_wavefront",
    "make_sample_function",
    "non_jax_number",
    "scalar_complex",
    "scalar_float",
    "scalar_integer",
    "scalar_numeric",
]
