"""Ptychography through differentiable programming in JAX.

Extended Summary
----------------
A comprehensive toolkit for ptychography simulations and reconstructions
using JAX for automatic differentiation and acceleration. Supports both
optical and electron microscopy applications with fully differentiable
and JIT-compilable functions.

Submodules
----------
utils
    Common utility functions used throughout the code.
invert
    Inversion algorithms for phase retrieval and ptychography.
models
    Models for generating datasets for testing and validation.
optics
    Variety of different optical elements.
lenses
    Lens implementations and optical calculations.

Key Features
------------
- JAX-compatible:
    All functions support jit, grad, vmap, and other JAX transformations
- Automatic differentiation:
    Full support for gradient-based optimization
- Complex-valued optimization: Wirtinger calculus for complex parameters
- Multi-modal support: Handles both single and multi-modal probes
- Parallel processing: Device mesh support for distributed computing
- Type safety: Comprehensive type checking with jaxtyping and beartype

Notes
-----
This package is designed for research and development in ptychography.
All functions are optimized for JAX transformations and support both
CPU and GPU execution. For best performance, use JIT compilation
and consider using the provided factory functions for data validation.
"""

from . import invert, lenses, models, optics, utils

__all__: list[str] = [
    "invert",
    "lenses",
    "models",
    "optics",
    "utils",
]
