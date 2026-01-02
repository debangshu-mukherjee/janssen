"""Ptychography through differentiable programming in JAX.

Extended Summary
----------------
A comprehensive toolkit for ptychography simulations and reconstructions
using JAX for automatic differentiation and acceleration. Supports both
optical and electron microscopy applications with fully differentiable
and JIT-compilable functions.

Submodules
----------
coherence
    Partial coherence support for spatial and temporal coherence effects.
invert
    Inversion algorithms for phase retrieval and ptychography.
lenses
    Lens implementations and optical calculations.
models
    Models for generating datasets for testing and validation.
optics
    Variety of different optical elements.
plots
    Plotting utilities for optical data visualization.
prop
    Propagation methods for optical wavefronts.
scopes
    Microscope implementations and forward models.
utils
    Common utility functions used throughout the code.

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

import os
from importlib.metadata import version

# Enable multi-threaded CPU execution for JAX (must be set before importing JAX)
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0",
)

# Enable 64-bit precision in JAX (must be set before importing submodules)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from . import (  # noqa: E402, I001
    coherence,
    invert,
    lenses,
    models,
    optics,
    plots,
    prop,
    scopes,
    utils,
)

__version__: str = version("janssen")

__all__: list[str] = [
    "__version__",
    "coherence",
    "invert",
    "lenses",
    "models",
    "optics",
    "plots",
    "prop",
    "scopes",
    "utils",
]
