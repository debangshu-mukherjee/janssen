"""Ptychography through differentiable programming in JAX.

Extended Summary
----------------
A comprehensive toolkit for ptychography simulations and reconstructions
using JAX for automatic differentiation and acceleration. Supports both
optical and electron microscopy applications with fully differentiable
and JIT-compilable functions.

Routine Listings
----------------
:mod:`coherence`
    Partial coherence support for spatial and temporal coherence effects.
:mod:`invert`
    Inversion algorithms for phase retrieval and ptychography.
:mod:`lenses`
    Lens implementations and optical calculations.
:mod:`models`
    Models for generating datasets for testing and validation.
:mod:`optics`
    Variety of different optical elements.
:mod:`plots`
    Plotting utilities for optical data visualization.
:mod:`prop`
    Propagation methods for optical wavefronts.
:mod:`scopes`
    Microscope implementations and forward models.
:mod:`utils`
    Common utility functions used throughout the code.

Examples
--------
>>> import janssen as js
>>> wavefront = js.models.collimated_gaussian(
...     wavelength=632.8e-9, waist=1e-3, grid_size=(256, 256), dx=10e-6
... )
>>> propagated = js.prop.angular_spectrum_prop(wavefront, distance=0.1)
>>> js.plots.plot_intensity(propagated)

Notes
-----
All computations are JAX-compatible and support automatic differentiation
for gradient-based optimization of optical systems and phase retrieval.
"""

import os
from importlib.metadata import version

# Enable multi-threaded CPU execution for JAX (before importing JAX)
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
