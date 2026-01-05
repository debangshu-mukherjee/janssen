"""Utility functions for distributed computing and math operations.

Extended Summary
----------------
Utilities for distributed JAX computing across multiple devices and
mathematical helper functions for complex-valued operations.

Routine Listings
----------------
:func:`create_mesh`
    Creates a device mesh for data parallelism across available devices.
:func:`get_device_count`
    Gets the number of available JAX devices.
:func:`shard_batch`
    Shards array data across the batch dimension for parallel processing.
:func:`fourier_shift`
    FFT-based sub-pixel shifting of 2D fields.
:func:`wirtinger_grad`
    Compute the Wirtinger gradient of a complex-valued function.
:func:`bessel_j0`
    Compute J_0(x), regular Bessel function of the first kind, order 0.
:func:`bessel_jn`
    Compute J_n(x), regular Bessel function of the first kind, order n.
:func:`bessel_kv`
    Compute K_v(x), modified Bessel function of the second kind.

Notes
-----
For type definitions and PyTree classes, see the :mod:`janssen.types` module.
"""

from .bessel import bessel_j0, bessel_jn, bessel_kv
from .distributed import (
    create_mesh,
    get_device_count,
    shard_batch,
)
from .math import fourier_shift, wirtinger_grad

__all__: list[str] = [
    "bessel_j0",
    "bessel_jn",
    "bessel_kv",
    "create_mesh",
    "fourier_shift",
    "get_device_count",
    "shard_batch",
    "wirtinger_grad",
]
