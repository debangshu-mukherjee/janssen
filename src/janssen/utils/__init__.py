"""Utility functions for distributed computing and math operations.

Extended Summary
----------------
Utilities for distributed JAX computing across multiple devices,
mathematical helper functions for complex-valued operations, and
general-purpose optimization algorithms.

Routine Listings
----------------
:func:`create_mesh`
    Creates a device mesh for data parallelism across available devices.
:func:`get_device_count`
    Gets the number of available JAX devices.
:func:`shard_batch`
    Shards array data across the batch dimension for parallel processing.
:func:`flatten_params`
    Flatten complex arrays to real parameter vector for optimization.
:func:`unflatten_params`
    Unflatten real parameter vector back to complex arrays.
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
:func:`make_jtj_matvec`
    Create Jacobian-free (J^T J + λI) operator for Gauss-Newton.
:func:`compute_jt_residual`
    Compute residuals and J^T @ r simultaneously.
:func:`make_hessian_matvec`
    Create exact Hessian-vector product operator.
:func:`gauss_newton_step`
    Generic Gauss-Newton step with trust-region damping.
:func:`gauss_newton_solve`
    High-level solver that runs Gauss-Newton until convergence.
:func:`estimate_max_eigenvalue`
    Estimate largest eigenvalue of J^T J via power iteration.
:func:`estimate_jtj_diagonal`
    Estimate diagonal of J^T J for preconditioning.

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
from .gauss_newton import (
    compute_jt_residual,
    estimate_jtj_diagonal,
    estimate_max_eigenvalue,
    gauss_newton_solve,
    gauss_newton_step,
    make_hessian_matvec,
    make_jtj_matvec,
)
from .math import (
    flatten_params,
    fourier_shift,
    unflatten_params,
    wirtinger_grad,
)

__all__: list[str] = [
    "bessel_j0",
    "bessel_jn",
    "bessel_kv",
    "compute_jt_residual",
    "create_mesh",
    "estimate_jtj_diagonal",
    "estimate_max_eigenvalue",
    "flatten_params",
    "fourier_shift",
    "gauss_newton_solve",
    "gauss_newton_step",
    "get_device_count",
    "make_hessian_matvec",
    "make_jtj_matvec",
    "shard_batch",
    "unflatten_params",
    "wirtinger_grad",
]
