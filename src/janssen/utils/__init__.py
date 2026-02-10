"""Utility functions for distributed computing and math operations.

Extended Summary
----------------
Utilities for distributed JAX computing across multiple devices,
mathematical helper functions for complex-valued operations, and
general-purpose optimization algorithms.

Routine Listings
----------------
:func:`bessel_j0`
    Compute J_0(x), regular Bessel function of the first kind, order 0.
:func:`bessel_jn`
    Compute J_n(x), regular Bessel function of the first kind, order n.
:func:`bessel_kv`
    Compute K_v(x), modified Bessel function of the second kind.
:func:`jt_residual`
    Compute residuals and J^T @ r simultaneously.
:func:`create_mesh`
    Creates a device mesh for data parallelism across available devices.
:func:`jtj_diag`
    Estimate diagonal of J^T J for preconditioning.
:func:`max_eigenval`
    Estimate largest eigenvalue of J^T J via power iteration.
:func:`flatten_params`
    Flatten complex arrays to real parameter vector for optimization.
:func:`fourier_shift`
    FFT-based sub-pixel shifting of 2D fields.
:func:`gn_solve`
    High-level solver that runs Gauss-Newton until convergence.
:func:`gn_loss_history`
    Gauss-Newton solver with per-iteration loss tracking only.
:func:`gn_history`
    Gauss-Newton solver with per-iteration history tracking.
:func:`gn_step`
    Generic Gauss-Newton step with trust-region damping.
:func:`get_device_count`
    Gets the number of available JAX devices.
:func:`get_device_memory_gb`
    Detects device count and memory per device (GB) for GPUs/CPUs.
:func:`hessian_matvec`
    Create exact Hessian-vector product operator.
:func:`jtj_matvec`
    Create Jacobian-free (J^T J + λI) operator for Gauss-Newton.
:func:`shard_batch`
    Shards array data across the batch dimension for parallel processing.
:func:`unflatten_params`
    Unflatten real parameter vector back to complex arrays.
:func:`wirtinger_grad`
    Compute the Wirtinger gradient of a complex-valued function.

Notes
-----
For type definitions and PyTree classes, see the :mod:`janssen.types` module.
"""

from .bessel import bessel_j0, bessel_jn, bessel_kv
from .distributed import (
    create_mesh,
    get_device_count,
    get_device_memory_gb,
    shard_batch,
)
from .gauss_newton import (
    jt_residual,
    jtj_diag,
    max_eigenval,
    gn_solve,
    gn_loss_history,
    gn_history,
    gn_step,
    hessian_matvec,
    jtj_matvec,
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
    "jt_residual",
    "create_mesh",
    "jtj_diag",
    "max_eigenval",
    "flatten_params",
    "fourier_shift",
    "gn_solve",
    "gn_loss_history",
    "gn_history",
    "gn_step",
    "get_device_count",
    "get_device_memory_gb",
    "hessian_matvec",
    "jtj_matvec",
    "shard_batch",
    "unflatten_params",
    "wirtinger_grad",
]
