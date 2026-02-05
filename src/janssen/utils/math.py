"""Mathematical utility functions for complex-valued computations.

Extended Summary
----------------
Core mathematical utilities for the janssen package, including
Wirtinger calculus for complex-valued optimization, FFT-based
shifting, and other mathematical operations commonly used in
optical simulations.

Routine Listings
----------------
flatten_params : function
    Flatten complex arrays to real parameter vector for optimization
unflatten_params : function
    Unflatten real parameter vector back to complex arrays
fourier_shift : function
    Shift a 2D field using FFT-based sub-pixel shifting
wirtinger_grad : function
    Compute the Wirtinger gradient of a complex-valued function

Notes
-----
All functions are JAX-compatible and support automatic differentiation.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, Optional, Sequence, Tuple, Union
from jaxtyping import Array, Complex, Float, jaxtyped

from janssen.types import ScalarFloat, ScalarInteger


@jax.jit
@jaxtyped(typechecker=beartype)
def flatten_params(
    sample: Complex[Array, " H W"],
    probe: Complex[Array, " H W"],
) -> Float[Array, " n"]:
    """Flatten complex arrays to real parameter vector for optimization.

    Converts two complex 2D arrays into a single real-valued vector by
    separating real and imaginary components. This is necessary for
    optimization algorithms that operate on real vector spaces.

    Parameters
    ----------
    sample : Complex[Array, " H W"]
        First complex array (e.g., object transmission function).
    probe : Complex[Array, " H W"]
        Second complex array (e.g., probe wavefront).

    Returns
    -------
    params : Float[Array, " n"]
        Flattened real parameter vector with n = 4 * H * W.
        Layout: [sample_real, sample_imag, probe_real, probe_imag]

    Examples
    --------
    >>> sample = jnp.ones((32, 32), dtype=jnp.complex128)
    >>> probe = jnp.ones((32, 32), dtype=jnp.complex128) * (1+1j)
    >>> params = flatten_params(sample, probe)
    >>> params.shape
    (4096,)

    See Also
    --------
    unflatten_params : Inverse operation to reconstruct complex arrays
    """
    result: Float[Array, " n"] = jnp.concatenate(
        [
            sample.real.ravel(),
            sample.imag.ravel(),
            probe.real.ravel(),
            probe.imag.ravel(),
        ]
    )
    return result


def unflatten_params(
    params: Float[Array, " n"],
    shape: Tuple[int, int],
) -> Tuple[Complex[Array, " H W"], Complex[Array, " H W"]]:
    """Unflatten real parameter vector back to complex arrays.

    Reconstructs two complex 2D arrays from a flattened real parameter
    vector. This is the inverse operation of flatten_params.

    Parameters
    ----------
    params : Float[Array, " n"]
        Flattened real parameter vector with n = 4 * H * W.
    shape : Tuple[int, int]
        Shape (H, W) of each output array.

    Returns
    -------
    sample : Complex[Array, " H W"]
        First complex array reconstructed from params[:2*H*W].
    probe : Complex[Array, " H W"]
        Second complex array reconstructed from params[2*H*W:].

    Notes
    -----
    This function cannot be JIT-compiled directly because it uses
    dynamic indexing based on the shape parameter. When used within
    a JIT-compiled function, ensure the shape is static via
    static_argnums.

    Examples
    --------
    >>> params = jnp.arange(4096, dtype=jnp.float64)
    >>> sample, probe = unflatten_params(params, (32, 32))
    >>> sample.shape, probe.shape
    ((32, 32), (32, 32))

    See Also
    --------
    flatten_params : Forward operation to create flattened vector
    """
    size: ScalarInteger = shape[0] * shape[1]
    sample: Complex[Array, " H W"] = params[:size].reshape(
        shape
    ) + 1j * params[size : 2 * size].reshape(shape)
    probe: Complex[Array, " H W"] = params[2 * size : 3 * size].reshape(
        shape
    ) + 1j * params[3 * size :].reshape(shape)
    return sample, probe


@jaxtyped(typechecker=beartype)
def fourier_shift(
    field: Complex[Array, " H W"],
    shift_x: ScalarFloat,
    shift_y: ScalarFloat,
) -> Complex[Array, " H W"]:
    r"""Shift a 2D field using FFT-based sub-pixel shifting.

    Applies a phase ramp in Fourier space to shift the field by
    (shift_x, shift_y) pixels relative to the center of the image.
    Supports sub-pixel shifts with high accuracy.

    Parameters
    ----------
    field : Complex[Array, " H W"]
        Input 2D complex field to shift. The field is assumed to be
        centered (i.e., the center of the image is the origin).
    shift_x : ScalarFloat
        Shift in x direction (columns) in pixels. Positive shifts
        move the field to the right.
    shift_y : ScalarFloat
        Shift in y direction (rows) in pixels. Positive shifts
        move the field downward.

    Returns
    -------
    shifted : Complex[Array, " H W"]
        Shifted field with same shape as input.

    Notes
    -----
    The shift is implemented by multiplying the Fourier transform of
    the field by a phase ramp:

    .. math::
        F_{shifted}(f_x, f_y) = F(f_x, f_y) \\cdot
        \\exp(-2\\pi i (f_x \\cdot \\Delta x + f_y \\cdot \\Delta y))

    This is equivalent to circular shifting with sub-pixel accuracy.
    The shift is relative to the center of the image, so a shift of
    (0, 0) leaves the field unchanged.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> field = jnp.ones((64, 64), dtype=jnp.complex128)
    >>> shifted = fourier_shift(field, 10.5, -5.25)  # Sub-pixel shift
    """
    ny: int = field.shape[0]
    nx: int = field.shape[1]
    freq_x: Float[Array, " W"] = jnp.fft.fftfreq(nx)
    freq_y: Float[Array, " H"] = jnp.fft.fftfreq(ny)
    fx: Float[Array, " H W"]
    fy: Float[Array, " H W"]
    fx, fy = jnp.meshgrid(freq_x, freq_y)
    phase_ramp: Complex[Array, " H W"] = jnp.exp(
        -2j * jnp.pi * (fx * shift_x + fy * shift_y)
    )
    field_ft: Complex[Array, " H W"] = jnp.fft.fft2(field)
    shifted_ft: Complex[Array, " H W"] = field_ft * phase_ramp
    shifted: Complex[Array, " H W"] = jnp.fft.ifft2(shifted_ft)
    return shifted


@jaxtyped(typechecker=beartype)
def wirtinger_grad(
    func2diff: Callable[..., Float[Array, " ..."]],
    argnums: Optional[Union[int, Sequence[int]]] = 0,
) -> Callable[
    ..., Union[Complex[Array, " ..."], Tuple[Complex[Array, " ..."], ...]]
]:
    r"""Compute the Wirtinger gradient of a complex-valued function.

    This function returns a new function that computes the Wirtinger
    gradient of the input function f with respect to the specified
    argument(s). This is based on the formula for Wirtinger derivative:

    .. math::
        \frac{\partial f}{\partial z} = \frac{1}{2} \left(
        \frac{\partial f}{\partial x} - i \frac{\partial f}{\partial y}
        \right)

    Parameters
    ----------
    func2diff : Callable[..., Float[Array, " ..."]]
        A complex-valued function to differentiate.
    argnums : Union[int, Sequence[int]], optional
        Specifies which argument(s) to compute the gradient with respect
        to. Can be an int or a sequence of ints. Default is 0.

    Returns
    -------
    grad_f : Callable
        A function that computes the Wirtinger gradient of f with
        respect to the specified argument(s). Returns a single array
        if argnums is an int, or a tuple of arrays if argnums is a
        sequence.

    Notes
    -----
    The Wirtinger derivative is essential for optimizing real-valued
    loss functions with respect to complex-valued parameters. It provides
    the correct gradient direction for steepest descent in the complex
    plane.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> def loss(z):
    ...     return jnp.sum(jnp.abs(z)**2)
    >>> grad_fn = wirtinger_grad(loss)
    >>> z = jnp.array([1+2j, 3+4j])
    >>> grad_fn(z)  # Returns the Wirtinger gradient
    """

    def grad_f(
        *args: Any,
    ) -> Union[Complex[Array, " ..."], Tuple[Complex[Array, " ..."], ...]]:
        def split_complex(args: Any) -> Tuple[Any, ...]:
            return tuple(
                jnp.real(arg) if jnp.iscomplexobj(arg) else arg for arg in args
            ) + tuple(
                jnp.imag(arg) if jnp.iscomplexobj(arg) else jnp.zeros_like(arg)
                for arg in args
            )

        def combine_complex(r: Any, i: Any) -> Tuple[Any, ...]:
            return tuple(
                rr + 1j * ii if jnp.iscomplexobj(arg) else rr
                for rr, ii, arg in zip(r, i, args, strict=False)
            )

        split_args: Tuple[Any, ...] = split_complex(args)
        n: ScalarInteger = len(args)

        def f_real(*split_args: Any) -> Float[Array, " ..."]:
            return jnp.real(
                func2diff(*combine_complex(split_args[:n], split_args[n:]))
            )

        def f_imag(*split_args: Any) -> Float[Array, " ..."]:
            return jnp.imag(
                func2diff(*combine_complex(split_args[:n], split_args[n:]))
            )

        gr: Union[Complex[Array, " ..."], Tuple[Complex[Array, " ..."], ...]]
        gi: Union[Complex[Array, " ..."], Tuple[Complex[Array, " ..."], ...]]
        gr = jax.grad(f_real, argnums=argnums)(*split_args)
        gi = jax.grad(f_imag, argnums=argnums)(*split_args)

        if isinstance(argnums, int):
            return 0.5 * (gr - 1j * gi)
        return tuple(
            0.5 * (grr - 1j * gii) for grr, gii in zip(gr, gi, strict=False)
        )

    return grad_f
