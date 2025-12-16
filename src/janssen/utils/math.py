"""Mathematical utility functions for complex-valued computations.

Extended Summary
----------------
Core mathematical utilities for the janssen package, including
Wirtinger calculus for complex-valued optimization, FFT-based
shifting, and other mathematical operations commonly used in
optical simulations.

Routine Listings
----------------
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

from .types import ScalarFloat


@jaxtyped(typechecker=beartype)
def fourier_shift(
    field: Complex[Array, " H W"],
    shift_x: ScalarFloat,
    shift_y: ScalarFloat,
) -> Complex[Array, " H W"]:
    """Shift a 2D field using FFT-based sub-pixel shifting.

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

        split_args = split_complex(args)
        n = len(args)

        def f_real(*split_args: Any) -> Float[Array, " ..."]:
            return jnp.real(
                func2diff(*combine_complex(split_args[:n], split_args[n:]))
            )

        def f_imag(*split_args: Any) -> Float[Array, " ..."]:
            return jnp.imag(
                func2diff(*combine_complex(split_args[:n], split_args[n:]))
            )

        gr = jax.grad(f_real, argnums=argnums)(*split_args)
        gi = jax.grad(f_imag, argnums=argnums)(*split_args)

        if isinstance(argnums, int):
            return 0.5 * (gr - 1j * gi)
        return tuple(
            0.5 * (grr - 1j * gii) for grr, gii in zip(gr, gi, strict=False)
        )

    return grad_f
