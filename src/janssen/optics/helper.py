"""Helper functions for optical simulations.

Extended Summary
----------------
Utility functions for creating computational grids, manipulating optical
fields, and performing common operations in optical simulations.

Routine Listings
----------------
create_spatial_grid : function
    Creates a 2D spatial grid for optical propagation
normalize_field : function
    Normalizes a complex field to unit power
add_phase_screen : function
    Adds a phase screen to a complex field
field_intensity : function
    Calculates intensity from a complex field
scale_pixel : function
    Rescales OpticalWavefront pixel size while keeping array shape fixed

Notes
-----
These helper functions provide common operations needed in optical
simulations and are optimized for use with JAX transformations.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple, Union
from jaxtyping import Array, Complex, Float, Int, Num, jaxtyped

from janssen.utils import (
    OpticalWavefront,
    ScalarFloat,
    ScalarInteger,
    ScalarNumeric,
    make_optical_wavefront,
)


@jaxtyped(typechecker=beartype)
def sellmeier(
    wavelength_nm: ScalarNumeric | Num[Array, " nn"],
    sellmeier_b: Num[Array, " 3"],
    sellmeier_c: Num[Array, " 3"],
) -> Union[Float[Array, " "], Float[Array, " nn"]]:
    r"""Calculate refractive index using the Sellmeier equation.

    Parameters
    ----------
    wavelength_nm : ScalarNumeric | Num[Array, " nn"]
        Wavelength in nanometers. Can be scalar or array.
    sellmeier_b : Num[Array, " 3"]
        Sellmeier B coefficients [B1, B2, B3].
    sellmeier_c : Num[Array, " 3"]
        Sellmeier C coefficients [C1, C2, C3] in micrometers squared.

    Returns
    -------
    n : Union[Float[Array, " "], Float[Array, " nn"]]
        Refractive index. Shape matches input wavelength.

    Notes
    -----
    The Sellmeier equation relates refractive index to wavelength:

    .. math::
        n^2(\lambda) = 1 + \sum_{i=1}^{3} \frac{B_i\lambda^2}{\lambda^2 - C_i}

    where :math:`\lambda` is in micrometers and :math:`C_i` are in
    micrometers squared.
    """
    wavelength_um_sq: Float[Array, " nn"] | Float[Array, " "] = (
        wavelength_nm / 1000.0
    ) ** 2
    terms: Num[Array, "... 3"] = (
        sellmeier_b * wavelength_um_sq[..., None]
    ) / (wavelength_um_sq[..., None] - sellmeier_c)
    n_squared: Float[Array, " nn"] | Float[Array, " "] = 1.0 + jnp.sum(
        terms, axis=-1
    )
    refractive_index: Float[Array, " nn"] | Float[Array, " "] = jnp.sqrt(
        n_squared
    )
    return refractive_index


@jaxtyped(typechecker=beartype)
def create_spatial_grid(
    diameter: ScalarNumeric | Num[Array, " 2"],
    num_points: ScalarInteger | Int[Array, " 2"],
) -> Tuple[Float[Array, " hh ww"], Float[Array, " hh ww"]]:
    """
    Create a 2D spatial grid for optical propagation.

    Parameters
    ----------
    diameter : ScalarNumeric | Num[Array, " 2"]
        Physical size of the grid in meters. Can be a scalar (square grid)
        or array of shape (2,) with [diameter_x, diameter_y] for rectangular
        grids.
    num_points : ScalarInteger | Int[Array, " 2"]
        Number of points in each dimension. Can be a scalar (square grid)
        or array of shape (2,) with [num_points_x, num_points_y] for
        rectangular grids.

    Returns
    -------
    xx : Float[Array, " hh ww"]
        X coordinate grid in meters.
    yy : Float[Array, " hh ww"]
        Y coordinate grid in meters.

    Notes
    -----
    - Create a linear space of points along the x-axis.
    - Create a linear space of points along the y-axis.
    - Create a meshgrid of spatial coordinates.
    - Return the meshgrid.
    - Supports both square and non-square grids without if-else statements.

    Examples
    --------
    Square grid:

    >>> xx, yy = create_spatial_grid(1e-3, 256)

    Rectangular grid:

    >>> grid_size = jnp.asarray((256, 512), dtype=jnp.int32)
    >>> xx, yy = create_spatial_grid(jnp.array([1e-3, 2e-3]), grid_size)
    """
    # Convert to arrays - scalars become (1,), arrays of shape (2,) stay (2,)
    diameter_arr: Num[Array, "..."] = jnp.atleast_1d(diameter)
    num_points_arr: Int[Array, "..."] = jnp.atleast_1d(num_points)

    # Pad scalar inputs to shape (2,) by repeating the value
    # If already shape (2,), take first 2 elements; if shape (1,), repeat
    diameter_arr = jnp.broadcast_to(diameter_arr, (2,))
    num_points_arr = jnp.broadcast_to(num_points_arr, (2,))

    diameter_x: Num[Array, " "] = diameter_arr[0]
    diameter_y: Num[Array, " "] = diameter_arr[1]
    num_points_x: Int[Array, " "] = num_points_arr[0]
    num_points_y: Int[Array, " "] = num_points_arr[1]

    x: Float[Array, " ww"] = jnp.linspace(
        -diameter_x / 2, diameter_x / 2, num_points_x
    )
    y: Float[Array, " hh"] = jnp.linspace(
        -diameter_y / 2, diameter_y / 2, num_points_y
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = jnp.meshgrid(x, y)
    return (xx, yy)


@jaxtyped(typechecker=beartype)
def normalize_field(
    field: Complex[Array, " hh ww"],
) -> Complex[Array, " hh ww"]:
    """
    Normalize complex field to unit power.

    Parameters
    ----------
    field : Complex[Array, " hh ww"]
        Input complex field.

    Returns
    -------
    normalized_field : Complex[Array, " hh ww"]
        Normalized complex field.

    Notes
    -----
    - Calculate the power of the field as the sum of the square of
        the absolute value of the field.
    - Normalize the field by dividing by the square root of the power.
    - Return the normalized field.
    """
    power: Float[Array, " "] = jnp.sum(jnp.abs(field) ** 2)
    normalized_field: Complex[Array, " hh ww"] = field / jnp.sqrt(power)
    return normalized_field


@jaxtyped(typechecker=beartype)
def add_phase_screen(
    field: Num[Array, " hh ww"],
    phase: Float[Array, " hh ww"],
) -> Complex[Array, " H W"]:
    """
    Add a phase screen to a complex field.

    Parameters
    ----------
    field : Num[Array, " hh ww"]
        Input complex field.
    phase : Float[Array, " hh ww"]
        Phase screen to add.

    Returns
    -------
    screened_field : Complex[Array, " hh ww"]
        Field with phase screen added.

    Notes
    -----
    - Multiply the input field by the exponential of the phase screen.
    - Return the screened field.
    """
    screened_field: Complex[Array, " hh ww"] = field * jnp.exp(1j * phase)
    return screened_field


@jaxtyped(typechecker=beartype)
def field_intensity(field: Complex[Array, " hh ww"]) -> Float[Array, " hh ww"]:
    """
    Calculate intensity from complex field.

    Parameters
    ----------
    field : Complex[Array, " hh ww"]
        Input complex field.

    Returns
    -------
    intensity : Float[Array, " hh ww"]
        Intensity of the field.

    Notes
    -----
    - Calculate the intensity as the square of the absolute value of the
    field.
    - Return the intensity.
    """
    intensity: Float[Array, " hh ww"] = jnp.power(jnp.abs(field), 2)
    return intensity


@jaxtyped(typechecker=beartype)
def scale_pixel(
    wavefront: OpticalWavefront,
    new_dx: ScalarFloat,
) -> OpticalWavefront:
    """
    Rescale OpticalWavefront pixel size while keeping array shape fixed.

    JAX-compatible (jit/vmap-safe). Crops or pads to preserve shape.

    Parameters
    ----------
    wavefront : OpticalWavefront
        OpticalWavefront to be resized.
    new_dx : ScalarFloat
        New pixel size (meters).

    Returns
    -------
    scaled_wavefront : OpticalWavefront
        Resized OpticalWavefront with updated pixel size
        and resized field, which is of the same size as
        the original field.

    Notes
    -----
    - If the new pixel size is smaller than the old one,
      then the new FOV is smaller too at the same field
      size. So we will first find the new smaller FOV,
      and crop to that size with the current pixel size.
      Then we will resize to the new pizel size with the
      cropped FOV so that the size of the field remains
      the same.
      So here the order is crop, then resize.
    - If the new pixel size is larger than the old one,
      then the new FOV of the final field is larger too
    - Return the resized OpticalWavefront.
    """
    field: Complex[Array, " hh ww"] = wavefront.field
    old_dx: ScalarFloat = wavefront.dx
    hh: int
    ww: int
    hh, ww = field.shape
    scale: ScalarFloat = new_dx / old_dx
    current_fov_h: ScalarFloat = hh * old_dx
    current_fov_w: ScalarFloat = ww * old_dx
    new_fov_h: ScalarFloat = hh * new_dx
    new_fov_w: ScalarFloat = ww * new_dx

    def _smaller_pixel_size(
        field: Complex[Array, " hh ww"],
    ) -> Complex[Array, " hh ww"]:
        """
        If the new pixel size is smaller than the old one.

        Then the new FOV is smaller too at the same field
        size. So we will first find the new smaller FOV,
        and crop to that size with the current pixel size.
        Then we will resize to the new pizel size with the
        cropped FOV so that the size of the field remains
        the same.
        So here the order is crop, then resize.
        """
        new_h: Int[Array, " "] = jnp.floor(new_fov_h / old_dx).astype(int)
        new_w: Int[Array, " "] = jnp.floor(new_fov_w / old_dx).astype(int)
        start_h: Int[Array, " "] = jnp.floor(
            (current_fov_h - new_fov_h) / (2 * old_dx)
        ).astype(int)
        start_w: Int[Array, " "] = jnp.floor(
            (current_fov_w - new_fov_w) / (2 * old_dx)
        ).astype(int)
        cropped: Complex[Array, " new_h new_w"] = jax.lax.dynamic_slice(
            field, (start_h, start_w), (new_h, new_w)
        )
        resized: Complex[Array, " hh ww"] = jax.image.resize(
            cropped,
            (hh, ww),
            method="linear",
            antialias=True,
        )
        return resized

    def _larger_pixel_size(
        field: Complex[Array, " hh ww"],
    ) -> Complex[Array, " hh ww"]:
        """
        If the new pixel size is larger than the old one.

        Then the new FOV of the final field is larger too
        at the same field size. So we will need to first
        get the current FOV data with the new pixel size,
        which will be smaller than the current field size.
        Following this, we need to pad out to fill the
        field.
        So here the order is resize then pad.
        """
        data_minima_h: Float[Array, " "] = jnp.min(jnp.abs(field))
        new_h: Int[Array, " "] = jnp.floor(current_fov_h / new_dx).astype(int)
        new_w: Int[Array, " "] = jnp.floor(current_fov_w / new_dx).astype(int)
        resized: Complex[Array, " H W"] = jax.image.resize(
            field,
            (new_h, new_w),
            method="linear",
            antialias=True,
        )
        pad_h_0: Int[Array, " "] = jnp.floor((hh - new_h) / 2).astype(int)
        pad_h_1: Int[Array, " "] = hh - (new_h + pad_h_0)
        pad_w_0: Int[Array, " "] = jnp.floor((ww - new_w) / 2).astype(int)
        pad_w_1: Int[Array, " "] = ww - (new_w + pad_w_0)
        return jnp.pad(
            resized,
            ((pad_h_0, pad_h_1), (pad_w_0, pad_w_1)),
            mode="constant",
            constant_values=data_minima_h,
        )

    resized_field = jax.lax.cond(
        scale > 1.0, _larger_pixel_size, _smaller_pixel_size, field
    )
    scaled_wavefront: OpticalWavefront = make_optical_wavefront(
        field=resized_field,
        dx=new_dx,
        wavelength=wavefront.wavelength,
        z_position=wavefront.z_position,
    )
    return scaled_wavefront
