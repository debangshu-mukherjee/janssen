"""Aperture functions for optical simulations.

Extended Summary
----------------
Optical aperture and apodization functions for controlling the amplitude
and phase of optical wavefronts. Includes both hard apertures and smooth
apodization functions commonly used in optical systems.

Routine Listings
----------------
circular_aperture : function
    Applies a circular aperture (optionally offset) with uniform
    transmittivity.
rectangular_aperture : function
    Applies an axis-aligned rectangular aperture with uniform
    transmittivity.
annular_aperture : function
    Applies a concentric ring (donut) aperture between inner/outer
    diameters.
variable_transmission_aperture : function
    Applies an arbitrary transmission mask (array or callable),
    including common apodizers such as Gaussian or super-Gaussian
gaussian_apodizer : function
    Applies a Gaussian apodizer (smooth transmission mask) to the
    wavefront.
supergaussian_apodizer : function
    Applies a super-Gaussian apodizer (smooth transmission mask) to
    wavefront.
gaussian_apodizer_elliptical : function
    Applies an elliptical Gaussian apodizer to the wavefront
supergaussian_apodizer_elliptical : function
    Applies an elliptical super-Gaussian apodizer to the wavefront
_arrayed_grids : function, internal
    Creates coordinate grids without array creation.

Notes
-----
All aperture functions are compatible with JAX transformations and
support automatic differentiation. The apertures can be combined to
create complex pupil functions for optical systems.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Bool, Float, Num, jaxtyped

from janssen.utils import (
    OpticalWavefront,
    ScalarFloat,
    ScalarNumeric,
    make_optical_wavefront,
)


def _arrayed_grids(
    x0: Num[Array, " hh ww"],
    y0: Num[Array, " hh ww"],
    dx: Union[ScalarFloat, Num[Array, " 2"]],
) -> Tuple[Float[Array, " hh ww"], Float[Array, " hh ww"]]:
    """Create coordinate grids without array creation.

    Parameters
    ----------
    x0 : Num[Array, " hh ww"]
        Zero-valued input array for x coordinates.
    y0 : Num[Array, " hh ww"]
        Zero-valued input array for y coordinates.
    dx : Union[ScalarFloat, Num[Array, " 2"]]
        Grid spacing in meters. Can be scalar or 2-element array [dx, dy].

    Returns
    -------
    xx : Float[Array, " hh ww"]
        X coordinate grid in meters.
    yy : Float[Array, " hh ww"]
        Y coordinate grid in meters.
    """
    hh: int
    ww: int
    hh, ww = x0.shape
    dx_arr: Union[Num[Array, " "], Num[Array, " 2"]] = jnp.asarray(dx)
    dx_arr = jnp.atleast_1d(dx_arr)
    expected_ndim: int = 2
    dx_2elem = jnp.where(
        dx_arr.size >= expected_ndim,
        dx_arr[:2],
        jnp.array([dx_arr[0], dx_arr[0]]),
    )
    dx_val: Num[Array, " "] = dx_2elem[0]
    dy_val: Num[Array, " "] = dx_2elem[1]

    def x_line(
        arr: Num[Array, " hh ww"], spacing: Num[Array, " "]
    ) -> Num[Array, " hh ww"]:
        arr_x: Num[Array, " ww"] = jnp.arange(-ww // 2, ww // 2) * spacing
        arr_full: Num[Array, " hh ww"] = arr + jnp.repeat(arr_x, hh).reshape(
            hh, ww, order="F"
        )
        return arr_full

    def y_line(
        arr: Num[Array, " hh ww"], spacing: Num[Array, " "]
    ) -> Num[Array, " hh ww"]:
        arr_y: Num[Array, " hh"] = jnp.arange(-hh // 2, hh // 2) * spacing
        arr_full: Num[Array, " hh ww"] = arr + jnp.repeat(arr_y, ww).reshape(
            hh, ww, order="C"
        )
        return arr_full

    xx: Num[Array, " hh ww"] = x_line(x0, dx_val)
    yy: Num[Array, " hh ww"] = y_line(y0, dy_val)
    return (xx, yy)


@jaxtyped(typechecker=beartype)
def circular_aperture(
    incoming: OpticalWavefront,
    diameter: ScalarFloat,
    center: Union[ScalarFloat, Float[Array, " 2"]] = 0.0,
    transmittivity: Optional[ScalarFloat] = 1.0,
) -> OpticalWavefront:
    """
    Apply a circular aperture to the incoming wavefront.

    The aperture is defined by its physical diameter and (optional)
    center.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront PyTree.
    diameter : ScalarFloat
        Aperture diameter in meters.
    center : Float[Array, " 2"], optional
        Physical center [x0, y0] of the aperture in meters, by default
        [0, 0].
    transmittivity : Optional[ScalarFloat], optional
        Uniform transmittivity inside the aperture (0..1), by default
        1.0.

    Returns
    -------
    apertured : OpticalWavefront
        Wavefront after applying the circular aperture.

    Notes
    -----
    - Build centered (x, y) grids in meters.
    - Compute radial distance from the specified center.
    - Create a binary mask for r <= diameter/2.
    - Multiply by transmittivity (clipped to [0, 1]).
    - Apply to the complex field and return.
    """
    center_array: Float[Array, " 2"] = jnp.atleast_2d(
        jnp.asarray(center, dtype=jnp.float64)
    ).ravel()[:2]
    arr_zeros: Float[Array, " hh ww"] = jnp.zeros_like(
        incoming.field, dtype=float
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = _arrayed_grids(arr_zeros, arr_zeros, incoming.dx)
    x0: Float[Array, " "]
    y0: Float[Array, " "]
    x0, y0 = center_array[0], center_array[1]
    r: Float[Array, " hh ww"] = jnp.sqrt(((xx - x0) ** 2) + ((yy - y0) ** 2))
    inside: Bool[Array, " hh ww"] = r <= (diameter / 2.0)
    t: Float[Array, " "] = jnp.clip(
        jnp.asarray(transmittivity, dtype=float), 0.0, 1.0
    )
    transmission: Float[Array, " hh ww"] = inside.astype(float) * t
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * transmission,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured


@jaxtyped(typechecker=beartype)
def rectangular_aperture(
    incoming: OpticalWavefront,
    width: ScalarFloat,
    height: ScalarFloat,
    center: Union[ScalarFloat, Float[Array, " 2"]] = 0.0,
    transmittivity: Optional[ScalarFloat] = 1.0,
) -> OpticalWavefront:
    """
    Apply an axis-aligned rectangular aperture to the incoming
    wavefront.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront PyTree.
    width : ScalarFloat
        Rectangle width along x in meters.
    height : ScalarFloat
        Rectangle height along y in meters.
    center : Float[Array, " 2"], optional
        Rectangle center [x0, y0] in meters, by default [0, 0].
    transmittivity : Optional[ScalarFloat], optional
        Uniform transmittivity inside the rectangle (0..1), by default
        1.0.

    Returns
    -------
    apertured : OpticalWavefront
        Wavefront after applying the rectangular aperture.

    Notes
    -----
    - Build centered (x, y) grids in meters.
    - Compute half-width/half-height and an inside-rectangle mask.
    - Multiply by transmittivity (clipped).
    - Apply to the complex field and return.
    """
    center_array: Float[Array, " 2"] = jnp.atleast_2d(
        jnp.asarray(center, dtype=jnp.float64)
    ).ravel()[:2]
    arr_zeros: Float[Array, " hh ww"] = jnp.zeros_like(
        incoming.field, dtype=float
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = _arrayed_grids(arr_zeros, arr_zeros, incoming.dx)
    x0: Float[Array, " "]
    y0: Float[Array, " "]
    x0, y0 = center_array[0], center_array[1]
    hx: Float[Array, " "] = width / 2.0
    hy: Float[Array, " "] = height / 2.0
    inside_x: Bool[Array, " hh ww"] = ((x0 - hx) <= xx) & ((x0 + hx) >= xx)
    inside_y: Bool[Array, " hh ww"] = ((y0 - hy) <= yy) & ((y0 + hy) >= yy)
    inside: Bool[Array, " hh ww"] = inside_x & inside_y
    t: Float[Array, " "] = jnp.clip(
        jnp.asarray(transmittivity, dtype=float), 0.0, 1.0
    )
    transmission: Float[Array, " hh ww"] = inside.astype(float) * t
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * transmission,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured


@jaxtyped(typechecker=beartype)
def annular_aperture(
    incoming: OpticalWavefront,
    inner_diameter: ScalarFloat,
    outer_diameter: ScalarFloat,
    center: Union[ScalarFloat, Float[Array, " 2"]] = 0.0,
    transmittivity: Optional[ScalarFloat] = 1.0,
) -> OpticalWavefront:
    """
    Apply an annular (ring) aperture with inner and outer diameters.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront PyTree.
    inner_diameter : ScalarFloat
        Inner blocked diameter in meters.
    outer_diameter : ScalarFloat
        Outer clear aperture diameter in meters.
    center : Float[Array, " 2"], optional
        Ring center [x0, y0] in meters, by default [0, 0].
    transmittivity : Optional[ScalarFloat], optional
        Uniform transmittivity in the ring (0..1), by default 1.0.

    Returns
    -------
    apertured : OpticalWavefront
        Wavefront after applying the annular aperture.

    Notes
    -----
    - Build centered (x, y) grids in meters.
    - Compute radial distance from center.
    - Create mask for inner_radius < r <= outer_radius.
    - Multiply by transmittivity (clipped), apply, and return.
    """
    center_array: Float[Array, " 2"] = jnp.atleast_2d(
        jnp.asarray(center, dtype=jnp.float64)
    ).ravel()[:2]
    arr_zeros: Float[Array, " hh ww"] = jnp.zeros_like(
        incoming.field, dtype=float
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = _arrayed_grids(arr_zeros, arr_zeros, incoming.dx)
    x0: Float[Array, " "]
    y0: Float[Array, " "]
    x0, y0 = center_array[0], center_array[1]
    r: Float[Array, " hh ww"] = jnp.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    r_in: Float[Array, " "] = inner_diameter / 2.0
    r_out: Float[Array, " "] = outer_diameter / 2.0
    ring: Bool[Array, " hh ww"] = (r > r_in) & (r <= r_out)
    t: Float[Array, " "] = jnp.clip(
        jnp.asarray(transmittivity, dtype=float), 0.0, 1.0
    )
    transmission: Float[Array, " hh ww"] = ring.astype(float) * t
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * transmission,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured


@jaxtyped(typechecker=beartype)
def variable_transmission_aperture(
    incoming: OpticalWavefront,
    transmission: Union[ScalarFloat, Float[Array, " ..."]],
) -> OpticalWavefront:
    """
    Apply an arbitrary (spatially varying) transmission to the
    wavefront.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront PyTree.
    transmission : Union[ScalarFloat, Float[Array, " H W"]]
        Precomputed transmission map (0..1) with shape "H W", or a
        scalar attenuation factor for uniform transmission.

    Returns
    -------
    transmitted : OpticalWavefront
        Wavefront after applying the transmission.

    Examples
    --------
    Uniform attenuation::

        >>> wf2 = variable_transmission_aperture(wf, 0.5)  # 50% trans

    Spatially varying transmission::

        >>> tmap = create_transmission_map(...)  # Shape (H, W)
        >>> wf2 = variable_transmission_aperture(wf, tmap)

    Notes
    -----
    - For scalar transmission: applies uniform attenuation.
    - For array transmission: applies spatially varying transmission
      map.
    - Transmission values are clipped to [0, 1].
    - This function is fully JAX-compatible and uses jax.lax.cond.
    """
    trans: Float[Array, " ..."] = jnp.asarray(transmission, dtype=float)

    def apply_scalar_transmission() -> OpticalWavefront:
        t: Float[Array, " hh ww"] = jnp.clip(trans, 0.0, 1.0)
        return make_optical_wavefront(
            field=incoming.field * t,
            wavelength=incoming.wavelength,
            dx=incoming.dx,
            z_position=incoming.z_position,
        )

    def apply_array_transmission() -> OpticalWavefront:
        tmap: Float[Array, " hh ww"] = jnp.clip(trans, 0.0, 1.0)
        return make_optical_wavefront(
            field=incoming.field * tmap,
            wavelength=incoming.wavelength,
            dx=incoming.dx,
            z_position=incoming.z_position,
        )

    transmitted: OpticalWavefront = jax.lax.cond(
        trans.ndim == 0, apply_scalar_transmission, apply_array_transmission
    )
    return transmitted


@jaxtyped(typechecker=beartype)
def gaussian_apodizer(
    incoming: OpticalWavefront,
    sigma: ScalarFloat,
    center: Union[ScalarFloat, Float[Array, " 2"]] = 0.0,
    peak_transmittivity: Optional[ScalarFloat] = 1.0,
) -> OpticalWavefront:
    """
    Apply a Gaussian apodizer (smooth transmission mask) to the
    wavefront.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input optical wavefront.
    sigma : ScalarFloat
        Gaussian width parameter in meters.
    center : Float[Array, " 2"], optional
        Physical center [x0, y0] of the Gaussian in meters, by default
        [0, 0].
    peak_transmittivity : Optional[ScalarFloat], optional
        Maximum transmission at the Gaussian center, by default 1.0.

    Returns
    -------
    apertured : OpticalWavefront
        Wavefront after applying Gaussian apodization.

    Notes
    -----
    - Build centered (x, y) grids.
    - Compute squared radial distance from center.
    - Evaluate Gaussian exp(-r^2 / (2*sigma^2)).
    - Scale by peak transmittivity, clip to [0,1].
    - Multiply with incoming field and return.
    """
    center_array: Float[Array, " 2"] = jnp.atleast_2d(
        jnp.asarray(center, dtype=jnp.float64)
    ).ravel()[:2]
    arr_zeros: Float[Array, " hh ww"] = jnp.zeros_like(
        incoming.field, dtype=float
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = _arrayed_grids(arr_zeros, arr_zeros, incoming.dx)
    x0: Float[Array, " "]
    y0: Float[Array, " "]
    x0, y0 = center_array[0], center_array[1]
    r2: Float[Array, " hh ww"] = ((xx - x0) ** 2) + ((yy - y0) ** 2)
    gauss: Float[Array, " hh ww"] = jnp.exp(-r2 / (2.0 * sigma**2))
    tmap: Float[Array, " hh ww"] = jnp.clip(
        gauss * peak_transmittivity, 0.0, 1.0
    )
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * tmap,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured


@jaxtyped(typechecker=beartype)
def supergaussian_apodizer(
    incoming: OpticalWavefront,
    sigma: ScalarFloat,
    m: ScalarNumeric,
    center: Union[ScalarFloat, Float[Array, " 2"]] = 0.0,
    peak_transmittivity: Optional[ScalarFloat] = 1.0,
) -> OpticalWavefront:
    """
    Apply a super-Gaussian apodizer to the wavefront.

    Transmission profile: exp(- (r^2 / sigma^2)^m ).

    Parameters
    ----------
    incoming : OpticalWavefront
        Input optical wavefront.
    sigma : ScalarFloat
        Width parameter in meters (sets the roll-off scale).
    m : ScalarNumeric
        Super-Gaussian order (m=1 → Gaussian, m>1 → flatter top).
    center : Float[Array, " 2"], optional
        Physical center [x0, y0] of the profile, by default [0, 0].
    peak_transmittivity : Optional[ScalarFloat], optional
        Maximum transmission at the center, by default 1.0.

    Returns
    -------
    apertured : OpticalWavefront
        Wavefront after applying super-Gaussian apodization.

    Notes
    -----
    - Build centered (x, y) grids.
    - Compute squared radial distance from center.
    - Evaluate exp(- (r^2 / sigma^2)^m ).
    - Scale by peak transmittivity, clip to [0,1].
    - Multiply with incoming field and return.
    """
    arr_zeros: Float[Array, " hh ww"] = jnp.zeros_like(
        incoming.field, dtype=float
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = _arrayed_grids(arr_zeros, arr_zeros, incoming.dx)
    x0, y0 = center[0], center[1]
    r2: Float[Array, " hh ww"] = (xx - x0) ** 2 + (yy - y0) ** 2
    super_gauss: Float[Array, " hh ww"] = jnp.exp(-((r2 / (sigma**2)) ** m))
    tmap: Float[Array, " hh ww"] = jnp.clip(
        super_gauss * peak_transmittivity, 0.0, 1.0
    )
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * tmap,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured


@jaxtyped(typechecker=beartype)
def gaussian_apodizer_elliptical(
    incoming: OpticalWavefront,
    sigma_x: ScalarFloat,
    sigma_y: ScalarFloat,
    theta: Optional[ScalarFloat] = 0.0,
    center: Union[ScalarFloat, Float[Array, " 2"]] = 0.0,
    peak_transmittivity: Optional[ScalarFloat] = 1.0,
) -> OpticalWavefront:
    """
    Apply an elliptical Gaussian apodizer to the wavefront.

    With optional rotation, through an angle `theta`.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input optical wavefront.
    sigma_x : ScalarFloat
        Gaussian width along the x'-axis (meters) after rotation by
        `theta`.
    sigma_y : ScalarFloat
        Gaussian width along the y'-axis (meters) after rotation by
        `theta`.
    theta : Optional[ScalarFloat], optional
        Rotation angle in radians (counter-clockwise), by default
        0.0.
    center : Float[Array, " 2"], optional
        Physical center [x0, y0] in meters, by default [0, 0].
    peak_transmittivity : Optional[ScalarFloat], optional
        Maximum transmission at the center, by default 1.0.

    Returns
    -------
    apertured : OpticalWavefront
        Wavefront after applying elliptical Gaussian apodization.

    See Also
    --------
    gaussian_apodizer : Apply a Gaussian apodizer (smooth transmission
        mask) to the wavefront.
    supergaussian_apodizer : Apply a super-Gaussian apodizer (smooth
        transmission mask) to the wavefront.

    Notes
    -----
    - Build centered (x, y) grids.
    - Translate by `center`, rotate by `theta` → (x', y').
    - Evaluate exp(-0.5 * ( (x'/sigma_x)^2 + (y'/sigma_y)^2 )).
    - Scale by `peak_transmittivity`, clip to [0, 1].
    - Multiply with incoming field and return.
    """
    center_array: Float[Array, " 2"] = jnp.atleast_2d(
        jnp.asarray(center, dtype=jnp.float64)
    ).ravel()[:2]
    arr_zeros: Float[Array, " hh ww"] = jnp.zeros_like(
        incoming.field, dtype=float
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = _arrayed_grids(arr_zeros, arr_zeros, incoming.dx)
    x0: Float[Array, " "]
    y0: Float[Array, " "]
    x0, y0 = center_array[0], center_array[1]
    xc: Float[Array, " hh ww"] = xx - x0
    yc: Float[Array, " hh ww"] = yy - y0
    ct: Float[Array, " "] = jnp.cos(theta)
    st: Float[Array, " "] = jnp.sin(theta)
    xp: Float[Array, " hh ww"] = (ct * xc) + (st * yc)
    yp: Float[Array, " hh ww"] = (ct * yc) - (st * xc)
    arg: Float[Array, " hh ww"] = ((xp / sigma_x) ** 2) + ((yp / sigma_y) ** 2)
    gauss: Float[Array, " hh ww"] = jnp.exp(-0.5 * arg)
    tmap: Float[Array, " hh ww"] = jnp.clip(
        gauss * peak_transmittivity, 0.0, 1.0
    )
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * tmap,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured


@jaxtyped(typechecker=beartype)
def supergaussian_apodizer_elliptical(
    incoming: OpticalWavefront,
    sigma_x: ScalarFloat,
    sigma_y: ScalarFloat,
    m: ScalarNumeric,
    theta: Optional[ScalarFloat] = 0.0,
    center: Union[ScalarFloat, Float[Array, " 2"]] = 0.0,
    peak_transmittivity: Optional[ScalarFloat] = 1.0,
) -> OpticalWavefront:
    """
    Apply an elliptical super-Gaussian apodizer with optional
    rotation.

    Transmission profile:
    exp( - ( (x'/sigma_x)^2 + (y'/sigma_y)^2 )^m ).

    Parameters
    ----------
    incoming : OpticalWavefront
        Input optical wavefront.
    sigma_x : ScalarFloat
        Width along x' (meters) after rotation by `theta`.
    sigma_y : ScalarFloat
        Width along y' (meters) after rotation by `theta`.
    m : ScalarNumeric
        Super-Gaussian order
        (m=1 → Gaussian; m>1 → flatter top, sharper edges).
    theta : Optional[ScalarFloat], optional
        Rotation angle in radians (counter-clockwise), by default
        0.0.
    center : Float[Array, " 2"], optional
        Physical center [x0, y0] in meters, by default [0, 0].
    peak_transmittivity : Optional[ScalarFloat], optional
        Maximum transmission at the center, by default 1.0.

    Returns
    -------
    apertured : OpticalWavefront
        Wavefront after applying elliptical super-Gaussian apodization.

    Notes
    -----
    - Build centered (x, y) grids.
    - Translate by `center`, rotate by `theta` → (x', y').
    - Evaluate exp( - ( (x'/sigma_x)^2 + (y'/sigma_y)^2 )^m ).
    - Scale by `peak_transmittivity`, clip to [0, 1].
    - Multiply with incoming field and return.
    """
    center_array: Float[Array, " 2"] = jnp.atleast_2d(
        jnp.asarray(center, dtype=jnp.float64)
    ).ravel()[:2]
    arr_zeros: Float[Array, " hh ww"] = jnp.zeros_like(
        incoming.field, dtype=float
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = _arrayed_grids(arr_zeros, arr_zeros, incoming.dx)
    x0: Float[Array, " "]
    y0: Float[Array, " "]
    x0, y0 = center_array[0], center_array[1]
    xc: Float[Array, " hh ww"] = xx - x0
    yc: Float[Array, " hh ww"] = yy - y0
    ct: Float[Array, " "] = jnp.cos(theta)
    st: Float[Array, " "] = jnp.sin(theta)
    xp: Float[Array, " hh ww"] = (ct * xc) + (st * yc)
    yp: Float[Array, " hh ww"] = (ct * yc) - (st * xc)
    base: Float[Array, " hh ww"] = ((xp / sigma_x) ** 2) + (
        (yp / sigma_y) ** 2
    )
    super_gauss: Float[Array, " hh ww"] = jnp.exp(-(base**m))
    tmap: Float[Array, " hh ww"] = jnp.clip(
        super_gauss * peak_transmittivity, 0.0, 1.0
    )
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * tmap,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured
