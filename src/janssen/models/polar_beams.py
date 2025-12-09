"""Polarized beam generators for vector optics simulations.

Extended Summary
----------------
This module provides factory functions for generating polarized optical
beams, including cylindrical vector beams (radial and azimuthal polarization)
and standard linear polarization states. These beam types are essential for
high-NA focusing simulations where vector effects become significant.

Cylindrical vector beams exhibit unique focusing properties:
- Radially polarized beams create strong longitudinal Ez fields at focus
- Azimuthally polarized beams create "donut" intensity profiles with no Ez
- These effects are invisible to scalar diffraction theory

Routine Listings
----------------
radially_polarized_beam : function
    Generate a radially polarized beam (E-field points radially outward)
azimuthally_polarized_beam : function
    Generate an azimuthally polarized beam (E-field circulates azimuthally)
linear_polarized_beam : function
    Generate a linearly polarized beam with arbitrary angle
x_polarized_beam : function
    Generate an x-polarized beam (convenience wrapper)
y_polarized_beam : function
    Generate a y-polarized beam (convenience wrapper)
circular_polarized_beam : function
    Generate a circularly polarized beam (left or right handed)
generalized_cylindrical_vector_beam : function
    Generate a generalized cylindrical vector beam of arbitrary order

Notes
-----
All beam generators follow Janssen's conventions:
- Pure JAX functions supporting jit, grad, vmap
- jaxtyping + beartype for runtime type checking
- Returns OpticalWavefront PyTree with (H, W, 2) polarized field

For high-NA focusing, these beams should be passed through the
Richards-Wolf focusing function which will compute the full 3D
vector field including Ez.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple, Union
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from janssen.optics import create_spatial_grid
from janssen.utils import (
    OpticalWavefront,
    ScalarFloat,
    ScalarInteger,
    make_optical_wavefront,
)

jax.config.update("jax_enable_x64", True)

BESSEL_SAFE_FLOOR: float = 1e-10


@jaxtyped(typechecker=beartype)
def _create_grid_and_polar_coords(
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
) -> Tuple[
    Float[Array, " ny nx"],
    Float[Array, " ny nx"],
    Float[Array, " ny nx"],
    Float[Array, " ny nx"],
]:
    """Create Cartesian and polar coordinate grids.

    Parameters
    ----------
    dx : ScalarFloat
        Spatial sampling interval in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Grid size as (ny, nx) or (height, width).

    Returns
    -------
    xx : Float[Array, " ny nx"]
        X coordinate grid (meters).
    yy : Float[Array, " ny nx"]
        Y coordinate grid (meters).
    rr : Float[Array, " ny nx"]
        Radial distance from center (meters).
    phi : Float[Array, " ny nx"]
        Azimuthal angle (radians), measured CCW from +x axis.
    """
    grid_size_arr = jnp.asarray(grid_size, dtype=jnp.int32)
    ny: ScalarInteger = grid_size_arr[0]
    nx: ScalarInteger = grid_size_arr[1]
    diameter: Float[Array, " 2"] = jnp.asarray(
        [nx * dx, ny * dx], dtype=jnp.float64
    )
    num_points: Int[Array, " 2"] = jnp.asarray([nx, ny], dtype=jnp.int32)

    xx: Float[Array, " ny nx"]
    yy: Float[Array, " ny nx"]
    xx, yy = create_spatial_grid(diameter, num_points)

    rr: Float[Array, " ny nx"] = jnp.sqrt(xx**2 + yy**2)
    phi: Float[Array, " ny nx"] = jnp.arctan2(yy, xx)

    return xx, yy, rr, phi


@jaxtyped(typechecker=beartype)
def radially_polarized_beam(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    beam_radius: ScalarFloat = None,
    amplitude: ScalarFloat = 1.0,
    z_position: ScalarFloat = 0.0,
    apodization: str = "gaussian",
) -> OpticalWavefront:
    r"""Generate a radially polarized beam.

    Creates a cylindrical vector beam where the electric field points
    radially outward from the optical axis at every point. The polarization
    direction is:

    .. math::
        \hat{e}_r = \cos(\phi)\hat{x} + \sin(\phi)\hat{y}

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid as (height, width) or (ny, nx).
    beam_radius : ScalarFloat, optional
        Characteristic beam radius in meters. If None, defaults to
        1/4 of the grid extent. For Gaussian apodization, this is the
        1/e² intensity radius.
    amplitude : ScalarFloat, optional
        Peak amplitude at beam edge, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position of the wavefront in meters, by default 0.0.
    apodization : str, optional
        Amplitude envelope type. Options are:
        - "gaussian" : Gaussian envelope exp(-r²/w²) (default)
        - "uniform" : Uniform amplitude within beam_radius
        - "bessel" : J₁(r) profile (natural for radial pol.)

    Returns
    -------
    OpticalWavefront
        Radially polarized beam as OpticalWavefront PyTree.
        Field shape is (H, W, 2) with [Ex, Ey] components.

    Notes
    -----
    Radially polarized beams have several unique properties:

    1. **Tight focusing**: When focused by a high-NA lens, radially
       polarized beams produce a strong longitudinal (Ez) field component
       at the focus, resulting in a tighter focal spot than linearly
       polarized beams.

    2. **Singular at center**: The polarization is undefined at r=0
       (the optical axis), creating a phase singularity. The amplitude
       naturally goes to zero at the center.

    3. **Zero orbital angular momentum**: Unlike Laguerre-Gaussian vortex
       beams, radially polarized beams carry no orbital angular momentum.

    The field components are:

    .. math::
        E_x(r, \phi) = A(r) \cos(\phi)

        E_y(r, \phi) = A(r) \sin(\phi)

    where A(r) is the amplitude envelope (Gaussian, uniform, or Bessel).

    References
    ----------
    .. [1] Dorn, R., Quabis, S., & Leuchs, G. (2003). "Sharper focus for a
           radially polarized light beam". Physical Review Letters, 91(23),
           233901.
    """
    _, _, rr, phi = _create_grid_and_polar_coords(dx, grid_size)

    grid_size_arr = jnp.asarray(grid_size, dtype=jnp.int32)
    grid_extent: Float[Array, " "] = jnp.minimum(
        grid_size_arr[0], grid_size_arr[1]
    ) * jnp.asarray(dx, dtype=jnp.float64)

    w: Float[Array, " "] = jax.lax.cond(
        beam_radius is None,
        lambda: grid_extent / 4.0,
        lambda: jnp.asarray(beam_radius, dtype=jnp.float64),
    )

    if apodization == "gaussian":
        amplitude_envelope: Float[Array, " ny nx"] = jnp.exp(
            -(rr**2) / (w**2)
        )
    elif apodization == "uniform":
        amplitude_envelope = jnp.where(rr <= w, 1.0, 0.0)
    elif apodization == "bessel":
        kr: Float[Array, " ny nx"] = 2.405 * rr / w
        safe_kr: Float[Array, " ny nx"] = jnp.where(
            kr < BESSEL_SAFE_FLOOR, BESSEL_SAFE_FLOOR, kr
        )
        amplitude_envelope = jnp.abs(jax.scipy.special.bessel_jn(1, safe_kr))
    else:
        amplitude_envelope = jnp.exp(-(rr**2) / (w**2))

    amp: Float[Array, " "] = jnp.asarray(amplitude, dtype=jnp.float64)
    scaled_envelope: Float[Array, " ny nx"] = amp * amplitude_envelope

    ex: Complex[Array, " ny nx"] = (
        scaled_envelope * jnp.cos(phi)
    ).astype(jnp.complex128)
    ey: Complex[Array, " ny nx"] = (
        scaled_envelope * jnp.sin(phi)
    ).astype(jnp.complex128)

    field: Complex[Array, " ny nx 2"] = jnp.stack([ex, ey], axis=-1)

    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
        polarization=True,
    )
    return wavefront


@jaxtyped(typechecker=beartype)
def azimuthally_polarized_beam(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    beam_radius: ScalarFloat = None,
    amplitude: ScalarFloat = 1.0,
    z_position: ScalarFloat = 0.0,
    apodization: str = "gaussian",
) -> OpticalWavefront:
    r"""Generate an azimuthally polarized beam.

    Creates a cylindrical vector beam where the electric field points
    in the azimuthal direction (tangent to circles centered on the axis).
    The polarization direction is:

    .. math::
        \hat{e}_\phi = -\sin(\phi)\hat{x} + \cos(\phi)\hat{y}

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid as (height, width) or (ny, nx).
    beam_radius : ScalarFloat, optional
        Characteristic beam radius in meters. If None, defaults to
        1/4 of the grid extent.
    amplitude : ScalarFloat, optional
        Peak amplitude, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position of the wavefront in meters, by default 0.0.
    apodization : str, optional
        Amplitude envelope type: "gaussian", "uniform", or "bessel".

    Returns
    -------
    OpticalWavefront
        Azimuthally polarized beam as OpticalWavefront PyTree.
        Field shape is (H, W, 2) with [Ex, Ey] components.

    Notes
    -----
    Azimuthally polarized beams have complementary properties to radial
    polarization:

    1. **Donut focus**: When focused by a high-NA lens, azimuthally
       polarized beams produce NO longitudinal (Ez) field component.
       The focal spot has a dark center ("donut" shape).

    2. **Singular at center**: Like radial polarization, the polarization
       is undefined at r=0, and the amplitude naturally goes to zero.

    3. **Orthogonal to radial**: At every point, the azimuthal polarization
       is perpendicular to the radial polarization.

    The field components are:

    .. math::
        E_x(r, \phi) = -A(r) \sin(\phi)

        E_y(r, \phi) = A(r) \cos(\phi)

    References
    ----------
    .. [1] Zhan, Q. (2009). "Cylindrical vector beams: from mathematical
           concepts to applications". Advances in Optics and Photonics,
           1(1), 1-57.
    """
    _, _, rr, phi = _create_grid_and_polar_coords(dx, grid_size)

    grid_size_arr = jnp.asarray(grid_size, dtype=jnp.int32)
    grid_extent: Float[Array, " "] = jnp.minimum(
        grid_size_arr[0], grid_size_arr[1]
    ) * jnp.asarray(dx, dtype=jnp.float64)

    w: Float[Array, " "] = jax.lax.cond(
        beam_radius is None,
        lambda: grid_extent / 4.0,
        lambda: jnp.asarray(beam_radius, dtype=jnp.float64),
    )

    if apodization == "gaussian":
        amplitude_envelope: Float[Array, " ny nx"] = jnp.exp(
            -(rr**2) / (w**2)
        )
    elif apodization == "uniform":
        amplitude_envelope = jnp.where(rr <= w, 1.0, 0.0)
    else:
        amplitude_envelope = jnp.exp(-(rr**2) / (w**2))

    amp: Float[Array, " "] = jnp.asarray(amplitude, dtype=jnp.float64)
    scaled_envelope: Float[Array, " ny nx"] = amp * amplitude_envelope

    ex: Complex[Array, " ny nx"] = (
        -scaled_envelope * jnp.sin(phi)
    ).astype(jnp.complex128)
    ey: Complex[Array, " ny nx"] = (
        scaled_envelope * jnp.cos(phi)
    ).astype(jnp.complex128)

    field: Complex[Array, " ny nx 2"] = jnp.stack([ex, ey], axis=-1)

    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
        polarization=True,
    )
    return wavefront


@jaxtyped(typechecker=beartype)
def linear_polarized_beam(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    polarization_angle: ScalarFloat = 0.0,
    beam_radius: ScalarFloat = None,
    amplitude: ScalarFloat = 1.0,
    z_position: ScalarFloat = 0.0,
    apodization: str = "gaussian",
) -> OpticalWavefront:
    r"""Generate a linearly polarized Gaussian beam.

    Creates a beam with uniform linear polarization across the aperture.
    The polarization direction is specified by an angle from the x-axis.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid as (height, width) or (ny, nx).
    polarization_angle : ScalarFloat, optional
        Angle of polarization direction measured CCW from +x axis,
        in radians. Default is 0.0 (x-polarized).
        - 0 : x-polarized
        - π/2 : y-polarized
        - π/4 : 45° polarized
    beam_radius : ScalarFloat, optional
        Beam waist (1/e² intensity radius) in meters.
        If None, defaults to 1/4 of grid extent.
    amplitude : ScalarFloat, optional
        Peak amplitude at beam center, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position of the wavefront in meters, by default 0.0.
    apodization : str, optional
        Amplitude envelope: "gaussian" (default) or "uniform".

    Returns
    -------
    OpticalWavefront
        Linearly polarized beam as OpticalWavefront PyTree.
        Field shape is (H, W, 2) with [Ex, Ey] components.

    Notes
    -----
    For a linearly polarized beam, the Jones vector is constant across
    the aperture:

    .. math::
        \vec{E} = A(r) \begin{pmatrix} \cos(\theta) \\ \sin(\theta)
        \end{pmatrix}

    where θ is the polarization angle.

    When focused by a high-NA lens:
    - The focal spot is slightly elliptical (elongated along the
      polarization direction)
    - A weak Ez component appears due to depolarization effects
    - The effect increases with NA
    """
    _, _, rr, _ = _create_grid_and_polar_coords(dx, grid_size)

    grid_size_arr = jnp.asarray(grid_size, dtype=jnp.int32)
    grid_extent: Float[Array, " "] = jnp.minimum(
        grid_size_arr[0], grid_size_arr[1]
    ) * jnp.asarray(dx, dtype=jnp.float64)

    w: Float[Array, " "] = jax.lax.cond(
        beam_radius is None,
        lambda: grid_extent / 4.0,
        lambda: jnp.asarray(beam_radius, dtype=jnp.float64),
    )

    if apodization == "gaussian":
        amplitude_envelope: Float[Array, " ny nx"] = jnp.exp(
            -(rr**2) / (w**2)
        )
    elif apodization == "uniform":
        amplitude_envelope = jnp.where(rr <= w, 1.0, 0.0)
    else:
        amplitude_envelope = jnp.exp(-(rr**2) / (w**2))

    amp: Float[Array, " "] = jnp.asarray(amplitude, dtype=jnp.float64)
    theta: Float[Array, " "] = jnp.asarray(
        polarization_angle, dtype=jnp.float64
    )
    scaled_envelope: Float[Array, " ny nx"] = amp * amplitude_envelope

    ex: Complex[Array, " ny nx"] = (
        scaled_envelope * jnp.cos(theta)
    ).astype(jnp.complex128)
    ey: Complex[Array, " ny nx"] = (
        scaled_envelope * jnp.sin(theta)
    ).astype(jnp.complex128)

    field: Complex[Array, " ny nx 2"] = jnp.stack([ex, ey], axis=-1)

    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
        polarization=True,
    )
    return wavefront


@jaxtyped(typechecker=beartype)
def x_polarized_beam(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    beam_radius: ScalarFloat = None,
    amplitude: ScalarFloat = 1.0,
    z_position: ScalarFloat = 0.0,
    apodization: str = "gaussian",
) -> OpticalWavefront:
    """Generate an x-polarized Gaussian beam.

    Convenience wrapper for linear_polarized_beam with angle = 0.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid as (height, width).
    beam_radius : ScalarFloat, optional
        Beam waist (1/e² intensity radius) in meters.
    amplitude : ScalarFloat, optional
        Peak amplitude at beam center, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position in meters, by default 0.0.
    apodization : str, optional
        Amplitude envelope type.

    Returns
    -------
    OpticalWavefront
        X-polarized beam with field shape (H, W, 2).
    """
    return linear_polarized_beam(
        wavelength=wavelength,
        dx=dx,
        grid_size=grid_size,
        polarization_angle=0.0,
        beam_radius=beam_radius,
        amplitude=amplitude,
        z_position=z_position,
        apodization=apodization,
    )


@jaxtyped(typechecker=beartype)
def y_polarized_beam(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    beam_radius: ScalarFloat = None,
    amplitude: ScalarFloat = 1.0,
    z_position: ScalarFloat = 0.0,
    apodization: str = "gaussian",
) -> OpticalWavefront:
    """Generate a y-polarized Gaussian beam.

    Convenience wrapper for linear_polarized_beam with angle = π/2.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid as (height, width).
    beam_radius : ScalarFloat, optional
        Beam waist (1/e² intensity radius) in meters.
    amplitude : ScalarFloat, optional
        Peak amplitude at beam center, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position in meters, by default 0.0.
    apodization : str, optional
        Amplitude envelope type.

    Returns
    -------
    OpticalWavefront
        Y-polarized beam with field shape (H, W, 2).
    """
    return linear_polarized_beam(
        wavelength=wavelength,
        dx=dx,
        grid_size=grid_size,
        polarization_angle=jnp.pi / 2.0,
        beam_radius=beam_radius,
        amplitude=amplitude,
        z_position=z_position,
        apodization=apodization,
    )


@jaxtyped(typechecker=beartype)
def circular_polarized_beam(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    handedness: str = "right",
    beam_radius: ScalarFloat = None,
    amplitude: ScalarFloat = 1.0,
    z_position: ScalarFloat = 0.0,
    apodization: str = "gaussian",
) -> OpticalWavefront:
    r"""Generate a circularly polarized Gaussian beam.

    Creates a beam with uniform circular polarization (left or right
    handed) across the aperture.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid as (height, width).
    handedness : str, optional
        Polarization handedness: "right" (default) or "left".
        - "right" : Right-handed (clockwise when viewed from receiver)
        - "left" : Left-handed (counter-clockwise when viewed from receiver)
    beam_radius : ScalarFloat, optional
        Beam waist (1/e² intensity radius) in meters.
    amplitude : ScalarFloat, optional
        Peak amplitude at beam center, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position in meters, by default 0.0.
    apodization : str, optional
        Amplitude envelope type.

    Returns
    -------
    OpticalWavefront
        Circularly polarized beam with field shape (H, W, 2).

    Notes
    -----
    The Jones vectors for circular polarization are:

    Right-handed (RCP):

    .. math::
        \vec{E}_{RCP} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -i
        \end{pmatrix}

    Left-handed (LCP):

    .. math::
        \vec{E}_{LCP} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ i
        \end{pmatrix}

    Note: Sign conventions vary in the literature. We use the physics
    convention where RCP has E_y = -i*E_x (clockwise rotation for an
    observer looking into the beam).
    """
    _, _, rr, _ = _create_grid_and_polar_coords(dx, grid_size)

    grid_size_arr = jnp.asarray(grid_size, dtype=jnp.int32)
    grid_extent: Float[Array, " "] = jnp.minimum(
        grid_size_arr[0], grid_size_arr[1]
    ) * jnp.asarray(dx, dtype=jnp.float64)

    w: Float[Array, " "] = jax.lax.cond(
        beam_radius is None,
        lambda: grid_extent / 4.0,
        lambda: jnp.asarray(beam_radius, dtype=jnp.float64),
    )

    if apodization == "gaussian":
        amplitude_envelope: Float[Array, " ny nx"] = jnp.exp(
            -(rr**2) / (w**2)
        )
    else:
        amplitude_envelope = jnp.exp(-(rr**2) / (w**2))

    amp: Float[Array, " "] = jnp.asarray(amplitude, dtype=jnp.float64)
    scaled_envelope: Float[Array, " ny nx"] = (
        amp * amplitude_envelope / jnp.sqrt(2.0)
    )

    if handedness.lower() == "right":
        ex: Complex[Array, " ny nx"] = scaled_envelope.astype(jnp.complex128)
        ey: Complex[Array, " ny nx"] = (
            -1j * scaled_envelope
        ).astype(jnp.complex128)
    else:
        ex = scaled_envelope.astype(jnp.complex128)
        ey = (1j * scaled_envelope).astype(jnp.complex128)

    field: Complex[Array, " ny nx 2"] = jnp.stack([ex, ey], axis=-1)

    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
        polarization=True,
    )
    return wavefront


@jaxtyped(typechecker=beartype)
def generalized_cylindrical_vector_beam(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    order: ScalarInteger = 1,
    phase_offset: ScalarFloat = 0.0,
    beam_radius: ScalarFloat = None,
    amplitude: ScalarFloat = 1.0,
    z_position: ScalarFloat = 0.0,
) -> OpticalWavefront:
    r"""Generate a generalized cylindrical vector beam of arbitrary order.

    Creates a cylindrical vector beam with polarization pattern
    determined by the topological order m. Standard radial (m=1, φ₀=0)
    and azimuthal (m=1, φ₀=π/2) polarizations are special cases.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid as (height, width).
    order : ScalarInteger, optional
        Topological order m of the polarization pattern, by default 1.
        - m=1 : Standard radial/azimuthal (1 polarization singularity)
        - m=2 : Higher-order with 2 singularities
        - etc.
    phase_offset : ScalarFloat, optional
        Phase offset φ₀ in radians, by default 0.0.
        - φ₀=0 : Radial-like
        - φ₀=π/2 : Azimuthal-like
    beam_radius : ScalarFloat, optional
        Beam waist in meters.
    amplitude : ScalarFloat, optional
        Peak amplitude, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position in meters, by default 0.0.

    Returns
    -------
    OpticalWavefront
        Generalized cylindrical vector beam with field shape (H, W, 2).

    Notes
    -----
    The generalized cylindrical vector beam has polarization:

    .. math::
        E_x = A(r) \cos(m\phi + \phi_0)

        E_y = A(r) \sin(m\phi + \phi_0)

    For m=1:
    - φ₀=0 gives radial polarization
    - φ₀=π/2 gives azimuthal polarization

    Higher-order beams (m>1) have multiple polarization singularities
    and create complex focal field distributions.
    """
    _, _, rr, phi = _create_grid_and_polar_coords(dx, grid_size)

    grid_size_arr = jnp.asarray(grid_size, dtype=jnp.int32)
    grid_extent: Float[Array, " "] = jnp.minimum(
        grid_size_arr[0], grid_size_arr[1]
    ) * jnp.asarray(dx, dtype=jnp.float64)

    w: Float[Array, " "] = jax.lax.cond(
        beam_radius is None,
        lambda: grid_extent / 4.0,
        lambda: jnp.asarray(beam_radius, dtype=jnp.float64),
    )

    amp: Float[Array, " "] = jnp.asarray(amplitude, dtype=jnp.float64)
    amplitude_envelope: Float[Array, " ny nx"] = amp * jnp.exp(
        -(rr**2) / (w**2)
    )

    m: Float[Array, " "] = jnp.asarray(order, dtype=jnp.float64)
    phi_0: Float[Array, " "] = jnp.asarray(phase_offset, dtype=jnp.float64)

    polarization_angle: Float[Array, " ny nx"] = m * phi + phi_0

    ex: Complex[Array, " ny nx"] = (
        amplitude_envelope * jnp.cos(polarization_angle)
    ).astype(jnp.complex128)
    ey: Complex[Array, " ny nx"] = (
        amplitude_envelope * jnp.sin(polarization_angle)
    ).astype(jnp.complex128)

    field: Complex[Array, " ny nx 2"] = jnp.stack([ex, ey], axis=-1)

    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
        polarization=True,
    )
    return wavefront
