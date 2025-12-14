"""Optical beam generation functions.

Extended Summary
----------------
Factory functions for creating optical beams with physically meaningful
parameters. These functions generate OpticalWavefront PyTrees with the
correct amplitude and phase profiles for various beam types commonly
used in optical microscopy and imaging systems.

Routine Listings
----------------
plane_wave : function
    Creates a uniform plane wave with optional tilt
sinusoidal_wave : function
    Creates a sinusoidal interference pattern
collimated_gaussian : function
    Creates a collimated Gaussian beam with flat phase
converging_gaussian : function
    Creates a Gaussian beam converging to a focus
diverging_gaussian : function
    Creates a Gaussian beam diverging from a virtual source
gaussian_beam : function
    Creates a Gaussian beam from complex beam parameter q
bessel_beam : function
    Creates a Bessel beam with specified cone angle
laguerre_gaussian : function
    Creates Laguerre-Gaussian modes (includes vortex beams)
hermite_gaussian : function
    Creates Hermite-Gaussian modes
propagate_beam : function
    Generates a beam at multiple z positions as a PropagatingWavefront

Notes
-----
All beam generators return OpticalWavefront PyTrees that are compatible
with JAX transformations. The phase profiles encode the propagation
behavior of the beam - converging beams have quadratic phase that causes
focusing, while collimated beams have flat phase.
The key insight is that intensity alone does not determine beam behavior.
Two beams with identical intensity profiles but different phase profiles
will evolve completely differently upon propagation.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from janssen.optics import create_spatial_grid
from janssen.optics.bessel import bessel_j0
from janssen.utils import (
    OpticalWavefront,
    PropagatingWavefront,
    ScalarFloat,
    ScalarInteger,
    make_optical_wavefront,
)
from janssen.utils.factory import make_propagating_wavefront


@jaxtyped(typechecker=beartype)
def plane_wave(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    amplitude: ScalarFloat = 1.0,
    tilt_x: ScalarFloat = 0.0,
    tilt_y: ScalarFloat = 0.0,
    z_position: ScalarFloat = 0.0,
) -> OpticalWavefront:
    r"""Create a uniform plane wave with optional tilt.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid. If int, creates square grid.
        If tuple, specifies (height, width).
    amplitude : ScalarFloat, optional
        Amplitude of the plane wave, by default 1.0.
    tilt_x : ScalarFloat, optional
        Tilt angle along x-axis in radians (small angle), by default 0.0.
    tilt_y : ScalarFloat, optional
        Tilt angle along y-axis in radians (small angle), by default 0.0.
    z_position : ScalarFloat, optional
        Initial z position of the wavefront in meters, by default 0.0.

    Returns
    -------
    wavefront : OpticalWavefront
        Plane wave OpticalWavefront PyTree.

    Notes
    -----
    A plane wave has uniform amplitude and linear phase (flat if no tilt).
    The tilt angles introduce a linear phase ramp corresponding to
    propagation at an angle to the optical axis:

    .. math::
        E(x, y) = A \\exp(i k (x \\sin\\theta_x + y \\sin\\theta_y))

    For small angles, :math:`\\sin\\theta \\approx \\theta`.
    """
    grid_size = jnp.asarray(grid_size, dtype=jnp.int32)
    ny: ScalarInteger = grid_size[0]
    nx: ScalarInteger = grid_size[1]
    diameter: Float[Array, " 2"] = jnp.asarray(
        [nx * dx, ny * dx], dtype=jnp.float64
    )
    num_points: Int[Array, " 2"] = jnp.asarray([nx, ny], dtype=jnp.int32)
    xx, yy = create_spatial_grid(diameter, num_points)

    k: Float[Array, " "] = 2.0 * jnp.pi / jnp.asarray(wavelength)
    phase: Float[Array, " ny nx"] = k * (
        xx * jnp.asarray(tilt_x) + yy * jnp.asarray(tilt_y)
    )
    field: Complex[Array, " ny nx"] = jnp.asarray(amplitude) * jnp.exp(
        1j * phase
    )
    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
    )
    return wavefront


@jaxtyped(typechecker=beartype)
def sinusoidal_wave(
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    period: ScalarFloat,
    direction: ScalarFloat = 0.0,
    amplitude: ScalarFloat = 1.0,
    z_position: ScalarFloat = 0.0,
) -> OpticalWavefront:
    r"""Create a sinusoidal interference pattern.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid. If int, creates square grid.
        If tuple, specifies (height, width).
    period : ScalarFloat
        Spatial period of the sinusoidal pattern in meters.
    direction : ScalarFloat
        Direction angle of the sinusoidal pattern in radians.
        0 = horizontal stripes, pi/2 = vertical stripes.
        By default 0.0.
    amplitude : ScalarFloat
        Peak amplitude of the wave, by default 1.0.
    z_position : ScalarFloat
        Initial z position of the wavefront in meters, by default 0.0.

    Returns
    -------
    wavefront : OpticalWavefront
        Sinusoidal wave OpticalWavefront PyTree.

    Notes
    -----
    A sinusoidal wave has an intensity profile that varies as:

    .. math::
        E(x, y) = A \cos\left(\frac{2\pi}{T}(x \cos\theta + y \sin\theta)
        \right)

    where :math:`T` is the spatial period and :math:`\theta` is the
    direction angle.

    This pattern represents the interference of two plane waves and is
    useful for testing optical systems, creating gratings, and studying
    diffraction phenomena.
    """
    grid_size = jnp.asarray(grid_size, dtype=jnp.int32)
    ny: ScalarInteger = grid_size[0]
    nx: ScalarInteger = grid_size[1]
    diameter: Float[Array, " 2"] = jnp.asarray(
        [nx * dx, ny * dx], dtype=jnp.float64
    )
    num_points: Int[Array, " 2"] = jnp.asarray([nx, ny], dtype=jnp.int32)
    xx, yy = create_spatial_grid(diameter, num_points)

    period_arr: Float[Array, " "] = jnp.asarray(period, dtype=jnp.float64)
    direction_arr: Float[Array, " "] = jnp.asarray(
        direction, dtype=jnp.float64
    )

    spatial_coord: Float[Array, " ny nx"] = xx * jnp.cos(
        direction_arr
    ) + yy * jnp.sin(direction_arr)
    sinusoid: Float[Array, " ny nx"] = jnp.cos(
        2.0 * jnp.pi * spatial_coord / period_arr
    )

    field: Complex[Array, " ny nx"] = (
        jnp.asarray(amplitude, dtype=jnp.float64)
        * sinusoid
        * jnp.ones_like(sinusoid, dtype=jnp.complex128)
    )

    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
    )
    return wavefront


@jaxtyped(typechecker=beartype)
def collimated_gaussian(
    wavelength: ScalarFloat,
    waist: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    center: Optional[Tuple[ScalarFloat, ScalarFloat]] = (0.0, 0.0),
    amplitude: Optional[ScalarFloat] = 1.0,
    z_position: Optional[ScalarFloat] = 0.0,
) -> OpticalWavefront:
    r"""Create a collimated Gaussian beam with flat phase.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    waist : ScalarFloat
        Beam waist (1/e² intensity radius) in meters.
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
        Size of the computational grid. If int, creates square grid.
        If tuple, specifies (height, width).
    center : Tuple[ScalarFloat, ScalarFloat], optional
        Center position (x0, y0) in meters, by default (0.0, 0.0).
    amplitude : ScalarFloat, optional
        Peak amplitude at beam center, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position of the wavefront in meters, by default 0.0.

    Returns
    -------
    wavefront : OpticalWavefront
        Collimated Gaussian beam OpticalWavefront PyTree.

    Notes
    -----
    A collimated Gaussian beam has a Gaussian intensity profile and
    flat (constant) phase across the beam:

    .. math::
        E(x, y) = A \\exp\\left(-\\frac{(x-x_0)^2 + (y-y_0)^2}{w^2}\\right)

    where :math:`w` is the beam waist (1/e² intensity radius).

    This represents a beam at its waist position where the wavefront
    is planar. Upon propagation, the beam will expand due to diffraction.
    """
    grid_size = jnp.asarray(grid_size, dtype=jnp.int32)
    ny: ScalarInteger = grid_size[0]
    nx: ScalarInteger = grid_size[1]
    diameter: Float[Array, " 2"] = jnp.asarray(
        [nx * dx, ny * dx], dtype=jnp.float64
    )
    num_points: Int[Array, " 2"] = jnp.asarray([nx, ny], dtype=jnp.int32)
    xx: Float[Array, " ny nx"]
    yy: Float[Array, " ny nx"]
    xx, yy = create_spatial_grid(diameter, num_points)
    x0: Float[Array, " "] = jnp.asarray(center[0], dtype=jnp.float64)
    y0: Float[Array, " "] = jnp.asarray(center[1], dtype=jnp.float64)
    w: Float[Array, " "] = jnp.asarray(waist, dtype=jnp.float64)
    r2: Float[Array, " ny nx"] = (xx - x0) ** 2 + (yy - y0) ** 2
    field: Complex[Array, " ny nx"] = (
        jnp.asarray(amplitude, dtype=jnp.float64)
        * jnp.exp(-r2 / (w**2))
        * jnp.ones_like(r2, dtype=jnp.complex128)
    )
    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
    )
    return wavefront


@jaxtyped(typechecker=beartype)
def converging_gaussian(
    wavelength: ScalarFloat,
    waist: ScalarFloat,
    focus_distance: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    center: Optional[Tuple[ScalarFloat, ScalarFloat]] = (0.0, 0.0),
    amplitude: Optional[ScalarFloat] = 1.0,
    z_position: Optional[ScalarFloat] = 0.0,
) -> OpticalWavefront:
    r"""Create a Gaussian beam converging to a focus.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    waist : ScalarFloat
        Current beam waist (1/e² intensity radius) in meters.
        This is the waist at the current plane, not at the focus.
    focus_distance : ScalarFloat
        Distance to the focus in meters (positive = focus downstream).
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[int, Tuple[int, int]]
        Size of the computational grid. If int, creates square grid.
        If tuple, specifies (height, width).
    center : Tuple[ScalarFloat, ScalarFloat], optional
        Center position (x0, y0) in meters, by default (0.0, 0.0).
    amplitude : ScalarFloat, optional
        Peak amplitude at beam center, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position of the wavefront in meters, by default 0.0.

    Returns
    -------
    wavefront : OpticalWavefront
        Converging Gaussian beam OpticalWavefront PyTree.

    Notes
    -----
    A converging Gaussian beam has a Gaussian intensity profile with a
    spherical (quadratic) converging phase:

    .. math::
        E(x, y) = A \\exp\\left(-\\frac{r^2}{w^2}\\right)
                    \\exp\\left(-i \\frac{k r^2}{2 f}\\right)

    where :math:`f` is the focus distance and the negative sign indicates
    convergence (wavefront curving inward toward the optical axis).

    The radius of curvature R equals the focus distance f for a beam
    that will come to a focus at distance f downstream.

    Upon propagation, this beam will decrease in size until reaching
    the focus, then expand as a diverging beam.
    """
    grid_size = jnp.asarray(grid_size, dtype=jnp.int32)
    ny: ScalarInteger = grid_size[0]
    nx: ScalarInteger = grid_size[1]
    diameter: Float[Array, " 2"] = jnp.asarray(
        [nx * dx, ny * dx], dtype=jnp.float64
    )
    num_points: Int[Array, " 2"] = jnp.asarray([nx, ny], dtype=jnp.int32)
    xx: Float[Array, " ny nx"]
    yy: Float[Array, " ny nx"]
    xx, yy = create_spatial_grid(diameter, num_points)

    x0: Float[Array, " "] = jnp.asarray(center[0], dtype=jnp.float64)
    y0: Float[Array, " "] = jnp.asarray(center[1], dtype=jnp.float64)
    w: Float[Array, " "] = jnp.asarray(waist, dtype=jnp.float64)
    f: Float[Array, " "] = jnp.asarray(focus_distance, dtype=jnp.float64)
    k: Float[Array, " "] = 2.0 * jnp.pi / jnp.asarray(wavelength)
    r2: Float[Array, " ny nx"] = (xx - x0) ** 2 + (yy - y0) ** 2
    gaussian_amplitude: Float[Array, " ny nx"] = jnp.asarray(
        amplitude, dtype=jnp.float64
    ) * jnp.exp(-r2 / (w**2))
    safe_floor: float = 1e-15
    f_safe: Float[Array, " "] = jnp.where(
        jnp.abs(f) < safe_floor, safe_floor, f
    )
    converging_phase: Float[Array, " ny nx"] = -k * r2 / (2.0 * f_safe)
    field: Complex[Array, " ny nx"] = gaussian_amplitude * jnp.exp(
        1j * converging_phase
    )
    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
    )
    return wavefront


@jaxtyped(typechecker=beartype)
def diverging_gaussian(
    wavelength: ScalarFloat,
    waist: ScalarFloat,
    source_distance: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    center: Optional[Tuple[ScalarFloat, ScalarFloat]] = (0.0, 0.0),
    amplitude: Optional[ScalarFloat] = 1.0,
    z_position: Optional[ScalarFloat] = 0.0,
) -> OpticalWavefront:
    r"""Create a Gaussian beam diverging from a virtual source.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    waist : ScalarFloat
        Current beam waist (1/e² intensity radius) in meters.
        This is the waist at the current plane.
    source_distance : ScalarFloat
        Distance from the virtual source point in meters
        (positive = source was upstream).
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid. If int, creates square grid.
        If tuple, specifies (height, width).
    center : Tuple[ScalarFloat, ScalarFloat], optional
        Center position (x0, y0) in meters, by default (0.0, 0.0).
    amplitude : ScalarFloat, optional
        Peak amplitude at beam center, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position of the wavefront in meters, by default 0.0.

    Returns
    -------
    wavefront : OpticalWavefront
        Diverging Gaussian beam OpticalWavefront PyTree.

    Notes
    -----
    A diverging Gaussian beam has a Gaussian intensity profile with a
    spherical (quadratic) diverging phase:

    .. math::
        E(x, y) = A \\exp\\left(-\\frac{r^2}{w^2}\\right)
                    \\exp\\left(+i \\frac{k r^2}{2 R}\\right)

    where :math:`R` is the radius of curvature (positive for diverging,
    equal to the source distance).

    This represents a beam that originated from a point source at
    distance `source_distance` upstream. Upon propagation, this beam
    will continue to expand.
    """
    grid_size = jnp.asarray(grid_size, dtype=jnp.int32)
    ny: ScalarInteger = grid_size[0]
    nx: ScalarInteger = grid_size[1]
    diameter: Float[Array, " 2"] = jnp.asarray(
        [nx * dx, ny * dx], dtype=jnp.float64
    )
    num_points: Int[Array, " 2"] = jnp.asarray([nx, ny], dtype=jnp.int32)
    xx: Float[Array, " ny nx"]
    yy: Float[Array, " ny nx"]
    xx, yy = create_spatial_grid(diameter, num_points)
    x0: Float[Array, " "] = jnp.asarray(center[0], dtype=jnp.float64)
    y0: Float[Array, " "] = jnp.asarray(center[1], dtype=jnp.float64)
    w: Float[Array, " "] = jnp.asarray(waist, dtype=jnp.float64)
    rr: Float[Array, " "] = jnp.asarray(source_distance, dtype=jnp.float64)
    k: Float[Array, " "] = 2.0 * jnp.pi / jnp.asarray(wavelength)
    r2: Float[Array, " ny nx"] = (xx - x0) ** 2 + (yy - y0) ** 2
    gaussian_amplitude: Float[Array, " ny nx"] = jnp.asarray(
        amplitude, dtype=jnp.float64
    ) * jnp.exp(-r2 / (w**2))
    safe_floor: float = 1e-15
    rr_safe: Float[Array, " "] = jnp.where(
        jnp.abs(rr) < safe_floor, safe_floor, rr
    )
    diverging_phase: Float[Array, " ny nx"] = k * r2 / (2.0 * rr_safe)
    field: Complex[Array, " ny nx"] = gaussian_amplitude * jnp.exp(
        1j * diverging_phase
    )
    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
    )
    return wavefront


@jaxtyped(typechecker=beartype)
def gaussian_beam(
    wavelength: ScalarFloat,
    waist_0: ScalarFloat,
    z_from_waist: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    center: Optional[Tuple[ScalarFloat, ScalarFloat]] = (0.0, 0.0),
    amplitude: Optional[ScalarFloat] = 1.0,
    z_position: Optional[ScalarFloat] = 0.0,
    include_gouy_phase: Optional[bool] = True,
) -> OpticalWavefront:
    r"""Create a Gaussian beam at arbitrary position from waist.

    This is the most general Gaussian beam generator, using the full
    Gaussian beam propagation formulas including Gouy phase.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    waist_0 : ScalarFloat
        Beam waist at the waist position (minimum spot size) in meters.
    z_from_waist : ScalarFloat
        Distance from the beam waist in meters.
        Positive = downstream from waist (diverging).
        Negative = upstream from waist (converging toward waist).
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid. If int, creates square grid.
        If tuple, specifies (height, width).
    center : Tuple[ScalarFloat, ScalarFloat], optional
        Center position (x0, y0) in meters, by default (0.0, 0.0).
    amplitude : ScalarFloat, optional
        Peak amplitude at beam waist, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position of the wavefront in meters, by default 0.0.
    include_gouy_phase : bool, optional
        Whether to include the Gouy phase shift, by default True.

    Returns
    -------
    wavefront : OpticalWavefront
        Gaussian beam OpticalWavefront PyTree.

    Notes
    -----
    The complete Gaussian beam field is:

    .. math::
        E(r, z) = A \\frac{w_0}{w(z)} \\exp\\left(-\\frac{r^2}{w(z)^2}\\right)
                  \\exp\\left(-i k z - i \\frac{k r^2}{2 R(z)} + i \\zeta(z)\\right)

    where:

    - :math:`w(z) = w_0 \\sqrt{1 + (z/z_R)^2}` is the beam radius
    - :math:`R(z) = z (1 + (z_R/z)^2)` is the radius of curvature
    - :math:`\\zeta(z) = \\arctan(z/z_R)` is the Gouy phase
    - :math:`z_R = \\pi w_0^2 / \\lambda` is the Rayleigh range

    At the waist (z=0), the beam has minimum size and flat phase.
    The Gouy phase represents an additional phase shift accumulated
    through the focus.
    """
    grid_size = jnp.asarray(grid_size, dtype=jnp.int32)
    ny: ScalarInteger = grid_size[0]
    nx: ScalarInteger = grid_size[1]
    diameter: Float[Array, " 2"] = jnp.asarray(
        [nx * dx, ny * dx], dtype=jnp.float64
    )
    num_points: Int[Array, " 2"] = jnp.asarray([nx, ny], dtype=jnp.int32)
    xx: Float[Array, " ny nx"]
    yy: Float[Array, " ny nx"]
    xx, yy = create_spatial_grid(diameter, num_points)
    x0: Float[Array, " "] = jnp.asarray(center[0], dtype=jnp.float64)
    y0: Float[Array, " "] = jnp.asarray(center[1], dtype=jnp.float64)
    w0: Float[Array, " "] = jnp.asarray(waist_0, dtype=jnp.float64)
    z: Float[Array, " "] = jnp.asarray(z_from_waist, dtype=jnp.float64)
    lam: Float[Array, " "] = jnp.asarray(wavelength, dtype=jnp.float64)
    k: Float[Array, " "] = 2.0 * jnp.pi / lam

    # Rayleigh range
    z_R: Float[Array, " "] = jnp.pi * w0**2 / lam

    # Beam radius at position z
    w_z: Float[Array, " "] = w0 * jnp.sqrt(1.0 + (z / z_R) ** 2)

    # Radius of curvature at position z (handle z=0 case)
    z_safe: Float[Array, " "] = jnp.where(jnp.abs(z) < 1e-15, 1e-15, z)
    R_z: Float[Array, " "] = z_safe * (1.0 + (z_R / z_safe) ** 2)

    # Gouy phase
    gouy: Float[Array, " "] = jnp.arctan2(z, z_R)

    # Radial distance squared from center
    r2: Float[Array, " ny nx"] = (xx - x0) ** 2 + (yy - y0) ** 2

    # Amplitude: includes w0/w(z) factor for energy conservation
    amp: Float[Array, " ny nx"] = (
        jnp.asarray(amplitude, dtype=jnp.float64)
        * (w0 / w_z)
        * jnp.exp(-r2 / (w_z**2))
    )

    # Phase: curvature term + Gouy phase (axial phase kz omitted as
    # it's a global phase)
    curvature_phase: Float[Array, " ny nx"] = -k * r2 / (2.0 * R_z)

    # At waist (z=0), R_z -> infinity, so curvature_phase -> 0
    # Handle this by checking if we're very close to waist
    is_at_waist: Float[Array, " "] = jnp.abs(z) < 1e-12
    curvature_phase = jnp.where(is_at_waist, 0.0, curvature_phase)

    gouy_phase: Float[Array, " "] = jnp.where(include_gouy_phase, gouy, 0.0)

    total_phase: Float[Array, " ny nx"] = curvature_phase + gouy_phase

    field: Complex[Array, " ny nx"] = amp * jnp.exp(1j * total_phase)

    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
    )
    return wavefront


@jaxtyped(typechecker=beartype)
def bessel_beam(
    wavelength: ScalarFloat,
    cone_angle: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    amplitude: Optional[ScalarFloat] = 1.0,
    z_position: Optional[ScalarFloat] = 0.0,
) -> OpticalWavefront:
    r"""Create a Bessel beam with specified cone angle.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    cone_angle : ScalarFloat
        Cone half-angle in radians. Determines the transverse wave
        vector component: k_r = k * sin(cone_angle).
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid. If int, creates square grid.
        If tuple, specifies (height, width).
    amplitude : ScalarFloat, optional
        Peak amplitude at beam center, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position of the wavefront in meters, by default 0.0.

    Returns
    -------
    wavefront : OpticalWavefront
        Bessel beam OpticalWavefront PyTree.

    Notes
    -----
    An ideal Bessel beam has a transverse profile given by the
    zeroth-order Bessel function:

    .. math::
        E(r) = A J_0(k_r r)

    where :math:`k_r = k \\sin(\\theta)` is the transverse wave vector
    and :math:`\\theta` is the cone half-angle.

    Bessel beams are "non-diffracting" - their transverse profile
    remains constant upon propagation (in the ideal infinite case).
    In practice, physical Bessel beams have finite extent and
    eventually diffract.

    The central lobe radius (first zero) is approximately
    :math:`r_0 \\approx 2.405 / k_r`.
    """
    grid_size = jnp.asarray(grid_size, dtype=jnp.int32)
    ny: ScalarInteger = grid_size[0]
    nx: ScalarInteger = grid_size[1]
    diameter: Float[Array, " 2"] = jnp.asarray(
        [nx * dx, ny * dx], dtype=jnp.float64
    )
    num_points: Int[Array, " 2"] = jnp.asarray([nx, ny], dtype=jnp.int32)
    xx: Float[Array, " ny nx"]
    yy: Float[Array, " ny nx"]
    xx, yy = create_spatial_grid(diameter, num_points)

    k: Float[Array, " "] = 2.0 * jnp.pi / jnp.asarray(wavelength)
    k_r: Float[Array, " "] = k * jnp.sin(jnp.asarray(cone_angle))

    # Radial distance from center
    r: Float[Array, " ny nx"] = jnp.sqrt(xx**2 + yy**2)

    # Bessel function J_0(k_r * r)
    bessel_profile: Float[Array, " ny nx"] = bessel_j0(k_r * r)

    field: Complex[Array, " ny nx"] = (
        jnp.asarray(amplitude, dtype=jnp.float64)
        * bessel_profile
        * jnp.ones_like(r, dtype=jnp.complex128)
    )

    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
    )
    return wavefront


@jaxtyped(typechecker=beartype)
def laguerre_gaussian(
    wavelength: ScalarFloat,
    waist: ScalarFloat,
    p: ScalarInteger,
    l: ScalarInteger,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    center: Optional[Tuple[ScalarFloat, ScalarFloat]] = (0.0, 0.0),
    amplitude: Optional[ScalarFloat] = 1.0,
    z_position: Optional[ScalarFloat] = 0.0,
) -> OpticalWavefront:
    r"""Create a Laguerre-Gaussian mode at the beam waist.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    waist : ScalarFloat
        Beam waist (1/e² intensity radius of fundamental mode) in meters.
    p : ScalarInteger
        Radial mode index (number of radial nodes), p >= 0.
    l : ScalarInteger
        Azimuthal mode index (topological charge for vortex beams).
        Can be positive or negative.
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid. If int, creates square grid.
        If tuple, specifies (height, width).
    center : Tuple[ScalarFloat, ScalarFloat], optional
        Center position (x0, y0) in meters, by default (0.0, 0.0).
    amplitude : ScalarFloat, optional
        Peak amplitude normalization, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position of the wavefront in meters, by default 0.0.

    Returns
    -------
    wavefront : OpticalWavefront
        Laguerre-Gaussian mode OpticalWavefront PyTree.

    Notes
    -----
    The Laguerre-Gaussian modes at the waist are:

    .. math::
        E_{p,l}(r, \\phi) = A \\left(\\frac{r\\sqrt{2}}{w}\\right)^{|l|}
                           L_p^{|l|}\\left(\\frac{2r^2}{w^2}\\right)
                           \\exp\\left(-\\frac{r^2}{w^2}\\right)
                           \\exp(i l \\phi)

    where :math:`L_p^{|l|}` is the generalized Laguerre polynomial.

    Special cases:

    - (p=0, l=0): Fundamental Gaussian mode
    - (p=0, l≠0): Optical vortex beams with topological charge l
    - (p>0, l=0): Radial modes with p rings
    """
    grid_size = jnp.asarray(grid_size, dtype=jnp.int32)
    ny: ScalarInteger = grid_size[0]
    nx: ScalarInteger = grid_size[1]
    diameter: Float[Array, " 2"] = jnp.asarray(
        [nx * dx, ny * dx], dtype=jnp.float64
    )
    num_points: Int[Array, " 2"] = jnp.asarray([nx, ny], dtype=jnp.int32)
    xx: Float[Array, " ny nx"]
    yy: Float[Array, " ny nx"]
    xx, yy = create_spatial_grid(diameter, num_points)
    x0: Float[Array, " "] = jnp.asarray(center[0], dtype=jnp.float64)
    y0: Float[Array, " "] = jnp.asarray(center[1], dtype=jnp.float64)
    w: Float[Array, " "] = jnp.asarray(waist, dtype=jnp.float64)
    x_shifted: Float[Array, " ny nx"] = xx - x0
    y_shifted: Float[Array, " ny nx"] = yy - y0
    r: Float[Array, " ny nx"] = jnp.sqrt(x_shifted**2 + y_shifted**2)
    phi: Float[Array, " ny nx"] = jnp.arctan2(y_shifted, x_shifted)
    rho: Float[Array, " ny nx"] = r * jnp.sqrt(2.0) / w
    rho2: Float[Array, " ny nx"] = 2.0 * r**2 / (w**2)
    abs_l: Int[Array, " "] = jnp.abs(jnp.asarray(l, dtype=jnp.int32))
    p_int: Int[Array, " "] = jnp.asarray(p, dtype=jnp.int32)

    def _laguerre_polynomial(
        n: Int[Array, " "], alpha: Int[Array, " "], x: Float[Array, " ny nx"]
    ) -> Float[Array, " ny nx"]:
        """Compute generalized Laguerre polynomial L_n^alpha(x)."""

        def body_fn(
            k: int,
            carry: Tuple[Float[Array, " ny nx"], Float[Array, " ny nx"]],
        ) -> Tuple[Float[Array, " ny nx"], Float[Array, " ny nx"]]:
            L_km1, L_km2 = carry
            k_float = jnp.float64(k)
            alpha_float = jnp.float64(alpha)
            L_k = (
                (2 * k_float - 1 + alpha_float - x) * L_km1
                - (k_float - 1 + alpha_float) * L_km2
            ) / k_float
            return (L_k, L_km1)

        L_0 = jnp.ones_like(x)
        L_1 = 1.0 + jnp.float64(alpha) - x

        result = jax.lax.cond(
            n == 0,
            lambda: L_0,
            lambda: jax.lax.cond(
                n == 1,
                lambda: L_1,
                lambda: jax.lax.fori_loop(2, n + 1, body_fn, (L_1, L_0))[0],
            ),
        )
        return result

    L_pl: Float[Array, " ny nx"] = _laguerre_polynomial(p_int, abs_l, rho2)
    radial_amplitude: Float[Array, " ny nx"] = (
        (rho ** jnp.float64(abs_l)) * jnp.exp(-(r**2) / (w**2)) * L_pl
    )
    l_float: Float[Array, " "] = jnp.float64(l)
    azimuthal_phase: Complex[Array, " ny nx"] = jnp.exp(1j * l_float * phi)
    field: Complex[Array, " ny nx"] = (
        jnp.asarray(amplitude, dtype=jnp.float64)
        * radial_amplitude
        * azimuthal_phase
    )
    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
    )
    return wavefront


@jaxtyped(typechecker=beartype)
def hermite_gaussian(
    wavelength: ScalarFloat,
    waist: ScalarFloat,
    n: ScalarInteger,
    m: ScalarInteger,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    center: Optional[Tuple[ScalarFloat, ScalarFloat]] = (0.0, 0.0),
    amplitude: Optional[ScalarFloat] = 1.0,
    z_position: Optional[ScalarFloat] = 0.0,
) -> OpticalWavefront:
    r"""Create a Hermite-Gaussian mode at the beam waist.

    Parameters
    ----------
    wavelength : ScalarFloat
        Wavelength of light in meters.
    waist : ScalarFloat
        Beam waist (1/e² intensity radius of fundamental mode) in meters.
    n : ScalarInteger
        Mode index in x direction (number of nodes along x), n >= 0.
    m : ScalarInteger
        Mode index in y direction (number of nodes along y), m >= 0.
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid. If int, creates square grid.
        If tuple, specifies (height, width).
    center : Tuple[ScalarFloat, ScalarFloat], optional
        Center position (x0, y0) in meters, by default (0.0, 0.0).
    amplitude : ScalarFloat, optional
        Peak amplitude normalization, by default 1.0.
    z_position : ScalarFloat, optional
        Initial z position of the wavefront in meters, by default 0.0.

    Returns
    -------
    wavefront : OpticalWavefront
        Hermite-Gaussian mode OpticalWavefront PyTree.

    Notes
    -----
    The Hermite-Gaussian modes at the waist are:

    .. math::
        E_{n,m}(x, y) = A H_n\\left(\\frac{x\\sqrt{2}}{w}\\right)
                         H_m\\left(\\frac{y\\sqrt{2}}{w}\\right)
                         \\exp\\left(-\\frac{x^2 + y^2}{w^2}\\right)

    where :math:`H_n` is the physicist's Hermite polynomial.

    Special cases:

    - (n=0, m=0): Fundamental Gaussian mode (TEM00)
    - (n=1, m=0): TEM10 mode with one node along x
    - (n=0, m=1): TEM01 mode with one node along y
    """
    grid_size = jnp.asarray(grid_size, dtype=jnp.int32)
    ny_grid: Int[Array, " "] = grid_size[0]
    nx_grid: Int[Array, " "] = grid_size[1]
    diameter: Float[Array, " 2"] = jnp.asarray(
        [nx_grid * dx, ny_grid * dx], dtype=jnp.float64
    )
    num_points: Int[Array, " 2"] = jnp.asarray(
        [nx_grid, ny_grid], dtype=jnp.int32
    )
    xx: Float[Array, " ny_grid nx_grid"]
    yy: Float[Array, " ny_grid nx_grid"]
    xx, yy = create_spatial_grid(diameter, num_points)
    x0: Float[Array, " "] = jnp.asarray(center[0], dtype=jnp.float64)
    y0: Float[Array, " "] = jnp.asarray(center[1], dtype=jnp.float64)
    w: Float[Array, " "] = jnp.asarray(waist, dtype=jnp.float64)
    x_norm: Float[Array, " ny_grid nx_grid"] = (xx - x0) * jnp.sqrt(2.0) / w
    y_norm: Float[Array, " ny_grid nx_grid"] = (yy - y0) * jnp.sqrt(2.0) / w
    n_int: Int[Array, " "] = jnp.asarray(n, dtype=jnp.int32)
    m_int: Int[Array, " "] = jnp.asarray(m, dtype=jnp.int32)

    def _hermite_polynomial(
        order: Int[Array, " "], x: Float[Array, " ny_grid nx_grid"]
    ) -> Float[Array, " ny_grid nx_grid"]:
        """Compute physicist's Hermite polynomial H_n(x)."""

        def _body_fn(
            k: int,
            carry: Tuple[
                Float[Array, " ny_grid nx_grid"],
                Float[Array, " ny_grid nx_grid"],
            ],
        ) -> Tuple[
            Float[Array, " ny_grid nx_grid"], Float[Array, " ny_grid nx_grid"]
        ]:
            hh_km1: Float[Array, " ny_grid nx_grid"]
            hh_km2: Float[Array, " ny_grid nx_grid"]
            hh_km1, hh_km2 = carry
            hh_k: Float[Array, " ny_grid nx_grid"] = (2.0 * x * hh_km1) - (
                2.0 * (k - 1) * hh_km2
            )
            return (hh_k, hh_km1)

        hh_0: Float[Array, " ny_grid nx_grid"] = jnp.ones_like(x)
        hh_1: Float[Array, " ny_grid nx_grid"] = 2.0 * x

        result: Float[Array, " ny_grid nx_grid"] = jax.lax.cond(
            order == 0,
            lambda: hh_0,
            lambda: jax.lax.cond(
                order == 1,
                lambda: hh_1,
                lambda: jax.lax.fori_loop(
                    2, order + 1, _body_fn, (hh_1, hh_0)
                )[0],
            ),
        )
        return result

    hh_n: Float[Array, " ny_grid nx_grid"] = _hermite_polynomial(n_int, x_norm)
    hh_m: Float[Array, " ny_grid nx_grid"] = _hermite_polynomial(m_int, y_norm)
    r2: Float[Array, " ny_grid nx_grid"] = (xx - x0) ** 2 + (yy - y0) ** 2
    gaussian: Float[Array, " ny_grid nx_grid"] = jnp.exp(-r2 / (w**2))
    field: Complex[Array, " ny_grid nx_grid"] = (
        jnp.asarray(amplitude, dtype=jnp.float64)
        * hh_n
        * hh_m
        * gaussian
        * jnp.ones_like(gaussian, dtype=jnp.complex128)
    )
    wavefront: OpticalWavefront = make_optical_wavefront(
        field=field,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
    )
    return wavefront


@jaxtyped(typechecker=beartype)
def propagate_beam(
    beam_type: str,
    z_positions: Float[Array, " zz"],
    wavelength: ScalarFloat,
    dx: ScalarFloat,
    grid_size: Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]],
    waist: ScalarFloat = 1e-3,
    amplitude: ScalarFloat = 1.0,
    center: Tuple[ScalarFloat, ScalarFloat] = (0.0, 0.0),
    tilt_x: ScalarFloat = 0.0,
    tilt_y: ScalarFloat = 0.0,
    focus_distance: ScalarFloat = 1.0,
    source_distance: ScalarFloat = 1.0,
    waist_0: ScalarFloat = 1e-3,
    include_gouy_phase: bool = True,
    cone_angle: ScalarFloat = 0.01,
    period: ScalarFloat = 1e-4,
    direction: ScalarFloat = 0.0,
    p: ScalarInteger = 0,
    l: ScalarInteger = 0,
    n: ScalarInteger = 0,
    m: ScalarInteger = 0,
) -> PropagatingWavefront:
    """Generate a beam at multiple z positions as a PropagatingWavefront.

    Parameters
    ----------
    beam_type : str
        Type of beam to generate. One of:
        - "plane_wave": Uniform plane wave with optional tilt
        - "sinusoidal_wave": Sinusoidal interference pattern
        - "collimated_gaussian": Collimated Gaussian beam with flat phase
        - "converging_gaussian": Gaussian beam converging to a focus
        - "diverging_gaussian": Gaussian beam diverging from a source
        - "gaussian_beam": General Gaussian beam at arbitrary z from waist
        - "bessel_beam": Bessel beam with specified cone angle
        - "laguerre_gaussian": Laguerre-Gaussian modes
        - "hermite_gaussian": Hermite-Gaussian modes
    z_positions : Float[Array, " zz"]
        Array of z positions at which to evaluate the beam (meters).
        For "gaussian_beam", these are distances from the waist.
        For other beam types, these set the z_position attribute.
    wavelength : ScalarFloat
        Wavelength of light in meters.
    dx : ScalarFloat
        Spatial sampling interval (pixel size) in meters.
    grid_size : Union[Int[Array, " 2"], Tuple[ScalarInteger, ScalarInteger]]
        Size of the computational grid as (height, width).
    waist : ScalarFloat
        Beam waist (1/e² intensity radius) in meters. Used by
        collimated_gaussian, converging_gaussian, diverging_gaussian,
        laguerre_gaussian, hermite_gaussian. Default is 1e-3.
    amplitude : ScalarFloat
        Peak amplitude, by default 1.0.
    center : Tuple[ScalarFloat, ScalarFloat]
        Center position (x0, y0) in meters, by default (0.0, 0.0).
    tilt_x : ScalarFloat
        Tilt angle along x-axis in radians for plane_wave, by default 0.0.
    tilt_y : ScalarFloat
        Tilt angle along y-axis in radians for plane_wave, by default 0.0.
    focus_distance : ScalarFloat
        Distance to focus for converging_gaussian (meters). Default is 1.0.
    source_distance : ScalarFloat
        Distance from source for diverging_gaussian (meters). Default is 1.0.
    waist_0 : ScalarFloat
        Beam waist at the waist position for gaussian_beam (meters).
        Default is 1e-3.
    include_gouy_phase : bool
        Whether to include Gouy phase for gaussian_beam, by default True.
    cone_angle : ScalarFloat
        Cone half-angle in radians for bessel_beam. Default is 0.01.
    period : ScalarFloat
        Spatial period for sinusoidal_wave (meters). Default is 1e-4.
    direction : ScalarFloat
        Direction angle for sinusoidal_wave (radians). Default is 0.0.
    p : ScalarInteger
        Radial mode index for laguerre_gaussian, by default 0.
    l : ScalarInteger
        Azimuthal mode index for laguerre_gaussian, by default 0.
    n : ScalarInteger
        Mode index in x direction for hermite_gaussian, by default 0.
    m : ScalarInteger
        Mode index in y direction for hermite_gaussian, by default 0.

    Returns
    -------
    propagating_wavefront : PropagatingWavefront
        A PropagatingWavefront containing the beam at all specified z
        positions.

    Raises
    ------
    ValueError
        If beam_type is not recognized.

    Notes
    -----
    This function uses jax.vmap to efficiently generate the beam at all
    z positions in parallel. The resulting PropagatingWavefront can be
    used to visualize beam evolution or as input to propagation algorithms.

    For "gaussian_beam", the z_positions represent distances from the beam
    waist, allowing visualization of beam evolution through focus.
    For other beam types, z_positions sets the z_position attribute but
    the field profile remains constant (as these are evaluated at a
    single plane).
    """
    z_positions_arr = jnp.asarray(z_positions, dtype=jnp.float64)

    if beam_type == "plane_wave":

        def _make_beam(z: Float[Array, " "]) -> Complex[Array, " hh ww"]:
            wf = plane_wave(
                wavelength=wavelength,
                dx=dx,
                grid_size=grid_size,
                amplitude=amplitude,
                tilt_x=tilt_x,
                tilt_y=tilt_y,
                z_position=z,
            )
            return wf.field

    elif beam_type == "sinusoidal_wave":

        def _make_beam(z: Float[Array, " "]) -> Complex[Array, " hh ww"]:
            wf = sinusoidal_wave(
                wavelength=wavelength,
                dx=dx,
                grid_size=grid_size,
                period=period,
                direction=direction,
                amplitude=amplitude,
                z_position=z,
            )
            return wf.field

    elif beam_type == "collimated_gaussian":

        def _make_beam(z: Float[Array, " "]) -> Complex[Array, " hh ww"]:
            wf = collimated_gaussian(
                wavelength=wavelength,
                waist=waist,
                dx=dx,
                grid_size=grid_size,
                center=center,
                amplitude=amplitude,
                z_position=z,
            )
            return wf.field

    elif beam_type == "converging_gaussian":

        def _make_beam(z: Float[Array, " "]) -> Complex[Array, " hh ww"]:
            wf = converging_gaussian(
                wavelength=wavelength,
                waist=waist,
                focus_distance=focus_distance,
                dx=dx,
                grid_size=grid_size,
                center=center,
                amplitude=amplitude,
                z_position=z,
            )
            return wf.field

    elif beam_type == "diverging_gaussian":

        def _make_beam(z: Float[Array, " "]) -> Complex[Array, " hh ww"]:
            wf = diverging_gaussian(
                wavelength=wavelength,
                waist=waist,
                source_distance=source_distance,
                dx=dx,
                grid_size=grid_size,
                center=center,
                amplitude=amplitude,
                z_position=z,
            )
            return wf.field

    elif beam_type == "gaussian_beam":

        def _make_beam(z: Float[Array, " "]) -> Complex[Array, " hh ww"]:
            wf = gaussian_beam(
                wavelength=wavelength,
                waist_0=waist_0,
                z_from_waist=z,
                dx=dx,
                grid_size=grid_size,
                center=center,
                amplitude=amplitude,
                z_position=z,
                include_gouy_phase=include_gouy_phase,
            )
            return wf.field

    elif beam_type == "bessel_beam":

        def _make_beam(z: Float[Array, " "]) -> Complex[Array, " hh ww"]:
            wf = bessel_beam(
                wavelength=wavelength,
                cone_angle=cone_angle,
                dx=dx,
                grid_size=grid_size,
                amplitude=amplitude,
                z_position=z,
            )
            return wf.field

    elif beam_type == "laguerre_gaussian":

        def _make_beam(z: Float[Array, " "]) -> Complex[Array, " hh ww"]:
            wf = laguerre_gaussian(
                wavelength=wavelength,
                waist=waist,
                p=p,
                l=l,
                dx=dx,
                grid_size=grid_size,
                center=center,
                amplitude=amplitude,
                z_position=z,
            )
            return wf.field

    elif beam_type == "hermite_gaussian":

        def _make_beam(z: Float[Array, " "]) -> Complex[Array, " hh ww"]:
            wf = hermite_gaussian(
                wavelength=wavelength,
                waist=waist,
                n=n,
                m=m,
                dx=dx,
                grid_size=grid_size,
                center=center,
                amplitude=amplitude,
                z_position=z,
            )
            return wf.field

    else:
        raise ValueError(
            f"Unknown beam_type: {beam_type}. Must be one of: "
            "plane_wave, sinusoidal_wave, collimated_gaussian, "
            "converging_gaussian, diverging_gaussian, gaussian_beam, "
            "bessel_beam, laguerre_gaussian, hermite_gaussian"
        )

    fields: Complex[Array, " zz hh ww"] = jax.vmap(_make_beam)(z_positions_arr)

    return make_propagating_wavefront(
        field=fields,
        wavelength=wavelength,
        dx=dx,
        z_positions=z_positions_arr,
        polarization=False,
    )
