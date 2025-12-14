"""Optical element implementations.

Extended Summary
----------------
Optical elements and components for building complex optical systems.
Includes gratings, waveplates, polarizers, beam splitters, and other
common optical components used in microscopy and optical systems.

Routine Listings
----------------
prism_phase_ramp : function
    Applies a linear phase ramp to simulate beam deviation/dispersion
beam_splitter : function
    Splits a field into transmitted and reflected arms with given (t, r)
mirror_reflection : function
    Applies mirror reflection(s):
    coordinate flip(s), optional conjugation, π phase
phase_grating_sine : function
    Sinusoidal phase grating
amplitude_grating_binary : function
    Binary amplitude grating with duty cycle
phase_grating_sawtooth : function
    Blazed (sawtooth) phase grating.
apply_phase_mask : function
    Applies an arbitrary phase mask (SLM / phase screen).
apply_phase_mask_fn : function
    Builds a phase mask from a callable f(xx, yy) and applies it.
polarizer_jones
    Linear polarizer at angle theta (Jones matrix) for 2-component
    fields.
waveplate_jones : function
    Waveplate (retarder) with retardance delta and fast axis angle
    theta.
nd_filter : function
    Neutral density filter with optical density (OD) or direct
    transmittance.
quarter_waveplate : function
    Quarter-waveplate with fast axis angle theta.
half_waveplate : function
    Half-waveplate with fast axis angle theta.
phase_grating_blazed_elliptical : function
    Elliptical blazed phase grating with period_x, period_y, theta,
    depth, and two_dim
_rotate_coords : function, internal
    Rotate coordinates by an angle theta.

Notes
-----
All optical elements are implemented as pure JAX functions and support
automatic differentiation. Elements can be composed to create complex
optical systems. Polarization-sensitive elements use Jones calculus for
vectorial field calculations.
    Rotates coordinates by an angle theta.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Optional, Tuple
from jax.lax import cond
from jaxtyping import Array, Bool, Complex, Float, Num, jaxtyped

from janssen.utils import (
    OpticalWavefront,
    ScalarBool,
    ScalarFloat,
    make_optical_wavefront,
)

from .apertures import _arrayed_grids
from .helper import add_phase_screen


def _rotate_coords(
    xx: Num[Array, " ..."], yy: Num[Array, " ..."], theta: ScalarFloat
) -> Tuple[Num[Array, " ..."], Num[Array, " ..."]]:
    """
    Rotate coordinates by an angle theta.

    Parameters
    ----------
    xx : Num[Array, " ..."]
        Grid of x coordinates.
    yy : Num[Array, " ..."]
        Grid of y coordinates.
    theta : ScalarFloat
        Angle of rotation in radians.

    Returns
    -------
    uu : Num[Array, " ..."]
        Rotated x coordinates.
    vv : Num[Array, " ..."]
        Rotated y coordinates.

    Notes
    -----
    - Rotates coordinates by an angle theta.
    - Uses cosine and sine to compute the rotation matrix.
    - Returns the rotated coordinates.
    """
    ct: Float[Array, " "] = jnp.cos(theta)
    st: Float[Array, " "] = jnp.sin(theta)
    uu: Num[Array, " ..."] = (ct * xx) + (st * yy)
    vv: Num[Array, " ..."] = (ct * yy) - (st * xx)
    return (uu, vv)


@jaxtyped(typechecker=beartype)
def prism_phase_ramp(
    incoming: OpticalWavefront,
    deflect_x: Optional[ScalarFloat] = 0.0,
    deflect_y: Optional[ScalarFloat] = 0.0,
    use_small_angle: Optional[ScalarBool] = True,
) -> OpticalWavefront:
    """
    Apply a linear phase ramp to simulate a prism-induced beam
    deviation.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input scalar wavefront.
    deflect_x : ScalarFloat, optional
        Deflection along +x.
        If `use_small_angle` is True, interpreted as angle (rad).
        Otherwise interpreted as spatial frequency kx [rad/m], by
        default 0.0.
    deflect_y : ScalarFloat, optional
        Deflection along +y (angle or ky), by default 0.0.
    use_small_angle : bool, optional
        If True, convert small angles to kx, ky via k*sin(angle) ~
        k*angle.
        Default True.

    Returns
    -------
    output_phasefront : OpticalWavefront
        Wavefront with added linear phase.

    Notes
    -----
    - Build xx, yy grids (m).
    - Compute kx, ky from deflections.
    - Phase = kx*xx + ky*yy; multiply by exp(i*phase).
    """
    arr_zeros: Float[Array, " hh ww"] = jnp.zeros_like(
        incoming.field, dtype=jnp.float64
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = _arrayed_grids(arr_zeros, arr_zeros, incoming.dx)
    k: ScalarFloat = (2.0 * jnp.pi) / incoming.wavelength
    kx: ScalarFloat
    ky: ScalarFloat
    kx, ky = cond(
        use_small_angle,
        lambda: (k * deflect_x, k * deflect_y),
        lambda: (deflect_x, deflect_y),
    )
    phase: Float[Array, " hh ww"] = (kx * xx) + (ky * yy)
    field_out: Complex[Array, " hh ww"] = add_phase_screen(
        incoming.field, phase
    )
    output_phasefront: OpticalWavefront = make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return output_phasefront


@jaxtyped(typechecker=beartype)
def beam_splitter(
    incoming: OpticalWavefront,
    t2: Optional[ScalarFloat] = 0.5,
    r2: Optional[ScalarFloat] = 0.5,
    normalize: Optional[ScalarBool] = True,
) -> Tuple[OpticalWavefront, OpticalWavefront]:
    """
    Split an input field into transmitted and reflected components.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront (scalar field).
    t2 : ScalarFloat, optional
        Complex transmission amplitude, by default jnp.sqrt(0.5).
    r2 : ScalarFloat, optional
        Complex reflection amplitude.
        Default 1j * jnp.sqrt(0.5) for 50/50 convention.
    normalize : bool, optional
        If True, scale (t, r) so that |t|^2 + |r|^2 = 1, by default
        True.

    Returns
    -------
    wf_T : OpticalWavefront
        Transmitted arm (t * field).
    wf_R : OpticalWavefront
        Reflected arm (r * field).

    Notes
    -----
    - Optionally renormalize (t, r).
    - Multiply field by t and r.
    - Return two wavefronts sharing same metadata.
    """
    t_val: Complex[Array, " "] = jnp.sqrt(
        jnp.asarray(t2, dtype=jnp.complex128)
    )
    r_val: Complex[Array, " "] = 1j * jnp.sqrt(
        jnp.asarray(r2, dtype=jnp.complex128)
    )

    def normalize_values() -> Tuple[Complex[Array, " "], Complex[Array, " "]]:
        power: Float[Array, " "] = (jnp.abs(t_val) ** 2) + (
            jnp.abs(r_val) ** 2
        )
        sqrt_power: Float[Array, " "] = jnp.sqrt(jnp.maximum(power, 1e-20))
        t_norm: Complex[Array, " "] = t_val / sqrt_power
        r_norm: Complex[Array, " "] = r_val / sqrt_power
        return (t_norm, r_norm)

    t_val, r_val = cond(normalize, normalize_values, lambda: (t_val, r_val))

    wf_t: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * t_val,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    wf_r: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * r_val,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return (wf_t, wf_r)


@jaxtyped(typechecker=beartype)
def mirror_reflection(
    incoming: OpticalWavefront,
    flip_x: Optional[ScalarBool] = True,
    flip_y: Optional[ScalarBool] = False,
    add_pi_phase: Optional[ScalarBool] = True,
    conjugate: Optional[ScalarBool] = True,
) -> OpticalWavefront:
    """
    Mirror reflection:
        coordinate flips with optional π-phase and conjugation.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront.
    flip_x : bool, optional
        Flip along x-axis (columns), by default True.
    flip_y : bool, optional
        Flip along y-axis (rows), by default False.
    add_pi_phase : bool, optional
        Multiply by exp(i*pi) = -1 to simulate phase inversion on
        reflection.
        Default True.
    conjugate : bool, optional
        Conjugate the complex field, useful when reversing propagation
        direction. Default is True.

    Returns
    -------
    OpticalWavefront
        Reflected wavefront.

    Notes
    -----
    - Flip axes as requested (jnp.flip).
    - Optional complex conjugation.
    - Optional -1 factor for π phase.
    """
    field = incoming.field
    field = cond(flip_x, lambda f: jnp.flip(f, axis=-1), lambda f: f, field)
    field = cond(flip_y, lambda f: jnp.flip(f, axis=-2), lambda f: f, field)
    field = cond(conjugate, jnp.conjugate, lambda f: f, field)
    field = cond(add_pi_phase, lambda f: -f, lambda f: f, field)

    return make_optical_wavefront(
        field=field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def phase_grating_sine(
    incoming: OpticalWavefront,
    period: ScalarFloat,
    depth: ScalarFloat,
    theta: Optional[ScalarFloat] = 0.0,
) -> OpticalWavefront:
    """
    Sinusoidal phase grating.

    Phase = depth * sin(2π * u / period), where u is the coordinate
    along the grating direction.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input field.
    period : ScalarFloat
        Grating period in meters.
    depth : ScalarFloat
        Phase modulation depth in radians.
    theta : ScalarFloat, optional
        Grating orientation (radians, CCW from x), by default 0.0.

    Returns
    -------
    OpticalWavefront
        Field after phase modulation.
    """
    arr_zeros: Float[Array, " hh ww"] = jnp.zeros_like(
        incoming.field, dtype=jnp.float64
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = _arrayed_grids(arr_zeros, arr_zeros, incoming.dx)
    uu: Num[Array, " hh ww"]
    uu, _ = _rotate_coords(xx, yy, theta)
    phase: Float[Array, " hh ww"]
    phase = depth * jnp.sin(2.0 * jnp.pi * uu / period)
    field_out: Complex[Array, " hh ww"] = add_phase_screen(
        incoming.field, phase
    )
    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def amplitude_grating_binary(
    incoming: OpticalWavefront,
    period: ScalarFloat,
    duty_cycle: Optional[ScalarFloat] = 0.5,
    theta: Optional[ScalarFloat] = 0.0,
    trans_high: Optional[ScalarFloat] = 1.0,
    trans_low: Optional[ScalarFloat] = 0.0,
) -> OpticalWavefront:
    """
    Binary amplitude grating with given duty cycle.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input field.
    period : ScalarFloat
        Period in meters.
    duty_cycle : ScalarFloat, optional
        Fraction of period in 'high' state (0..1), by default 0.5.
    theta : ScalarFloat, optional
        Orientation (radians), by default 0.0.
    trans_high : ScalarFloat, optional
        Amplitude transmittance for 'high' bars, by default 1.0.
    trans_low : ScalarFloat, optional
        Amplitude transmittance for 'low' bars, by default 0.0.

    Returns
    -------
    OpticalWavefront
        Field after amplitude modulation.

    Notes
    -----
    - Compute u along grating direction.
    - Map u modulo period → binary mask via duty cycle.
    - Apply amplitude levels to field.
    """
    arr_zeros: Float[Array, " hh ww"] = jnp.zeros_like(
        incoming.field, dtype=jnp.float64
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = _arrayed_grids(arr_zeros, arr_zeros, incoming.dx)
    uu: Num[Array, " hh ww"]
    uu, _ = _rotate_coords(xx, yy, theta)
    duty: Float[Array, " "] = jnp.clip(duty_cycle, 0.0, 1.0)
    frac: Num[Array, " hh ww"] = (uu / period) - jnp.floor(uu / period)
    mask_high: Bool[Array, " hh ww"] = frac < duty
    tmap: Float[Array, " hh ww"] = jnp.where(
        mask_high, trans_high, trans_low
    ).astype(jnp.float64)
    field_out: Complex[Array, " hh ww"] = incoming.field * tmap
    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def phase_grating_sawtooth(
    incoming: OpticalWavefront,
    period: ScalarFloat,
    depth: ScalarFloat,
    theta: ScalarFloat = 0.0,
) -> OpticalWavefront:
    """
    Sawtooth phase grating with peak-to-peak depth (radians).

    Parameters
    ----------
    incoming : OpticalWavefront
        Input field.
    period : ScalarFloat
        Grating period in meters.
    depth : ScalarFloat
        Phase depth over one period in radians.
    theta : ScalarFloat, optional
        Orientation (radians), by default 0.0.

    Returns
    -------
    grating : OpticalWavefront
        Field after blazed phase modulation.

    Notes
    -----
    - Compute fractional coordinate within each period.
    - Sawtooth phase in [0, depth) → shift to mean-zero if desired
        (kept at [0, depth)).
    - Apply phase with exp(i*phase).
    """
    arr_zeros: Float[Array, " hh ww"] = jnp.zeros_like(
        incoming.field, dtype=jnp.float64
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = _arrayed_grids(arr_zeros, arr_zeros, incoming.dx)
    uu: Num[Array, " hh ww"]
    uu, _ = _rotate_coords(xx, yy, theta)

    frac: Float[Array, " hh ww"] = (uu / period) - jnp.floor(uu / period)
    phase: Float[Array, " hh ww"] = depth * frac
    field_out: Complex[Array, " hh ww"] = add_phase_screen(
        incoming.field, phase
    )
    grating: OpticalWavefront = make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return grating


@jaxtyped(typechecker=beartype)
def apply_phase_mask(
    incoming: OpticalWavefront,
    phase_map: Float[Array, " hh ww"],
) -> OpticalWavefront:
    """
    Apply an arbitrary phase mask (e.g., SLM, turbulence screen).

    Field_out = field_in * exp(i * phase_map).

    Parameters
    ----------
    incoming : OpticalWavefront
        Input field.
    phase_map : Float[Array, " hh ww"]
        Phase in radians, same spatial shape as field.

    Returns
    -------
    masked_wavefront : OpticalWavefront
        Field with added phase.
    """
    field_out: Complex[Array, " hh ww"] = add_phase_screen(
        incoming.field, phase_map
    )
    masked_wavefront: OpticalWavefront = make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return masked_wavefront


@jaxtyped(typechecker=beartype)
def apply_phase_mask_fn(
    incoming: OpticalWavefront,
    phase_fn: Callable[
        [Float[Array, " hh ww"], Float[Array, " hh ww"]],
        Float[Array, " hh ww"],
    ],
) -> OpticalWavefront:
    """
    Build and apply a phase mask from a callable `phase_fn(xx, yy)`.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input field.
    phase_fn : callable
        Function producing a phase map (radians) given
        centered grids xx, yy (meters).

    Returns
    -------
    masked_wavefront : OpticalWavefront
        Field with added phase.
    """
    arr_zeros: Float[Array, " hh ww"] = jnp.zeros_like(
        incoming.field, dtype=jnp.float64
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = _arrayed_grids(arr_zeros, arr_zeros, incoming.dx)
    phase_map: Float[Array, " hh ww"] = phase_fn(xx, yy)
    field_out: Complex[Array, " hh ww"] = add_phase_screen(
        incoming.field, phase_map
    )
    masked_wavefront: OpticalWavefront = make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return masked_wavefront


@jaxtyped(typechecker=beartype)
def polarizer_jones(
    incoming: OpticalWavefront,
    theta: ScalarFloat = 0.0,
) -> OpticalWavefront:
    """
    Linear polarizer at angle `theta` (radians, CCW from x-axis).

    Applied to a 2-component Jones field (ex, ey) stored in the last
    dimension.

    Parameters
    ----------
    incoming : OpticalWavefront
        Field shape must be Complex[H, W, 2].
    theta : ScalarFloat, optional
        Transmission axis angle (radians), by default 0.0.

    Returns
    -------
    OpticalWavefront
        Polarized field with same shape.

    Notes
    -----
    - Jones matrix: P = R(-θ) @ [[1, 0],[0, 0]] @ R(θ).
    - Apply P to [ex, ey] at each pixel.
    """
    field: Complex[Array, " hh ww 2"] = incoming.field
    ct: ScalarFloat = jnp.cos(theta)
    st: ScalarFloat = jnp.sin(theta)
    ex: Complex[Array, " hh ww"] = field[..., 0]
    ey: Complex[Array, " hh ww"] = field[..., 1]
    e_par: Complex[Array, " hh ww"] = (ex * ct) + (ey * st)
    ex_out: Complex[Array, " hh ww"] = e_par * ct
    ey_out: Complex[Array, " hh ww"] = e_par * st
    field_out: Complex[Array, " hh ww 2"] = jnp.stack(
        [ex_out, ey_out], axis=-1
    )
    polar_wavefront: OpticalWavefront = make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
        polarization=True,
    )
    return polar_wavefront


@jaxtyped(typechecker=beartype)
def waveplate_jones(
    incoming: OpticalWavefront,
    delta: ScalarFloat,
    theta: ScalarFloat = 0.0,
) -> OpticalWavefront:
    """
    Waveplate/retarder with retardance `delta` and fast-axis angle
    `theta`.

    Special cases: quarter-wave (delta=π/2), half-wave (delta=π).

    Parameters
    ----------
    incoming : OpticalWavefront
        Field shape must be Complex[H, W, 2].
    delta : ScalarFloat
        Phase delay between fast and slow axes in radians.
    theta : ScalarFloat, optional
        Fast-axis angle (radians, CCW from x), by default 0.0.

    Returns
    -------
    jones_wavefront : OpticalWavefront
        Retarded field with same shape.

    Notes
    -----
    - Jones matrix: J = R(-θ) @ diag(1, e^{iδ}) @ R(θ).
    - Apply J to [ex, ey] per pixel.
    """
    field: Complex[Array, " hh ww 2"] = incoming.field
    ct: Float[Array, " "] = jnp.cos(theta)
    st: Float[Array, " "] = jnp.sin(theta)
    e: Complex[Array, " "] = jnp.exp(1j * delta)
    ex: Complex[Array, " hh ww"]
    ey: Complex[Array, " hh ww"]
    ex, ey = field[..., 0], field[..., 1]
    a: Complex[Array, " hh ww"] = (ct * ct) + (e * st * st)
    b: Complex[Array, " hh ww"] = (1.0 - e) * ct * st
    c: Complex[Array, " hh ww"] = b
    d: Complex[Array, " hh ww"] = (st * st) + (e * ct * ct)
    ex_out: Complex[Array, " hh ww"] = (a * ex) + (b * ey)
    ey_out: Complex[Array, " hh ww"] = (c * ex) + (d * ey)
    field_out: Complex[Array, " hh ww 2"] = jnp.stack(
        [ex_out, ey_out], axis=-1
    )
    jones_wavefront: OpticalWavefront = make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
        polarization=True,
    )
    return jones_wavefront


@jaxtyped(typechecker=beartype)
def nd_filter(
    incoming: OpticalWavefront,
    optical_density: Optional[ScalarFloat] = 0.0,
    transmittance: Optional[ScalarFloat] = -1.0,
) -> OpticalWavefront:
    """
    Neutral density (ND) filter as a uniform amplitude attenuator.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input field.
    optical_density : ScalarFloat, optional
        OD; intensity transmittance T = 10^(-OD).
        If given, overrides `transmittance`. Default is 0.0.
    transmittance : ScalarFloat, optional
        Intensity transmittance T in [0, 1].
        Used if `optical_density` is 0.

    Returns
    -------
    nd_wavefront : OpticalWavefront
        Attenuated wavefront.

    Notes
    -----
    - Determine intensity T from OD or provided T.
    - Amplitude factor a = sqrt(T).
    - Multiply field by a and return.
    """
    tt = cond(
        optical_density != 0,
        lambda: jnp.power(10.0, -jnp.asarray(optical_density)),
        lambda: jnp.clip(jnp.asarray(transmittance), 0.0, 1.0),
    )
    a = jnp.sqrt(tt).astype(incoming.field.real.dtype)
    field_out = incoming.field * a
    nd_wavefront: OpticalWavefront = make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return nd_wavefront


@jaxtyped(typechecker=beartype)
def quarter_waveplate(
    incoming: OpticalWavefront,
    theta: Optional[ScalarFloat] = 0.0,
) -> OpticalWavefront:
    """
    Apply a quarter-wave plate (δ = π/2) with fast-axis angle `theta`.

    Parameters
    ----------
    incoming : OpticalWavefront
        Vector field Complex[H, W, 2] (Jones: ex, ey).
    theta : ScalarFloat, optional
        Fast-axis angle in radians (CCW from x), by default 0.0.

    Returns
    -------
    qw_wavefront : OpticalWavefront
        Retarded field after quarter-wave plate.

    Notes
    -----
    Call `waveplate_jones` with delta = π/2.
    """
    qw_wavefront: OpticalWavefront = waveplate_jones(
        incoming, delta=jnp.pi / 2.0, theta=theta
    )
    return qw_wavefront


@jaxtyped(typechecker=beartype)
def half_waveplate(
    incoming: OpticalWavefront,
    theta: Optional[ScalarFloat] = 0.0,
) -> OpticalWavefront:
    """
    Apply a half-wave plate (δ = π) with fast-axis angle `theta`.

    Parameters
    ----------
    incoming : OpticalWavefront
        Vector field Complex[H, W, 2] (Jones: ex, ey).
    theta : ScalarFloat, optional
        Fast-axis angle in radians (CCW from x), by default 0.0.

    Returns
    -------
    hw_wavefront : OpticalWavefront
        Retarded field after half-wave plate.

    Notes
    -----
    Call `waveplate_jones` with delta = π.
    """
    hw_wavefront: OpticalWavefront = waveplate_jones(
        incoming, delta=jnp.pi, theta=theta
    )
    return hw_wavefront


@jaxtyped(typechecker=beartype)
def phase_grating_blazed_elliptical(
    incoming: OpticalWavefront,
    period_x: ScalarFloat,
    period_y: ScalarFloat,
    theta: Optional[ScalarFloat] = 0.0,
    depth: Optional[ScalarFloat] = 2.0 * jnp.pi,
    two_dim: Optional[ScalarBool] = False,
) -> OpticalWavefront:
    r"""
    Orientation-aware elliptical blazed grating.

    Supports anisotropic periods along rotated axes (x', y')
    and optional 2D blaze.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input scalar wavefront.
    period_x : ScalarFloat
        Blaze period along x' in meters (after rotation by `theta`).
    period_y : ScalarFloat
        Blaze period along y' in meters (after rotation by `theta`).
    theta : ScalarFloat, optional
        Grating orientation angle in radians (CCW from x), by default
        0.0.
    depth : ScalarFloat, optional
        Peak-to-peak phase depth in radians, by default 2π.
    two_dim : bool, optional
        If False (default), apply a 1D blaze along x' only.
        If True, create a 2D blazed lattice using both x' and y'.

    Returns
    -------
    phase_grating_wavefront : OpticalWavefront
        Field after applying the elliptical blazed phase.

    Notes
    -----
    - Build centered grids xx, yy (meters) and rotate → (x', y').
    - Compute fractional coordinates
        ..math::
        fu = frac(x'/period_x)
        fv = frac(y'/period_y)
    - if `two_dim` is True
        ..math::
        phase = depth * frac(fu + fv)
      else,
        ..math::
        phase = depth * fu
    - Multiply by exp(i * phase) and return.
    """
    arr_zeros: Float[Array, " hh ww"] = jnp.zeros_like(
        incoming.field, dtype=jnp.float64
    )
    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    xx, yy = _arrayed_grids(arr_zeros, arr_zeros, incoming.dx)
    uu: Float[Array, " hh ww"]
    vv: Float[Array, " hh ww"]
    uu, vv = _rotate_coords(xx, yy, theta)
    eps: ScalarFloat = 1e-30
    px: ScalarFloat = jnp.where(jnp.abs(period_x) < eps, eps, period_x)
    py: ScalarFloat = jnp.where(jnp.abs(period_y) < eps, eps, period_y)
    fu: Float[Array, " hh ww"] = (uu / px) - jnp.floor(uu / px)
    fv: Float[Array, " hh ww"] = (vv / py) - jnp.floor(vv / py)
    phase: Float[Array, " hh ww"] = cond(
        two_dim,
        lambda: depth * ((fu + fv) - jnp.floor(fu + fv)),
        lambda: depth * fu,
    )
    field_out: Complex[Array, " hh ww"] = add_phase_screen(
        incoming.field, phase
    )
    phase_grating_wavefront: OpticalWavefront = make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return phase_grating_wavefront
