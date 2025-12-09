r"""High-NA vector focusing using Richards-Wolf diffraction integrals.

Extended Summary
----------------
This module implements the Richards-Wolf vector diffraction theory for
computing the focal field of high numerical aperture (NA) optical systems.
Unlike scalar diffraction theory, vector theory correctly accounts for the
polarization rotation that occurs when light is focused by a high-NA lens.

At high NA (> 0.7), three key effects become significant:
1. **Depolarization**: Linear polarization develops a weak longitudinal (Ez)
   component and becomes slightly elliptical at focus.
2. **Radial focusing enhancement**: Radially polarized beams create strong
   Ez at focus, producing a tighter focal spot than linearly polarized light.
3. **Azimuthal donut**: Azimuthally polarized beams produce no Ez and create
   a dark-center "donut" focal spot.

These effects are invisible to scalar diffraction theory (Fresnel, Fraunhofer,
Angular Spectrum) and require full vector treatment.

Routine Listings
----------------
high_na_focus : function
    Compute focal field using Richards-Wolf vector diffraction integrals
debye_wolf_focus : function
    Alternative interface using Debye-Wolf formulation
compute_focal_volume : function
    Compute 3D focal volume at multiple z planes
aplanatic_apodization : function
    Apply √cos(θ) apodization for aplanatic lens systems

Notes
-----
The Richards-Wolf integrals express the focal field as:

.. math::
\\vec{E}(\\rho_f, \\phi_f, z_f) = -\\frac{i k f}{2\\pi} \\int_0^{\\theta_{max}}
\\int_0^{2\\pi} \\sqrt{\\cos\\theta} \\, \\mathbf{P}(\\theta, \\phi)
\\cdot \\vec{E}_{pupil}(\\theta, \\phi) \\,
e^{i k z_f \\cos\\theta} \\, e^{i k \\rho_f \\sin\\theta \\cos(\\phi - \\phi_f)}
\\sin\\theta \\, d\\phi \\, d\\theta

where P(θ,φ) is the polarization rotation matrix that accounts for how the
electric field vector rotates as light refracts through the lens.

References
----------
.. [1] Richards, B., & Wolf, E. (1959). "Electromagnetic diffraction in
       optical systems, II. Structure of the image field in an aplanatic
       system". Proc. R. Soc. Lond. A, 253(1274), 358-379.
.. [2] Youngworth, K. S., & Brown, T. G. (2000). "Focusing of high
       numerical aperture cylindrical-vector beams". Opt. Express, 7(2),
       77-87.
.. [3] Novotny, L., & Hecht, B. (2012). "Principles of Nano-Optics",
       2nd ed. Cambridge University Press. Chapter 3.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple, Union
from jax import lax
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from janssen.optics import create_spatial_grid
from janssen.utils import (
    OpticalWavefront,
    ScalarFloat,
    VectorWavefront3D,
    make_optical_wavefront,
    make_vector_wavefront_3d,
)

jax.config.update("jax_enable_x64", True)

SAFE_DIVIDE_FLOOR: float = 1e-15
POLARIZED_FIELD_NDIM: int = 3
JONES_VECTOR_DIM: int = 2


@jaxtyped(typechecker=beartype)
def _create_pupil_coordinates(
    grid_size: Tuple[int, int],
    dx: ScalarFloat,
    na: ScalarFloat,
) -> Tuple[
    Float[Array, " ny nx"],
    Float[Array, " ny nx"],
    Float[Array, " ny nx"],
    Float[Array, " ny nx"],
    Float[Array, " ny nx"],
    Bool[Array, " ny nx"],
]:
    """Create pupil plane coordinates and angular mappings.

    Parameters
    ----------
    grid_size : Tuple[int, int]
        Grid dimensions (ny, nx).
    dx : ScalarFloat
        Pupil plane pixel spacing in meters.
    na : ScalarFloat
        Numerical aperture of the focusing lens.

    Returns
    -------
    rho_norm : Float[Array, " ny nx"]
        Normalized radial coordinate (0 at center, 1 at NA edge).
    phi : Float[Array, " ny nx"]
        Azimuthal angle in pupil plane.
    sin_theta : Float[Array, " ny nx"]
        sin(θ) where θ is the convergence angle.
    cos_theta : Float[Array, " ny nx"]
        cos(θ) where θ is the convergence angle.
    theta : Float[Array, " ny nx"]
        Convergence angle in radians.
    pupil_mask : Bool[Array, " ny nx"]
        Binary mask for valid pupil region (rho_norm <= 1).

    Notes
    -----
    The pupil radius corresponds to NA for a unit magnification system.
    In a real system, pupil_radius = f * NA where f is focal length.
    Here we normalize so that the edge of the illuminated region corresponds
    to the NA. The sin(θ) = NA * ρ_norm mapping is for an aplanatic system.
    """
    ny, nx = grid_size
    diameter: Float[Array, " 2"] = jnp.asarray(
        [nx * dx, ny * dx], dtype=jnp.float64
    )
    num_points: Int[Array, " 2"] = jnp.asarray([nx, ny], dtype=jnp.int32)
    xx: Float[Array, " ny nx"]
    yy: Float[Array, " ny nx"]
    xx, yy = create_spatial_grid(diameter, num_points)
    rho: Float[Array, " ny nx"] = jnp.sqrt(xx**2 + yy**2)
    pupil_radius: Float[Array, " "] = jnp.max(rho)
    rho_norm: Float[Array, " ny nx"] = rho / (pupil_radius + SAFE_DIVIDE_FLOOR)
    sin_theta: Float[Array, " ny nx"] = na * rho_norm
    sin_theta = jnp.clip(sin_theta, 0.0, 1.0 - SAFE_DIVIDE_FLOOR)
    cos_theta: Float[Array, " ny nx"] = jnp.sqrt(1.0 - sin_theta**2)
    theta: Float[Array, " ny nx"] = jnp.arcsin(sin_theta)
    phi: Float[Array, " ny nx"] = jnp.arctan2(yy, xx)
    pupil_mask: Bool[Array, " ny nx"] = rho_norm <= 1.0
    return rho_norm, phi, sin_theta, cos_theta, theta, pupil_mask


@jaxtyped(typechecker=beartype)
def aplanatic_apodization(
    cos_theta: Float[Array, " ny nx"],
) -> Float[Array, " ny nx"]:
    """Apply aplanatic lens apodization factor.

    For an aplanatic (sine-condition satisfying) lens, the amplitude
    apodization is √cos(θ) where θ is the convergence angle.

    Parameters
    ----------
    cos_theta : Float[Array, " ny nx"]
        Cosine of convergence angle at each pupil point.

    Returns
    -------
    apodization : Float[Array, " ny nx"]
        Apodization factor √cos(θ).

    Notes
    -----
    The √cos(θ) factor arises from energy conservation when mapping
    a uniform pupil plane wave to converging spherical wavefronts.
    This is the standard apodization for microscope objectives.
    """
    apodization: Float[Array, " ny nx"] = jnp.sqrt(
        jnp.maximum(cos_theta, SAFE_DIVIDE_FLOOR)
    )
    return apodization


@jaxtyped(typechecker=beartype)
def _polarization_rotation_matrix(
    sin_theta: Float[Array, " ny nx"],
    cos_theta: Float[Array, " ny nx"],
    phi: Float[Array, " ny nx"],
) -> Tuple[
    Float[Array, " ny nx"],
    Float[Array, " ny nx"],
    Float[Array, " ny nx"],
    Float[Array, " ny nx"],
    Float[Array, " ny nx"],
    Float[Array, " ny nx"],
]:
    """Compute polarization rotation matrix elements for high-NA focusing.

    When light refracts through a high-NA lens, the electric field vector
    must remain transverse to the k-vector. This rotates the polarization
    and generates Ez components.

    Parameters
    ----------
    sin_theta : Float[Array, " ny nx"]
        sin(θ) convergence angle.
    cos_theta : Float[Array, " ny nx"]
        cos(θ) convergence angle.
    phi : Float[Array, " ny nx"]
        Azimuthal angle in pupil.

    Returns
    -------
    p_xx, p_xy, p_yx, p_yy, p_zx, p_zy : Float[Array, " ny nx"]
        Elements of the 3x2 polarization rotation matrix P where:
        [Ex_focal]   [p_xx  p_xy] [Ex_pupil]
        [Ey_focal] = [p_yx  p_yy] [Ey_pupil]
        [Ez_focal]   [p_zx  p_zy]

    Notes
    -----
    The full rotation matrix accounts for:
    1. Rotation into the meridional plane (by -φ)
    2. Tilt of k-vector by angle θ
    3. Rotation back out of meridional plane (by +φ)

    The matrix elements are derived in Novotny & Hecht, Chapter 3:
        p_xx = cos(θ)cos²(φ) + sin²(φ)
        p_xy = (cos(θ) - 1)sin(φ)cos(φ)
        p_yx = (cos(θ) - 1)sin(φ)cos(φ)
        p_yy = cos(θ)sin²(φ) + cos²(φ)
        p_zx = -sin(θ)cos(φ)
        p_zy = -sin(θ)sin(φ)
    """
    cos_phi: Float[Array, " ny nx"] = jnp.cos(phi)
    sin_phi: Float[Array, " ny nx"] = jnp.sin(phi)
    cos_phi_sq: Float[Array, " ny nx"] = cos_phi**2
    sin_phi_sq: Float[Array, " ny nx"] = sin_phi**2
    sin_cos_phi: Float[Array, " ny nx"] = sin_phi * cos_phi
    p_xx: Float[Array, " ny nx"] = cos_theta * cos_phi_sq + sin_phi_sq
    p_xy: Float[Array, " ny nx"] = (cos_theta - 1.0) * sin_cos_phi
    p_yx: Float[Array, " ny nx"] = (cos_theta - 1.0) * sin_cos_phi
    p_yy: Float[Array, " ny nx"] = cos_theta * sin_phi_sq + cos_phi_sq
    p_zx: Float[Array, " ny nx"] = -sin_theta * cos_phi
    p_zy: Float[Array, " ny nx"] = -sin_theta * sin_phi
    return p_xx, p_xy, p_yx, p_yy, p_zx, p_zy


@jaxtyped(typechecker=beartype)
def _apply_polarization_rotation(
    ex_pupil: Complex[Array, " ny nx"],
    ey_pupil: Complex[Array, " ny nx"],
    p_xx: Float[Array, " ny nx"],
    p_xy: Float[Array, " ny nx"],
    p_yx: Float[Array, " ny nx"],
    p_yy: Float[Array, " ny nx"],
    p_zx: Float[Array, " ny nx"],
    p_zy: Float[Array, " ny nx"],
) -> Tuple[
    Complex[Array, " ny nx"],
    Complex[Array, " ny nx"],
    Complex[Array, " ny nx"],
]:
    """Apply polarization rotation to pupil field.

    Parameters
    ----------
    ex_pupil, ey_pupil : Complex[Array, " ny nx"]
        Input pupil field components.
    p_xx, p_xy, p_yx, p_yy, p_zx, p_zy : Float[Array, " ny nx"]
        Polarization rotation matrix elements.

    Returns
    -------
    ex_rot, ey_rot, ez_rot : Complex[Array, " ny nx"]
        Rotated field components ready for focusing integral.
    """
    ex_rot: Complex[Array, " ny nx"] = p_xx * ex_pupil + p_xy * ey_pupil
    ey_rot: Complex[Array, " ny nx"] = p_yx * ex_pupil + p_yy * ey_pupil
    ez_rot: Complex[Array, " ny nx"] = p_zx * ex_pupil + p_zy * ey_pupil
    return ex_rot, ey_rot, ez_rot


@jaxtyped(typechecker=beartype)
def _compute_defocus_phase(
    cos_theta: Float[Array, " ny nx"],
    z_focus: ScalarFloat,
    wavenumber: ScalarFloat,
) -> Complex[Array, " ny nx"]:
    """Compute defocus phase factor for off-focal-plane calculations.

    Parameters
    ----------
    cos_theta : Float[Array, " ny nx"]
        cos(θ) convergence angle.
    z_focus : ScalarFloat
        Axial position relative to focal plane (z=0) in meters.
    wavenumber : ScalarFloat
        Wavenumber k = 2π/λ.

    Returns
    -------
    defocus_phase : Complex[Array, " ny nx"]
        Phase factor exp(i k z cos(θ)).
    """
    defocus_phase: Complex[Array, " ny nx"] = jnp.exp(
        1j * wavenumber * z_focus * cos_theta
    )
    return defocus_phase


@jaxtyped(typechecker=beartype)
def high_na_focus(
    pupil_field: OpticalWavefront,
    na: ScalarFloat,
    focal_length: ScalarFloat,
    z_focus: ScalarFloat = 0.0,
    output_dx: Union[ScalarFloat, None] = None,
    output_grid_size: Union[Tuple[int, int], Int[Array, " 2"], None] = None,
    refractive_index: ScalarFloat = 1.0,
    include_aplanatic_factor: bool = True,
) -> VectorWavefront3D:
    """Compute focal field using Richards-Wolf vector diffraction integrals.

    This is the main entry point for high-NA vector focusing simulations.
    It takes a polarized pupil field and computes the full 3D vector field
    (Ex, Ey, Ez) at the focal plane.

    Parameters
    ----------
    pupil_field : OpticalWavefront
        Input field in the pupil plane. Must be polarized with shape
        (H, W, 2) containing [Ex, Ey] Jones components.
    na : ScalarFloat
        Numerical aperture of the focusing lens.
    focal_length : ScalarFloat
        Focal length of the lens in meters.
    z_focus : ScalarFloat, optional
        Axial position relative to geometric focus in meters.
        z_focus = 0 gives the focal plane. Default is 0.0.
    output_dx : ScalarFloat, optional
        Output pixel size in focal plane. If None, computed from
        diffraction limit: dx ≈ λ/(2*NA) / 4.
    output_grid_size : Tuple[int, int], optional
        Output grid size. If None, uses same size as input.
    refractive_index : ScalarFloat, optional
        Refractive index of focal medium. Default is 1.0 (air).
    include_aplanatic_factor : bool, optional
        Whether to include √cos(θ) apodization. Default is True.

    Returns
    -------
    focal_field : VectorWavefront3D
        Vector field at focal plane with shape (H, W, 3) containing
        [Ex, Ey, Ez] components.

    Notes
    -----
    The algorithm:
    1. Create pupil coordinates and map to convergence angles
    2. Apply aplanatic apodization √cos(θ)
    3. Compute polarization rotation matrix P(θ, φ)
    4. Apply rotation to get (Ex', Ey', Ez') in focal coordinates
    5. Apply defocus phase exp(ikz cos(θ))
    6. Use 2D FFT to evaluate the focusing integral
    7. Package result as VectorWavefront3D

    For the focal plane (z=0), the integral reduces to a 2D Fourier
    transform relationship, enabling efficient FFT-based computation.

    Examples
    --------
    >>> from janssen.models import radially_polarized_beam
    >>> from janssen.prop import high_na_focus
    >>>
    >>> # Create radially polarized beam in pupil
    >>> pupil = radially_polarized_beam(
    ...     wavelength=633e-9,
    ...     dx=10e-6,
    ...     grid_size=(256, 256),
    ...     beam_radius=1e-3,
    ... )
    >>>
    >>> # Focus with high-NA lens
    >>> focal = high_na_focus(
    ...     pupil_field=pupil,
    ...     na=0.9,
    ...     focal_length=3e-3,
    ... )
    >>>
    >>> print(f"Ez peak: {jnp.max(jnp.abs(focal.ez)**2):.3e}")
    """
    ny, nx = pupil_field.field.shape[:2]
    wavelength: Float[Array, " "] = pupil_field.wavelength
    wavenumber: Float[Array, " "] = (
        2.0 * jnp.pi * refractive_index / wavelength
    )
    out_ny, out_nx = lax.cond(
        output_grid_size is None,
        lambda: (ny, nx),
        lambda: (
            int(jnp.asarray(output_grid_size, dtype=jnp.int32)[0]),
            int(jnp.asarray(output_grid_size, dtype=jnp.int32)[1]),
        ),
    )
    output_dx = lax.cond(
        output_dx is None,
        lambda: wavelength / (8.0 * na),
        lambda: jnp.asarray(output_dx, dtype=jnp.float64),
    )
    (
        _,
        phi,
        sin_theta,
        cos_theta,
        _,
        pupil_mask,
    ) = _create_pupil_coordinates(
        grid_size=(ny, nx),
        dx=pupil_field.dx,
        na=na,
    )
    ex_pupil: Complex[Array, " ny nx"] = pupil_field.field[:, :, 0]
    ey_pupil: Complex[Array, " ny nx"] = pupil_field.field[:, :, 1]
    ex_pupil = ex_pupil * pupil_mask
    ey_pupil = ey_pupil * pupil_mask
    apod: Float[Array, " ny nx"] = lax.cond(
        include_aplanatic_factor,
        lambda: aplanatic_apodization(cos_theta),
        lambda: jnp.ones_like(cos_theta),
    )
    ex_pupil = ex_pupil * apod
    ey_pupil = ey_pupil * apod
    p_xx, p_xy, p_yx, p_yy, p_zx, p_zy = _polarization_rotation_matrix(
        sin_theta, cos_theta, phi
    )
    ex_rot, ey_rot, ez_rot = _apply_polarization_rotation(
        ex_pupil, ey_pupil, p_xx, p_xy, p_yx, p_yy, p_zx, p_zy
    )
    defocus_phase: Complex[Array, " ny nx"] = _compute_defocus_phase(
        cos_theta, z_focus, wavenumber
    )
    ex_rot = ex_rot * defocus_phase
    ey_rot = ey_rot * defocus_phase
    ez_rot = ez_rot * defocus_phase
    jacobian: Float[Array, " ny nx"] = sin_theta
    ex_integrand: Complex[Array, " ny nx"] = ex_rot * jacobian
    ey_integrand: Complex[Array, " ny nx"] = ey_rot * jacobian
    ez_integrand: Complex[Array, " ny nx"] = ez_rot * jacobian
    ex_focal: Complex[Array, " ny nx"] = jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.ifftshift(ex_integrand))
    )
    ey_focal: Complex[Array, " ny nx"] = jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.ifftshift(ey_integrand))
    )
    ez_focal: Complex[Array, " ny nx"] = jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.ifftshift(ez_integrand))
    )
    scale_factor: Complex[Array, " "] = (
        -1j
        * wavenumber
        * focal_length
        / (2.0 * jnp.pi)
        * (pupil_field.dx**2)
    )
    ex_focal = ex_focal * scale_factor
    ey_focal = ey_focal * scale_factor
    ez_focal = ez_focal * scale_factor
    focal_dx: Float[Array, " "] = (
        wavelength * focal_length / (nx * pupil_field.dx * na)
    )
    actual_dx: Float[Array, " "] = focal_dx
    field_3d: Complex[Array, " ny nx 3"] = jnp.stack(
        [ex_focal, ey_focal, ez_focal], axis=-1
    )
    focal_field: VectorWavefront3D = make_vector_wavefront_3d(
        field=field_3d,
        wavelength=wavelength,
        dx=actual_dx,
        z_position=z_focus,
    )
    return focal_field


@jaxtyped(typechecker=beartype)
def debye_wolf_focus(
    pupil_field: OpticalWavefront,
    na: ScalarFloat,
    focal_length: ScalarFloat,
    z_focus: ScalarFloat = 0.0,
    refractive_index: ScalarFloat = 1.0,
) -> VectorWavefront3D:
    """Compute focal field using Debye-Wolf formulation.

    This is an alias for `high_na_focus` using the alternative naming
    convention from the Debye approximation literature.

    Parameters
    ----------
    pupil_field : OpticalWavefront
        Input polarized field in pupil plane.
    na : ScalarFloat
        Numerical aperture.
    focal_length : ScalarFloat
        Focal length in meters.
    z_focus : ScalarFloat, optional
        Axial position relative to focus. Default is 0.0.
    refractive_index : ScalarFloat, optional
        Refractive index of focal medium. Default is 1.0.

    Returns
    -------
    focal_field : VectorWavefront3D
        Vector field at focal plane.

    See Also
    --------
    high_na_focus : Main implementation with additional options.
    """
    return high_na_focus(
        pupil_field=pupil_field,
        na=na,
        focal_length=focal_length,
        z_focus=z_focus,
        refractive_index=refractive_index,
    )


@jaxtyped(typechecker=beartype)
def compute_focal_volume(
    pupil_field: OpticalWavefront,
    na: ScalarFloat,
    focal_length: ScalarFloat,
    z_positions: Float[Array, " nz"],
    refractive_index: ScalarFloat = 1.0,
) -> Tuple[Complex[Array, " nz ny nx 3"], Float[Array, " "]]:
    """Compute 3D focal volume at multiple z planes.

    Uses vmap to efficiently compute the focal field at multiple
    axial positions.

    Parameters
    ----------
    pupil_field : OpticalWavefront
        Input polarized field in pupil plane.
    na : ScalarFloat
        Numerical aperture.
    focal_length : ScalarFloat
        Focal length in meters.
    z_positions : Float[Array, " nz"]
        Array of axial positions relative to focus.
    refractive_index : ScalarFloat, optional
        Refractive index of focal medium. Default is 1.0.

    Returns
    -------
    focal_volume : Complex[Array, " nz ny nx 3"]
        3D vector field volume with [Ex, Ey, Ez] at each z.
    dx_focal : Float[Array, " "]
        Transverse pixel size at focal plane.

    Examples
    --------
    >>> z_range = jnp.linspace(-2e-6, 2e-6, 41)  # ±2 μm
    >>> volume, dx = compute_focal_volume(
    ...     pupil, na=0.9, focal_length=3e-3, z_positions=z_range
    ... )
    >>> # volume has shape (41, ny, nx, 3)
    """

    def focus_at_z(z: Float[Array, " "]) -> Complex[Array, " ny nx 3"]:
        result = high_na_focus(
            pupil_field=pupil_field,
            na=na,
            focal_length=focal_length,
            z_focus=z,
            refractive_index=refractive_index,
        )
        return result.field

    first_result = high_na_focus(
        pupil_field=pupil_field,
        na=na,
        focal_length=focal_length,
        z_focus=z_positions[0],
        refractive_index=refractive_index,
    )
    dx_focal: Float[Array, " "] = first_result.dx
    focal_volume: Complex[Array, " nz ny nx 3"] = jax.vmap(focus_at_z)(
        z_positions
    )
    return focal_volume, dx_focal


@jaxtyped(typechecker=beartype)
def scalar_focus_for_comparison(
    pupil_field: OpticalWavefront,
    focal_length: ScalarFloat,
    z_focus: ScalarFloat = 0.0,
) -> OpticalWavefront:
    """Compute scalar focal field for comparison with vector result.

    This function ignores polarization and computes a simple Fourier
    transform focal field, demonstrating what scalar theory predicts.
    Useful for comparing with vector results to highlight the differences.

    Parameters
    ----------
    pupil_field : OpticalWavefront
        Input field (polarization ignored, uses total amplitude).
    focal_length : ScalarFloat
        Focal length in meters.
    z_focus : ScalarFloat, optional
        Axial position. Default is 0.0.

    Returns
    -------
    scalar_focal : OpticalWavefront
        Scalar focal field (Ez effects not modeled).

    Notes
    -----
    The scalar approximation:
    - Ignores polarization rotation (no Ez generation)
    - Predicts identical PSF for all input polarizations
    - Fails at high NA where vector effects are significant
    """
    ny, nx = pupil_field.field.shape[:2]
    wavelength = pupil_field.wavelength

    def polarized_to_scalar() -> Complex[Array, " ny nx"]:
        total_field: Complex[Array, " ny nx"] = jnp.sqrt(
            jnp.abs(pupil_field.field[:, :, 0]) ** 2
            + jnp.abs(pupil_field.field[:, :, 1]) ** 2
        )
        phase: Float[Array, " ny nx"] = jnp.angle(pupil_field.field[:, :, 0])
        return total_field * jnp.exp(1j * phase)

    def scalar_passthrough() -> Complex[Array, " ny nx"]:
        return pupil_field.field

    scalar_pupil: Complex[Array, " ny nx"] = lax.cond(
        pupil_field.field.ndim == POLARIZED_FIELD_NDIM,
        polarized_to_scalar,
        scalar_passthrough,
    )
    focal_field: Complex[Array, " ny nx"] = jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.ifftshift(scalar_pupil))
    )
    focal_dx: Float[Array, " "] = (
        wavelength * focal_length / (nx * pupil_field.dx)
    )
    scalar_focal: OpticalWavefront = make_optical_wavefront(
        field=focal_field,
        wavelength=wavelength,
        dx=focal_dx,
        z_position=z_focus,
        polarization=False,
    )
    return scalar_focal