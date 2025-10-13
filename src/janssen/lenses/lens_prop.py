"""Lens propagation functions.

Extended Summary
----------------
Optical field propagation methods based on scalar diffraction theory.
Implements various propagation algorithms including angular spectrum,
Fresnel, and Fraunhofer propagation methods for simulating light
propagation in optical systems.

Routine Listings
----------------
angular_spectrum_prop : function
    Propagates a complex optical field using the angular spectrum method
fresnel_prop : function
    Propagates a complex optical field using the Fresnel approximation
fraunhofer_prop : function
    Propagates a complex optical field using the Fraunhofer
    approximation
digital_zoom : function
    Zooms an optical wavefront by a specified factor
optical_zoom : function
    Modifies the calibration of an optical wavefront without changing
    its field
lens_propagation : function
    Propagates an optical wavefront through a lens

Notes
-----
All propagation methods are implemented using FFT-based algorithms for
efficiency. The choice of propagation method depends on the Fresnel
number
and the specific requirements of the simulation.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Bool, Complex, Float, Integer, jaxtyped

from janssen.utils import (
    LensParams,
    OpticalWavefront,
    make_optical_wavefront,
    scalar_float,
    scalar_integer,
    scalar_numeric,
)

from .lens_elements import create_lens_phase

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def angular_spectrum_prop(
    incoming: OpticalWavefront,
    z_move: scalar_numeric,
    refractive_index: Optional[scalar_numeric] = 1.0,
) -> OpticalWavefront:
    """Propagate a complex field using the angular spectrum method.

    Parameters
    ----------
    incoming : OpticalWavefront
        PyTree with the following parameters:

        field : Complex[Array, " hh ww"]
            Input complex field
        wavelength : Float[Array, " "]
            Wavelength of light in meters
        dx : Float[Array, " "]
            Grid spacing in meters
        z_position : Float[Array, " "]
            Wave front position in meters
    z_move : scalar_numeric
        Propagation distance in meters
        This is in free space.
    refractive_index : Optional[scalar_numeric], optional
        Index of refraction of the medium. Default is 1.0 (vacuum).

    Returns
    -------
    propagated : OpticalWavefront
        Propagated wave front

    Notes
    -----
    Algorithm:

    - Get the shape of the input field
    - Calculate the wavenumber
    - Compute the path length
    - Create spatial frequency coordinates
    - Compute the squared spatial frequencies
    - Angular spectrum transfer function
    - Ensure evanescent waves are properly handled
    - Fourier transform of the input field
    - Apply the transfer function in the Fourier domain
    - Inverse Fourier transform to get the propagated field
    - Return the propagated field
    """
    ny: scalar_integer = incoming.field.shape[0]
    nx: scalar_integer = incoming.field.shape[1]
    wavenumber: Float[Array, " "] = 2 * jnp.pi / incoming.wavelength
    path_length = refractive_index * z_move
    fx: Float[Array, " hh"] = jnp.fft.fftfreq(nx, d=incoming.dx)
    fy: Float[Array, " ww"] = jnp.fft.fftfreq(ny, d=incoming.dx)
    fx_mesh: Float[Array, " hh ww"]
    fy_mesh: Float[Array, " hh ww"]
    fx_mesh, fy_mesh = jnp.meshgrid(fx, fy)
    fsq_mesh: Float[Array, " hh ww"] = (fx_mesh**2) + (fy_mesh**2)
    asp_transfer: Complex[Array, " "] = jnp.exp(
        1j
        * wavenumber
        * path_length
        * jnp.sqrt(1 - (incoming.wavelength**2) * fsq_mesh),
    )
    evanescent_mask: Bool[Array, " hh ww"] = (
        1 / incoming.wavelength
    ) ** 2 >= fsq_mesh
    h_mask: Complex[Array, " hh ww"] = asp_transfer * evanescent_mask
    field_ft: Complex[Array, " hh ww"] = jnp.fft.fft2(incoming.field)
    propagated_ft: Complex[Array, " hh ww"] = field_ft * h_mask
    propagated_field: Complex[Array, " hh ww"] = jnp.fft.ifft2(propagated_ft)
    propagated: OpticalWavefront = make_optical_wavefront(
        field=propagated_field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position + path_length,
    )
    return propagated


@jaxtyped(typechecker=beartype)
def fresnel_prop(
    incoming: OpticalWavefront,
    z_move: scalar_numeric,
    refractive_index: Optional[scalar_numeric] = 1.0,
) -> OpticalWavefront:
    """Propagate a complex field using the Fresnel approximation.

    Parameters
    ----------
    incoming : OpticalWavefront
        PyTree with the following parameters:
        field : Complex[Array, " hh ww"]
            Input complex field
        wavelength : Float[Array, " "]
            Wavelength of light in meters
        dx : Float[Array, " "]
            Grid spacing in meters
        z_position : Float[Array, " "]
            Wave front position in meters
    z_move : scalar_numeric
        Propagation distance in meters
        This is in free space.
    refractive_index : Optional[scalar_numeric], optional
        Index of refraction of the medium. Default is 1.0 (vacuum).

    Returns
    -------
    propagated : OpticalWavefront
        Propagated wave front

    Notes
    -----
    Algorithm:

    - Calculate the wavenumber
    - Create spatial coordinates
    - Quadratic phase factor for Fresnel approximation
        (pre-free-space propagation)
    - Apply quadratic phase to the input field
    - Compute Fourier transform of the input field
    - Compute spatial frequency coordinates
    - Transfer function for Fresnel propagation
    - Apply the transfer function in the Fourier domain
    - Inverse Fourier transform to get the propagated field
    - Final quadratic phase factor (post-free-space propagation)
    - Apply final quadratic phase factor
    - Return the propagated field
    """
    ny: scalar_integer = incoming.field.shape[0]
    nx: scalar_integer = incoming.field.shape[1]
    k: Float[Array, " "] = (2 * jnp.pi) / incoming.wavelength
    x: Float[Array, " hh"] = jnp.arange(-nx // 2, nx // 2) * incoming.dx
    y: Float[Array, " ww"] = jnp.arange(-ny // 2, ny // 2) * incoming.dx
    x_mesh: Float[Array, " hh ww"]
    y_mesh: Float[Array, " hh ww"]
    x_mesh, y_mesh = jnp.meshgrid(x, y)
    path_length = refractive_index * z_move
    quadratic_phase: Float[Array, " hh ww"] = (
        k / (2 * path_length) * (x_mesh**2 + y_mesh**2)
    )
    field_with_phase: Complex[Array, " hh ww"] = incoming.field * jnp.exp(
        1j * quadratic_phase
    )
    field_ft: Complex[Array, " hh ww"] = jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.ifftshift(field_with_phase)),
    )
    fx: Float[Array, " hh"] = jnp.fft.fftfreq(nx, d=incoming.dx)
    fy: Float[Array, " ww"] = jnp.fft.fftfreq(ny, d=incoming.dx)
    fx_mesh: Float[Array, " hh ww"]
    fy_mesh: Float[Array, " hh ww"]
    fx_mesh, fy_mesh = jnp.meshgrid(fx, fy)
    transfer_phase: Float[Array, " hh ww"] = (
        (-1)
        * jnp.pi
        * incoming.wavelength
        * path_length
        * (fx_mesh**2 + fy_mesh**2)
    )
    propagated_ft: Complex[Array, " hh ww"] = field_ft * jnp.exp(
        1j * transfer_phase
    )
    propagated_field: Complex[Array, " hh ww"] = jnp.fft.fftshift(
        jnp.fft.ifft2(jnp.fft.ifftshift(propagated_ft)),
    )
    final_quadratic_phase: Float[Array, " hh ww"] = (
        k / (2 * path_length) * (x_mesh**2 + y_mesh**2)
    )
    final_propagated_field: Complex[Array, " hh ww"] = jnp.fft.ifftshift(
        propagated_field * jnp.exp(1j * final_quadratic_phase),
    )
    propagated: OpticalWavefront = make_optical_wavefront(
        field=final_propagated_field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position + path_length,
    )
    return propagated


@jaxtyped(typechecker=beartype)
def fraunhofer_prop(
    incoming: OpticalWavefront,
    z_move: scalar_float,
    refractive_index: Optional[scalar_float] = 1.0,
) -> OpticalWavefront:
    """Propagate a complex field using the Fraunhofer approximation.

    Parameters
    ----------
    incoming : OpticalWavefront
        PyTree with the following parameters:

        field : Complex[Array, " hh ww"]
            Input complex field
        wavelength : Float[Array, " "]
            Wavelength of light in meters
        dx : Float[Array, " "]
            Grid spacing in meters
        z_position : Float[Array, " "]
            Wave front position in meters
    z_move : scalar_float
        Propagation distance in meters.
        This is in free space.
    refractive_index : scalar_float, optional
        Index of refraction of the medium. Default is 1.0 (vacuum).

    Returns
    -------
    propagated : OpticalWavefront
        Propagated wave front

    Notes
    -----
    Algorithm:

    - Get the shape of the input field
    - Calculate the spatial frequency coordinates
    - Create the meshgrid of spatial frequencies
    - Compute the transfer function for Fraunhofer propagation
    - Compute the Fourier transform of the input field
    - Apply the transfer function in the Fourier domain
    - Inverse Fourier transform to get the propagated field
    - Return the propagated field
    """
    ny: scalar_integer = incoming.field.shape[0]
    nx: scalar_integer = incoming.field.shape[1]
    fx: Float[Array, " hh"] = jnp.fft.fftfreq(nx, d=incoming.dx)
    fy: Float[Array, " ww"] = jnp.fft.fftfreq(ny, d=incoming.dx)
    fx_mesh: Float[Array, " hh ww"]
    fy_mesh: Float[Array, " hh ww"]
    fx_mesh, fy_mesh = jnp.meshgrid(fx, fy)
    path_length = refractive_index * z_move
    hh: Complex[Array, " hh ww"] = jnp.exp(
        -1j
        * jnp.pi
        * incoming.wavelength
        * path_length
        * (fx_mesh**2 + fy_mesh**2),
    ) / (1j * incoming.wavelength * path_length)
    field_ft: Complex[Array, " hh ww"] = jnp.fft.fft2(incoming.field)
    propagated_ft: Complex[Array, " hh ww"] = field_ft * hh
    propagated_field: Complex[Array, " hh ww"] = jnp.fft.ifft2(propagated_ft)
    propagated: OpticalWavefront = make_optical_wavefront(
        field=propagated_field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position + path_length,
    )
    return propagated


@jaxtyped(typechecker=beartype)
def digital_zoom(
    wavefront: OpticalWavefront,
    zoom_factor: scalar_numeric,
) -> OpticalWavefront:
    """Zoom an optical wavefront by a specified factor.

    Key is this returns the same sized array as the original wavefront.

    Parameters
    ----------
    wavefront : OpticalWavefront
        Incoming optical wavefront.
    zoom_factor : scalar_numeric
        Zoom factor (greater than 1 to zoom in, less than 1 to zoom
        out).

    Returns
    -------
    zoomed_wavefront : OpticalWavefront
        Zoomed optical wavefront of the same spatial dimensions.

    Notes
    -----
    Algorithm:

    For zoom in (zoom_factor >= 1.0):
    - Calculate the crop fraction (1 / zoom_factor) to determine the
        central region to extract
    - Create interpolation coordinates for the zoomed region centered
        on the image
    - Use scipy.ndimage.map_coordinates with bilinear interpolation
        to sample the field
    - Return the zoomed field with adjusted pixel size (dx /
    zoom_factor)

    For zoom out (zoom_factor < 1.0):
    - Calculate the shrink fraction (zoom_factor) to determine the
        final image size
    - Create a coordinate mapping from the full image to the shrunken
    region
    - Use scipy.ndimage.map_coordinates to interpolate the original
    field
    - Apply a mask to zero out regions outside the shrunken
        area (padding effect)
    - Return the zoomed field with adjusted pixel size (dx /
    zoom_factor)
    """
    epsilon: Float[Array, " "] = 1e-10
    zoom_factor: Float[Array, " "] = jnp.maximum(zoom_factor, epsilon)
    hh: int
    ww: int
    hh, ww = wavefront.field.shape

    def zoom_in_fn() -> Complex[Array, " hh ww"]:
        crop_fraction: Float[Array, " "] = 1.0 / zoom_factor
        center_y: Float[Array, " "] = (hh - 1) / 2
        center_x: Float[Array, " "] = (ww - 1) / 2
        half_crop_y: Float[Array, " "] = (hh * crop_fraction) / 2
        half_crop_x: Float[Array, " "] = (ww * crop_fraction) / 2
        y_interp: Float[Array, " hh"] = jnp.linspace(
            center_y - half_crop_y, center_y + half_crop_y, hh
        )
        x_interp: Float[Array, " ww"] = jnp.linspace(
            center_x - half_crop_x, center_x + half_crop_x, ww
        )
        y_grid: Float[Array, " hh ww"]
        x_grid: Float[Array, " hh ww"]
        y_grid, x_grid = jnp.meshgrid(y_interp, x_interp, indexing="ij")
        zoomed: Complex[Array, " hh ww"] = jax.scipy.ndimage.map_coordinates(
            wavefront.field.real,
            [y_grid, x_grid],
            order=1,
            mode="constant",
            cval=0.0,
        ) + 1j * jax.scipy.ndimage.map_coordinates(
            wavefront.field.imag,
            [y_grid, x_grid],
            order=1,
            mode="constant",
            cval=0.0,
        )
        return zoomed

    def zoom_out_fn() -> Complex[Array, " hh ww"]:
        shrink_fraction: Float[Array, " "] = zoom_factor
        shrunk_h: Integer[Array, " "] = jnp.round(hh * shrink_fraction).astype(
            jnp.int32
        )
        shrunk_w: Integer[Array, " "] = jnp.round(ww * shrink_fraction).astype(
            jnp.int32
        )
        shrunk_h: Integer[Array, " "] = jnp.minimum(shrunk_h, hh)
        shrunk_w: Integer[Array, " "] = jnp.minimum(shrunk_w, ww)
        center_y: Float[Array, " "] = (hh - 1) / 2
        center_x: Float[Array, " "] = (ww - 1) / 2
        half_shrunk_y: Float[Array, " "] = shrunk_h / 2
        half_shrunk_x: Float[Array, " "] = shrunk_w / 2
        y_coords: Float[Array, " hh"] = jnp.linspace(0, hh - 1, hh)
        x_coords: Float[Array, " ww"] = jnp.linspace(0, ww - 1, ww)

        def get_interp_coord(
            coord: Float[Array, " "],
            center: Float[Array, " "],
            half_size: Float[Array, " "],
            full_size: Integer[Array, " "],
        ) -> Float[Array, " "]:
            norm_coord: Float[Array, " "] = (coord - (center - half_size)) / (
                2 * half_size
            )
            return norm_coord * (full_size - 1)

        y_grid: Float[Array, " hh ww"]
        x_grid: Float[Array, " hh ww"]
        y_grid, x_grid = jnp.meshgrid(y_coords, x_coords, indexing="ij")
        mask: Bool[Array, " hh ww"] = (
            jnp.abs(y_grid - center_y) <= half_shrunk_y
        ) & (jnp.abs(x_grid - center_x) <= half_shrunk_x)
        y_interp: Float[Array, " hh ww"] = get_interp_coord(
            y_grid, center_y, half_shrunk_y, hh
        )
        x_interp: Float[Array, " hh ww"] = get_interp_coord(
            x_grid, center_x, half_shrunk_x, ww
        )
        zoomed_real: Float[Array, " hh ww"] = (
            jax.scipy.ndimage.map_coordinates(
                wavefront.field.real,
                [y_interp, x_interp],
                order=1,
                mode="constant",
                cval=0.0,
            )
        )
        zoomed_imag: Float[Array, " hh ww"] = (
            jax.scipy.ndimage.map_coordinates(
                wavefront.field.imag,
                [y_interp, x_interp],
                order=1,
                mode="constant",
                cval=0.0,
            )
        )
        zoomed: Complex[Array, " hh ww"] = (
            zoomed_real + 1j * zoomed_imag
        ) * mask
        return zoomed

    zoomed_field: Complex[Array, " hh ww"] = jax.lax.cond(
        zoom_factor >= 1.0,
        zoom_in_fn,
        zoom_out_fn,
    )

    zoomed_wavefront: OpticalWavefront = make_optical_wavefront(
        field=zoomed_field,
        wavelength=wavefront.wavelength,
        dx=wavefront.dx / zoom_factor,
        z_position=wavefront.z_position,
    )
    return zoomed_wavefront


@jaxtyped(typechecker=beartype)
def optical_zoom(
    wavefront: OpticalWavefront,
    zoom_factor: scalar_numeric,
) -> OpticalWavefront:
    """Modify the calibration of an optical wavefront without changing
    field.

    Parameters
    ----------
    wavefront : OpticalWavefront
        Incoming optical wavefront.
    zoom_factor : scalar_numeric
        Zoom factor (greater than 1 to zoom in, less than 1 to zoom
        out).

    Returns
    -------
    zoomed_wavefront : OpticalWavefront
        Zoomed optical wavefront of the same spatial dimensions.
    """
    new_dx = wavefront.dx * zoom_factor
    zoomed_wavefront: OpticalWavefront = make_optical_wavefront(
        field=wavefront.field,
        wavelength=wavefront.wavelength,
        dx=new_dx,
        z_position=wavefront.z_position,
    )
    return zoomed_wavefront


@jaxtyped(typechecker=beartype)
def lens_propagation(
    incoming: OpticalWavefront, lens: LensParams
) -> OpticalWavefront:
    """Propagate an optical wavefront through a lens.

    The lens is modeled as a thin lens with a given focal length and
    diameter.

    Parameters
    ----------
    incoming : OpticalWavefront
        The incoming optical wavefront
    lens : LensParams
        The lens parameters including focal length and diameter

    Returns
    -------
    outgoing : OpticalWavefront
        The propagated optical wavefront after passing through the lens

    Notes
    -----
    Algorithm:

    - Create a meshgrid of coordinates based on the incoming wavefront's
        shape and pixel size.
    - Calculate the phase profile and transmission function of the lens.
    - Apply the phase screen to the incoming wavefront's field.
    - Return the new optical wavefront with the updated field,
    wavelength,
        and pixel size.
    """
    hh: int
    ww: int
    hh, ww = incoming.field.shape
    xline: Float[Array, " ww"] = (
        jnp.linspace(-ww // 2, ww // 2 - 1, ww) * incoming.dx
    )
    yline: Float[Array, " hh"] = (
        jnp.linspace(-hh // 2, hh // 2 - 1, hh) * incoming.dx
    )
    xarr: Float[Array, " hh ww"]
    yarr: Float[Array, " hh ww"]
    xarr, yarr = jnp.meshgrid(xline, yline)
    phase_profile: Float[Array, " hh ww"]
    transmission: Float[Array, " hh ww"]
    phase_profile, transmission = create_lens_phase(
        xarr, yarr, lens, incoming.wavelength
    )
    transmitted_field: Complex[Array, " hh ww"] = (
        incoming.field * transmission * jnp.exp(1j * phase_profile)
    )
    outgoing: OpticalWavefront = make_optical_wavefront(
        field=transmitted_field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return outgoing


@jaxtyped(typechecker=beartype)
def correct_propagator(
    incoming: OpticalWavefront,
    z_move: scalar_numeric,
    refractive_index: Optional[scalar_numeric] = 1.0,
) -> OpticalWavefront:
    """Automatically select and apply the most appropriate propagator.

    This function selects the optimal propagation method based on the
    Fresnel number and sampling criteria. It uses:
    - Angular spectrum for very short distances or high spatial frequencies
    - Fresnel propagation for intermediate distances
    - Fraunhofer propagation for far-field distances

    Parameters
    ----------
    incoming : OpticalWavefront
        PyTree with the following parameters:
        field : Complex[Array, " hh ww"]
            Input complex field
        wavelength : Float[Array, " "]
            Wavelength of light in meters
        dx : Float[Array, " "]
            Grid spacing in meters
        z_position : Float[Array, " "]
            Wave front position in meters
    z_move : scalar_numeric
        Propagation distance in meters (in free space)
    refractive_index : Optional[scalar_numeric], optional
        Index of refraction of the medium. Default is 1.0 (vacuum)

    Returns
    -------
    propagated : OpticalWavefront
        Propagated wave front using the most appropriate method

    Notes
    -----
    Selection criteria:
    1. Calculate the characteristic aperture size from the field
    2. Compute the Fresnel number F = a²/(λz)
    3. Check sampling criteria for each method
    4. Select propagator based on:
       - F >> 1 and short distance: Angular spectrum (most accurate)
       - F > 0.1: Fresnel propagation
       - F < 0.1: Fraunhofer propagation (far-field)

    The angular spectrum method is preferred when applicable as it
    makes no paraxial approximations.
    """
    # Get field dimensions
    ny: scalar_integer = incoming.field.shape[0]
    nx: scalar_integer = incoming.field.shape[1]

    # Calculate effective aperture size from field extent
    # Use the RMS width of the field intensity as characteristic size
    field_intensity: Float[Array, " hh ww"] = jnp.abs(incoming.field) ** 2
    total_intensity: Float[Array, " "] = jnp.sum(field_intensity)

    # Create coordinate arrays
    y_coords: Float[Array, " hh"] = (jnp.arange(ny) - ny / 2) * incoming.dx
    x_coords: Float[Array, " ww"] = (jnp.arange(nx) - nx / 2) * incoming.dx
    y_mesh: Float[Array, " hh ww"]
    x_mesh: Float[Array, " hh ww"]
    y_mesh, x_mesh = jnp.meshgrid(y_coords, x_coords, indexing="ij")

    # Calculate RMS width as characteristic aperture size
    x_rms: Float[Array, " "] = jnp.sqrt(
        jnp.sum(field_intensity * x_mesh**2) / (total_intensity + 1e-10)
    )
    y_rms: Float[Array, " "] = jnp.sqrt(
        jnp.sum(field_intensity * y_mesh**2) / (total_intensity + 1e-10)
    )

    # Use the larger RMS as characteristic aperture size
    aperture_size: Float[Array, " "] = (
        jnp.maximum(x_rms, y_rms) * 2
    )  # 2*RMS for full width

    # Account for refractive index in propagation
    path_length: Float[Array, " "] = refractive_index * z_move

    # Calculate Fresnel number
    fresnel_number: Float[Array, " "] = aperture_size**2 / (
        incoming.wavelength * jnp.abs(path_length)
    )

    # Calculate sampling criteria
    # For angular spectrum: need dx << z*λ/L where L is the field size
    field_size: Float[Array, " "] = jnp.maximum(
        nx * incoming.dx, ny * incoming.dx
    )
    angular_spectrum_valid: Bool[Array, " "] = (
        incoming.dx
        < 0.5 * jnp.abs(path_length) * incoming.wavelength / field_size
    )

    # Define the three propagator functions with matching signatures
    def use_angular_spectrum() -> OpticalWavefront:
        return angular_spectrum_prop(incoming, z_move, refractive_index)

    def use_fresnel() -> OpticalWavefront:
        return fresnel_prop(incoming, z_move, refractive_index)

    def use_fraunhofer() -> OpticalWavefront:
        return fraunhofer_prop(incoming, z_move, refractive_index)

    # Select propagator based on Fresnel number and validity criteria
    # Use nested conditionals for proper jax.lax.cond syntax
    def select_fresnel_or_fraunhofer() -> OpticalWavefront:
        # If Fresnel number > 0.1, use Fresnel, otherwise Fraunhofer
        return jax.lax.cond(
            fresnel_number > 0.1,
            use_fresnel,
            use_fraunhofer,
        )

    # First check if angular spectrum is valid and we're in near field
    # Angular spectrum is preferred when valid as it's most accurate
    propagated: OpticalWavefront = jax.lax.cond(
        (fresnel_number > 1.0) & angular_spectrum_valid,
        use_angular_spectrum,
        select_fresnel_or_fraunhofer,
    )

    return propagated
