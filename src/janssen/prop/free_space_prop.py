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
    without making any paraxial approximations.
correct_propagator : function
    Automatically selects the most appropriate propagation method.
digital_zoom : function
    Zooms an optical wavefront by a specified factor.
fresnel_prop : function
    Propagates a complex optical field using the Fresnel approximation
fraunhofer_prop : function
    Propagates a complex optical field using the Fraunhofer
    approximation.
lens_propagation : function
    Propagates an optical wavefront through a lens.
optical_zoom : function
    Modifies the calibration of an optical wavefront without changing
    its field.

Notes
-----
All propagation methods are implemented using FFT-based algorithms for
efficiency. The choice of propagation method depends on the Fresnel
number and the specific requirements of the simulation.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Bool, Complex, Float, Integer, jaxtyped

from janssen.lenses import create_lens_phase
from janssen.utils import (
    LensParams,
    OpticalWavefront,
    ScalarFloat,
    ScalarInteger,
    ScalarNumeric,
    make_optical_wavefront,
)


@jaxtyped(typechecker=beartype)
def angular_spectrum_prop(
    incoming: OpticalWavefront,
    z_move: ScalarNumeric,
    refractive_index: Optional[ScalarNumeric] = 1.0,
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
    z_move : ScalarNumeric
        Propagation distance in meters
        This is in free space.
    refractive_index : Optional[ScalarNumeric], optional
        Index of refraction of the medium. Default is 1.0 (vacuum).

    Returns
    -------
    propagated : OpticalWavefront
        Propagated wave front

    Notes
    -----
    The angular spectrum method is an exact solution to the Helmholtz equation
    for propagation in homogeneous media. It decomposes the field into plane
    waves, propagates each component, and reconstructs the field.

    The transfer function is H(fx,fy) = exp(i*k*z*sqrt(1 - (lambda*fx)^2 -
    (lambda*fy)^2)) where k = 2*pi/lambda is the wavenumber. For spatial
    frequencies where (lambda*fx)^2 + (lambda*fy)^2 > 1, the waves become
    evanescent and are set to zero to prevent numerical instability.

    This method makes no paraxial approximation and is valid for all
    propagation distances, though numerical accuracy may degrade for very
    large distances due to sampling limitations.

    Algorithm:

    1. Compute spatial frequency grids fx, fy using FFT frequencies
    2. Build transfer function H = exp(i*k*z*sqrt(1 - lambda^2*(fx^2+fy^2)))
    3. Create evanescent mask where fx^2 + fy^2 <= 1/lambda^2
    4. FFT input field, multiply by masked transfer function, inverse FFT
    5. Return propagated wavefront with updated z_position
    """
    ny: ScalarInteger = incoming.field.shape[0]
    nx: ScalarInteger = incoming.field.shape[1]
    wavenumber: Float[Array, " "] = 2 * jnp.pi / incoming.wavelength
    path_length: Float[Array, " "] = refractive_index * z_move
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
    z_move: ScalarNumeric,
    refractive_index: Optional[ScalarNumeric] = 1.0,
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
    z_move : ScalarNumeric
        Propagation distance in meters
        This is in free space.
    refractive_index : Optional[ScalarNumeric], optional
        Index of refraction of the medium. Default is 1.0 (vacuum).

    Returns
    -------
    propagated : OpticalWavefront
        Propagated wave front

    Notes
    -----
    The Fresnel approximation is a paraxial approximation to scalar
    diffraction theory. It assumes that the propagation angle is small,
    which allows simplification of the angular spectrum transfer function
    using a Taylor expansion: sqrt(1 - lambda^2*(fx^2+fy^2)) ≈ 1 -
    lambda^2*(fx^2+fy^2)/2.

    This leads to the Fresnel transfer function:
    H(fx,fy) = exp(-i*pi*lambda*z*(fx^2 + fy^2))

    The output field is multiplied by a global phase factor exp(i*k*z)
    representing the on-axis propagation phase.

    The Fresnel approximation is valid when the Fresnel number F = a^2/(λz)
    is large (typically F > 1), where a is the characteristic aperture size.
    For small Fresnel numbers, use fraunhofer_prop instead.

    Algorithm:

    1. Compute spatial frequency grids fx, fy using FFT frequencies
    2. Build Fresnel transfer function H = exp(-i*pi*lambda*z*(fx^2+fy^2))
    3. FFT input field, multiply by transfer function, inverse FFT
    4. Multiply result by global phase exp(i*k*z)
    5. Return propagated wavefront with updated z_position
    """
    ny: ScalarInteger = incoming.field.shape[0]
    nx: ScalarInteger = incoming.field.shape[1]
    k: Float[Array, " "] = (2 * jnp.pi) / incoming.wavelength
    path_length: Float[Array, " "] = refractive_index * z_move
    fx: Float[Array, " hh"] = jnp.fft.fftfreq(nx, d=incoming.dx)
    fy: Float[Array, " ww"] = jnp.fft.fftfreq(ny, d=incoming.dx)
    fx_mesh: Float[Array, " hh ww"]
    fy_mesh: Float[Array, " hh ww"]
    fx_mesh, fy_mesh = jnp.meshgrid(fx, fy)
    transfer_phase: Float[Array, " hh ww"] = (
        -jnp.pi * incoming.wavelength * path_length * (fx_mesh**2 + fy_mesh**2)
    )
    transfer_function: Complex[Array, " hh ww"] = jnp.exp(1j * transfer_phase)
    field_ft: Complex[Array, " hh ww"] = jnp.fft.fft2(incoming.field)
    propagated_ft: Complex[Array, " hh ww"] = field_ft * transfer_function
    propagated_field: Complex[Array, " hh ww"] = jnp.fft.ifft2(propagated_ft)
    global_phase: Complex[Array, " "] = jnp.exp(1j * k * path_length)
    final_propagated_field: Complex[Array, " hh ww"] = (
        global_phase * propagated_field
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
    z_move: ScalarNumeric,
    refractive_index: ScalarNumeric = 1.0,
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
    z_move : ScalarNumeric
        Propagation distance in meters.
        This is in free space.
    refractive_index : ScalarNumeric, optional
        Index of refraction of the medium. Default is 1.0 (vacuum).

    Returns
    -------
    propagated : OpticalWavefront
        Propagated wave front. Note that the output pixel size (dx) changes
        according to Fraunhofer scaling: dx_out = wavelength * z / (N * dx_in)

    Notes
    -----
    The Fraunhofer approximation represents far-field diffraction where
    the diffraction pattern is proportional to the Fourier transform of
    the aperture function. This is the limiting case of Fresnel diffraction
    when the propagation distance is very large.

    The full Fraunhofer diffraction integral gives:
    U(x',y') = exp(i*k*z)/(i*lambda*z) * exp(i*k*(x'^2+y'^2)/(2*z)) *
               FT{U(x,y)} * dx^2

    where FT denotes the Fourier transform and the output coordinates are
    related to spatial frequencies by x' = lambda*z*fx, y' = lambda*z*fy.

    The quadratic phase term exp(i*k*(x'^2+y'^2)/(2*z)) represents the
    spherical wavefront curvature in the output plane and is essential
    for coherent imaging and phase-sensitive applications.

    The output pixel size changes according to the Fraunhofer scaling
    relation: dx_out = lambda * z / (N * dx_in), where N is the array size.

    The Fraunhofer approximation is valid when the Fresnel number F = a^2/(λz)
    is small (typically F < 1), where a is the characteristic aperture size.
    For large Fresnel numbers, use fresnel_prop or angular_spectrum_prop.

    Algorithm
    ---------

    1. Compute centered FFT of input field using ifftshift/fft2/fftshift
    2. Compute output pixel size dx_out = lambda*z/(N*dx_in)
    3. Create output coordinate grid and compute quadratic phase
    4. Apply global phase factor exp(i*k*z)
    5. Apply amplitude scaling 1/(i*lambda*z) and area element dx^2
    6. Multiply by quadratic phase term
    7. Return propagated wavefront with new dx and updated z_position
    """
    ny: ScalarInteger = incoming.field.shape[0]
    nx: ScalarInteger = incoming.field.shape[1]
    k: Float[Array, " "] = 2 * jnp.pi / incoming.wavelength
    path_length: Float[Array, " "] = refractive_index * z_move

    # Compute output pixel size first (needed for quadratic phase)
    dx_out: Float[Array, " "] = (
        incoming.wavelength * path_length / (nx * incoming.dx)
    )

    # FFT of input field (centered)
    field_ft: Complex[Array, " hh ww"] = jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.ifftshift(incoming.field))
    )

    # Create output coordinate grid for quadratic phase
    x_out: Float[Array, " ww"] = (jnp.arange(nx) - nx / 2) * dx_out
    y_out: Float[Array, " hh"] = (jnp.arange(ny) - ny / 2) * dx_out
    x_mesh: Float[Array, " hh ww"]
    y_mesh: Float[Array, " hh ww"]
    x_mesh, y_mesh = jnp.meshgrid(x_out, y_out)

    # Quadratic phase term: exp(i*k*(x'^2 + y'^2)/(2*z))
    quadratic_phase: Complex[Array, " hh ww"] = jnp.exp(
        1j * k * (x_mesh**2 + y_mesh**2) / (2 * path_length)
    )

    # Global phase and amplitude scaling
    global_phase: Complex[Array, " "] = jnp.exp(1j * k * path_length)
    scale_factor: Complex[Array, " "] = 1 / (
        1j * incoming.wavelength * path_length
    )

    # Combine all terms
    propagated_field: Complex[Array, " hh ww"] = (
        global_phase
        * scale_factor
        * quadratic_phase
        * field_ft
        * (incoming.dx**2)
    )

    propagated: OpticalWavefront = make_optical_wavefront(
        field=propagated_field,
        wavelength=incoming.wavelength,
        dx=dx_out,
        z_position=incoming.z_position + path_length,
    )
    return propagated


@jaxtyped(typechecker=beartype)
def fraunhofer_prop_scaled(
    incoming: OpticalWavefront,
    z_move: ScalarNumeric,
    output_dx: ScalarFloat,
    refractive_index: ScalarNumeric = 1.0,
) -> OpticalWavefront:
    """Propagate using Fraunhofer with output at specified pixel size.

    Performs Fraunhofer propagation with the output sampled at the
    desired pixel size. The output array has the same shape as the input.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input optical wavefront.
    z_move : ScalarNumeric
        Propagation distance in meters.
    output_dx : ScalarFloat
        Desired output pixel size in meters.
    refractive_index : ScalarNumeric, optional
        Index of refraction. Default is 1.0 (vacuum).

    Returns
    -------
    propagated : OpticalWavefront
        Propagated wavefront with specified output pixel size.
        Output array shape equals input array shape.

    Notes
    -----
    The standard Fraunhofer relation is:
        U(x') = C * FT{U(x)} evaluated at fx = x'/(λz)

    where C includes phase and amplitude factors.

    For output pixel n (centered), we want x'_n = (n - N/2) * output_dx.
    This corresponds to spatial frequency fx_n = x'_n/(λz).

    We achieve this by using a chirp-z transform (CZT) approach:
    instead of FFT which samples at fx = n/(N*dx_in), we use
    interpolation in Fourier space to sample at the desired frequencies.

    The key insight is that the DFT samples at frequencies:
        fx_fft[n] = (n - N/2) / (N * dx_in)

    And we want to sample at:
        fx_out[n] = (n - N/2) * output_dx / (λz)

    The ratio is: fx_out / fx_fft = output_dx * N * dx_in / (λz)

    This is equivalent to scaling the output coordinates, which we
    implement by interpolating the FFT result.
    """
    ny: ScalarInteger = incoming.field.shape[0]
    nx: ScalarInteger = incoming.field.shape[1]
    k: Float[Array, " "] = 2 * jnp.pi / incoming.wavelength
    path_length: Float[Array, " "] = refractive_index * z_move

    # Standard Fraunhofer dx for reference
    dx_fraunhofer: Float[Array, " "] = (
        incoming.wavelength * path_length / (nx * incoming.dx)
    )

    # Scaling factor: how much to zoom the FFT result
    # scale > 1 means output_dx > dx_fraunhofer (zoom out / larger pixels)
    # scale < 1 means output_dx < dx_fraunhofer (zoom in / smaller pixels)
    scale: Float[Array, " "] = output_dx / dx_fraunhofer

    # FFT of input field (centered)
    field_ft: Complex[Array, " hh ww"] = jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.ifftshift(incoming.field))
    )

    # Create interpolation coordinates to resample FFT at scaled frequencies
    # Original FFT is sampled at indices 0..N-1 (after fftshift, centered)
    # We want to sample at scaled positions
    center_y: Float[Array, " "] = (ny - 1) / 2.0
    center_x: Float[Array, " "] = (nx - 1) / 2.0

    # Output indices (0 to N-1)
    out_y: Float[Array, " hh"] = jnp.arange(ny, dtype=jnp.float64)
    out_x: Float[Array, " ww"] = jnp.arange(nx, dtype=jnp.float64)

    # Map output indices to input FFT indices via scaling
    # (out - center) * scale + center = input_index
    in_y: Float[Array, " hh"] = (out_y - center_y) * scale + center_y
    in_x: Float[Array, " ww"] = (out_x - center_x) * scale + center_x

    # Create meshgrid for 2D interpolation
    in_y_mesh: Float[Array, " hh ww"]
    in_x_mesh: Float[Array, " hh ww"]
    in_y_mesh, in_x_mesh = jnp.meshgrid(in_y, in_x, indexing="ij")

    # Interpolate FFT (real and imaginary separately)
    scaled_ft_real: Float[Array, " hh ww"] = jax.scipy.ndimage.map_coordinates(
        field_ft.real,
        [in_y_mesh, in_x_mesh],
        order=1,
        mode="constant",
        cval=0.0,
    )
    scaled_ft_imag: Float[Array, " hh ww"] = jax.scipy.ndimage.map_coordinates(
        field_ft.imag,
        [in_y_mesh, in_x_mesh],
        order=1,
        mode="constant",
        cval=0.0,
    )
    scaled_ft: Complex[Array, " hh ww"] = scaled_ft_real + 1j * scaled_ft_imag

    # Output coordinate grid for quadratic phase
    x_out: Float[Array, " ww"] = (jnp.arange(nx) - nx / 2) * output_dx
    y_out: Float[Array, " hh"] = (jnp.arange(ny) - ny / 2) * output_dx
    x_mesh: Float[Array, " hh ww"]
    y_mesh: Float[Array, " hh ww"]
    x_mesh, y_mesh = jnp.meshgrid(x_out, y_out)

    # Quadratic phase term: exp(i*k*(x'^2 + y'^2)/(2*z))
    quadratic_phase: Complex[Array, " hh ww"] = jnp.exp(
        1j * k * (x_mesh**2 + y_mesh**2) / (2 * path_length)
    )

    # Global phase and amplitude scaling
    global_phase: Complex[Array, " "] = jnp.exp(1j * k * path_length)
    scale_factor: Complex[Array, " "] = 1 / (
        1j * incoming.wavelength * path_length
    )

    # Combine all terms
    propagated_field: Complex[Array, " hh ww"] = (
        global_phase
        * scale_factor
        * quadratic_phase
        * scaled_ft
        * (incoming.dx**2)
    )

    propagated: OpticalWavefront = make_optical_wavefront(
        field=propagated_field,
        wavelength=incoming.wavelength,
        dx=output_dx,
        z_position=incoming.z_position + path_length,
    )
    return propagated


@jaxtyped(typechecker=beartype)
def digital_zoom(
    wavefront: OpticalWavefront,
    zoom_factor: ScalarNumeric,
) -> OpticalWavefront:
    """Zoom an optical wavefront by a specified factor.

    Key is this returns the same sized array as the original wavefront.

    Parameters
    ----------
    wavefront : OpticalWavefront
        Incoming optical wavefront.
    zoom_factor : ScalarNumeric
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
    zoom_factor: ScalarNumeric,
) -> OpticalWavefront:
    """Modify the calibration of an optical wavefront without changing
    field.

    Parameters
    ----------
    wavefront : OpticalWavefront
        Incoming optical wavefront.
    zoom_factor : ScalarNumeric
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
    z_move: ScalarNumeric,
    refractive_index: Optional[ScalarNumeric] = 1.0,
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
    z_move : ScalarNumeric
        Propagation distance in meters (in free space)
    refractive_index : Optional[ScalarNumeric], optional
        Index of refraction of the medium. Default is 1.0 (vacuum)

    Returns
    -------
    propagated : OpticalWavefront
        Propagated wave front using the most appropriate method

    Notes
    -----
    Implementation:
    1. Get field dimensions (ny, nx)
    2. Calculate field intensity distribution
    3. Create coordinate arrays centered at field center
    4. Calculate RMS width in both x and y directions
    5. Use larger RMS width times 2 as characteristic aperture size
    6. Account for refractive index in path length calculation
    7. Calculate Fresnel number: F = a²/(λz)
    8. Check angular spectrum validity criterion: dx < 0.5 * z * λ / L
       where L is the field size
    9. Use nested jax.lax.cond to select propagator:
       - If F > 1.0 AND angular spectrum valid: use angular spectrum
       - Else if F > 0.1: use Fresnel propagation
       - Else: use Fraunhofer propagation (far-field)

    Selection criteria:
    - Angular spectrum: F > 1 and sampling valid (most accurate, no
      paraxial approximation)
    - Fresnel: 0.1 < F ≤ 1 (near to intermediate field)
    - Fraunhofer: F < 0.1 (far-field)

    The angular spectrum method is preferred when applicable as it
    makes no paraxial approximations.
    """
    fresnel_number_threshold: ScalarFloat = 0.1
    ny: ScalarInteger = incoming.field.shape[0]
    nx: ScalarInteger = incoming.field.shape[1]

    field_intensity: Float[Array, " hh ww"] = jnp.abs(incoming.field) ** 2
    total_intensity: Float[Array, " "] = jnp.sum(field_intensity)

    y_coords: Float[Array, " hh"] = (jnp.arange(ny) - ny / 2) * incoming.dx
    x_coords: Float[Array, " ww"] = (jnp.arange(nx) - nx / 2) * incoming.dx
    y_mesh: Float[Array, " hh ww"]
    x_mesh: Float[Array, " hh ww"]
    y_mesh, x_mesh = jnp.meshgrid(y_coords, x_coords, indexing="ij")

    x_rms: Float[Array, " "] = jnp.sqrt(
        jnp.sum(field_intensity * x_mesh**2) / (total_intensity + 1e-10)
    )
    y_rms: Float[Array, " "] = jnp.sqrt(
        jnp.sum(field_intensity * y_mesh**2) / (total_intensity + 1e-10)
    )

    aperture_size: Float[Array, " "] = jnp.maximum(x_rms, y_rms) * 2

    path_length: Float[Array, " "] = refractive_index * z_move

    fresnel_number: Float[Array, " "] = aperture_size**2 / (
        incoming.wavelength * jnp.abs(path_length)
    )

    field_size: Float[Array, " "] = jnp.maximum(
        nx * incoming.dx, ny * incoming.dx
    )
    angular_spectrum_valid: Bool[Array, " "] = (
        incoming.dx
        < 0.5 * jnp.abs(path_length) * incoming.wavelength / field_size
    )

    def use_angular_spectrum() -> OpticalWavefront:
        return angular_spectrum_prop(incoming, z_move, refractive_index)

    def use_fresnel() -> OpticalWavefront:
        return fresnel_prop(incoming, z_move, refractive_index)

    def use_fraunhofer() -> OpticalWavefront:
        return fraunhofer_prop(incoming, z_move, refractive_index)

    def select_fresnel_or_fraunhofer() -> OpticalWavefront:
        return jax.lax.cond(
            fresnel_number > fresnel_number_threshold,
            use_fresnel,
            use_fraunhofer,
        )

    propagated: OpticalWavefront = jax.lax.cond(
        (fresnel_number > 1.0) & angular_spectrum_valid,
        use_angular_spectrum,
        select_fresnel_or_fraunhofer,
    )

    return propagated
