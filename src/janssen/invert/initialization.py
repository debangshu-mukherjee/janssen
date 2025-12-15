"""Initialization functions for ptychography reconstruction.

Extended Summary
----------------
Provides initialization by running the microscope model in reverse.
Takes diffraction patterns, propagates backwards through the optical
system, and places the results at their scan positions to build an
initial sample estimate.

Routine Listings
----------------
init_simple_microscope : function
    Initialize sample by inverting the simple microscope forward model
compute_fov_and_positions : function
    Compute FOV size and normalized positions from experimental data

Notes
-----
All functions are JAX-compatible and return complex-valued arrays
suitable for gradient-based optimization.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from janssen.scopes import simple_microscope
from janssen.utils import (
    MicroscopeData,
    OpticalWavefront,
    PtychographyReconstruction,
    SampleFunction,
    ScalarFloat,
    ScalarInteger,
    make_optical_wavefront,
    make_ptychography_reconstruction,
    make_sample_function,
)


@jaxtyped(typechecker=beartype)
def compute_fov_and_positions(
    experimental_data: MicroscopeData,
    probe_lightwave: OpticalWavefront,
    padding: Optional[ScalarInteger] = None,
) -> Tuple[
    int,
    int,
    Float[Array, " N 2"],
    Float[Array, " "],
]:
    """Compute FOV dimensions and normalized positions.

    Converts scan positions from meters to pixels, computes the required
    FOV size to contain all positions plus probe size and padding, then
    normalizes positions so they start at (padding + half_probe) in the
    FOV coordinate system.

    Parameters
    ----------
    experimental_data : MicroscopeData
        Experimental diffraction patterns with positions in meters.
    probe_lightwave : OpticalWavefront
        The probe/lightwave with field shape and pixel size.
    padding : ScalarInteger, optional
        Additional padding in pixels. If None, defaults to half probe size.

    Returns
    -------
    fov_size_y : int
        FOV height in pixels.
    fov_size_x : int
        FOV width in pixels.
    translated_positions : Float[Array, " N 2"]
        Positions translated to FOV coordinates (in meters).
    sample_dx : Float[Array, " "]
        Pixel size for the sample (same as probe dx).
    """
    probe_size_y: int
    probe_size_x: int
    probe_size_y, probe_size_x = probe_lightwave.field.shape
    sample_dx: Float[Array, " "] = probe_lightwave.dx

    pixel_positions: Float[Array, " N 2"] = (
        experimental_data.positions / sample_dx
    )

    min_pos_x: Float[Array, " "] = jnp.min(pixel_positions[:, 0])
    max_pos_x: Float[Array, " "] = jnp.max(pixel_positions[:, 0])
    min_pos_y: Float[Array, " "] = jnp.min(pixel_positions[:, 1])
    max_pos_y: Float[Array, " "] = jnp.max(pixel_positions[:, 1])

    scan_fov_x: Float[Array, " "] = max_pos_x - min_pos_x
    scan_fov_y: Float[Array, " "] = max_pos_y - min_pos_y

    half_probe_x: int = probe_size_x // 2
    half_probe_y: int = probe_size_y // 2

    if padding is None:
        padding = max(half_probe_x, half_probe_y)

    fov_size_x: int = int(jnp.ceil(scan_fov_x)) + probe_size_x + 2 * padding
    fov_size_y: int = int(jnp.ceil(scan_fov_y)) + probe_size_y + 2 * padding

    normalized_positions_x: Float[Array, " N"] = (
        pixel_positions[:, 0] - min_pos_x + padding + half_probe_x
    )
    normalized_positions_y: Float[Array, " N"] = (
        pixel_positions[:, 1] - min_pos_y + padding + half_probe_y
    )

    translated_positions: Float[Array, " N 2"] = (
        jnp.stack([normalized_positions_x, normalized_positions_y], axis=1)
        * sample_dx
    )

    return fov_size_y, fov_size_x, translated_positions, sample_dx


@jaxtyped(typechecker=beartype)
def _inverse_fraunhofer_prop_scaled(
    at_camera: OpticalWavefront,
    z_move: ScalarFloat,
    output_dx: ScalarFloat,
    refractive_index: ScalarFloat = 1.0,
) -> OpticalWavefront:
    """Inverse scaled Fraunhofer propagation (propagate backwards).

    Inverts fraunhofer_prop_scaled by:
    1. Removing the quadratic phase term (conjugate)
    2. Removing global phase and amplitude scaling (inverse)
    3. Reversing the interpolation-based scaling
    4. Applying inverse centered FFT to recover the field

    Parameters
    ----------
    at_camera : OpticalWavefront
        Field at camera plane.
    z_move : ScalarFloat
        Original propagation distance (positive). We propagate -z_move.
    output_dx : ScalarFloat
        Desired output pixel size (the original input dx before forward prop).
    refractive_index : ScalarFloat, optional
        Index of refraction. Default is 1.0.

    Returns
    -------
    before_prop : OpticalWavefront
        Field before propagation (at aperture plane).
    """
    ny: int
    nx: int
    ny, nx = at_camera.field.shape
    k: Float[Array, " "] = 2 * jnp.pi / at_camera.wavelength
    path_length: Float[Array, " "] = refractive_index * z_move

    x_cam: Float[Array, " W"] = (jnp.arange(nx) - nx / 2) * at_camera.dx
    y_cam: Float[Array, " H"] = (jnp.arange(ny) - ny / 2) * at_camera.dx
    x_mesh: Float[Array, " H W"]
    y_mesh: Float[Array, " H W"]
    x_mesh, y_mesh = jnp.meshgrid(x_cam, y_cam)

    quadratic_phase_conj: Complex[Array, " H W"] = jnp.exp(
        -1j * k * (x_mesh**2 + y_mesh**2) / (2 * path_length)
    )

    global_phase_conj: Complex[Array, " "] = jnp.exp(-1j * k * path_length)
    scale_factor_inv: Complex[Array, " "] = (
        1j * at_camera.wavelength * path_length
    )

    field_unscaled: Complex[Array, " H W"] = (
        at_camera.field
        * global_phase_conj
        * scale_factor_inv
        * quadratic_phase_conj
        / (output_dx**2)
    )

    dx_fraunhofer: Float[Array, " "] = (
        at_camera.wavelength * path_length / (nx * output_dx)
    )

    scale: Float[Array, " "] = at_camera.dx / dx_fraunhofer

    center_y: float = (ny - 1) / 2.0
    center_x: float = (nx - 1) / 2.0

    out_y: Float[Array, " H"] = jnp.arange(ny, dtype=jnp.float64)
    out_x: Float[Array, " W"] = jnp.arange(nx, dtype=jnp.float64)

    in_y: Float[Array, " H"] = (out_y - center_y) / scale + center_y
    in_x: Float[Array, " W"] = (out_x - center_x) / scale + center_x

    in_y_mesh: Float[Array, " H W"]
    in_x_mesh: Float[Array, " H W"]
    in_y_mesh, in_x_mesh = jnp.meshgrid(in_y, in_x, indexing="ij")

    unscaled_ft_real: Float[Array, " H W"] = jax.scipy.ndimage.map_coordinates(
        field_unscaled.real,
        [in_y_mesh, in_x_mesh],
        order=1,
        mode="constant",
        cval=0.0,
    )
    unscaled_ft_imag: Float[Array, " H W"] = jax.scipy.ndimage.map_coordinates(
        field_unscaled.imag,
        [in_y_mesh, in_x_mesh],
        order=1,
        mode="constant",
        cval=0.0,
    )
    field_ft: Complex[Array, " H W"] = unscaled_ft_real + 1j * unscaled_ft_imag

    # The forward model uses centered FFT: fftshift(fft2(ifftshift(field)))
    # The camera field (after forward prop) is centered (DC at array center).
    # After removing phase terms and unscaling, field_ft corresponds to the
    # centered FFT output. To invert back to spatial domain with DC at center:
    # Use centered IFFT: fftshift(ifft2(ifftshift(field_ft)))
    field_before: Complex[Array, " H W"] = jnp.fft.fftshift(
        jnp.fft.ifft2(jnp.fft.ifftshift(field_ft))
    )

    return make_optical_wavefront(
        field=field_before,
        wavelength=at_camera.wavelength,
        dx=output_dx,
        z_position=at_camera.z_position - path_length,
    )


@jaxtyped(typechecker=beartype)
def _inverse_optical_zoom(
    wavefront: OpticalWavefront,
    zoom_factor: ScalarFloat,
) -> OpticalWavefront:
    """Inverse optical zoom.

    Reverses optical_zoom by dividing dx by the zoom_factor instead of
    multiplying.

    Parameters
    ----------
    wavefront : OpticalWavefront
        Zoomed wavefront.
    zoom_factor : ScalarFloat
        Original zoom factor used in forward model.

    Returns
    -------
    unzoomed : OpticalWavefront
        Wavefront with original pixel size (dx / zoom_factor).
    """
    new_dx: Float[Array, " "] = wavefront.dx / zoom_factor
    return make_optical_wavefront(
        field=wavefront.field,
        wavelength=wavefront.wavelength,
        dx=new_dx,
        z_position=wavefront.z_position,
    )


@jaxtyped(typechecker=beartype)
def _get_aperture_mask(
    shape: Tuple[int, int],
    dx: ScalarFloat,
    aperture_diameter: ScalarFloat,
    aperture_center: Optional[Float[Array, " 2"]] = None,
) -> Float[Array, " H W"]:
    """Create circular aperture mask.

    Generates a binary mask with 1.0 inside the aperture and 0.0 outside.
    The mask is centered on the array with optional offset.

    Parameters
    ----------
    shape : Tuple[int, int]
        (height, width) of the mask.
    dx : ScalarFloat
        Pixel size in meters.
    aperture_diameter : ScalarFloat
        Aperture diameter in meters.
    aperture_center : Float[Array, " 2"], optional
        Center of aperture [x, y] in meters. Default is [0, 0].

    Returns
    -------
    mask : Float[Array, " H W"]
        Binary aperture mask (1 inside, 0 outside).
    """
    ny: int
    nx: int
    ny, nx = shape
    center: Float[Array, " 2"] = (
        aperture_center if aperture_center is not None else jnp.zeros(2)
    )

    x: Float[Array, " W"] = (jnp.arange(nx) - nx // 2) * dx
    y: Float[Array, " H"] = (jnp.arange(ny) - ny // 2) * dx
    xx: Float[Array, " H W"]
    yy: Float[Array, " H W"]
    xx, yy = jnp.meshgrid(x, y)

    r: Float[Array, " H W"] = jnp.sqrt(
        (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    )

    mask: Float[Array, " H W"] = (r <= aperture_diameter / 2.0).astype(
        jnp.float64
    )
    return mask


@jaxtyped(typechecker=beartype)
def init_simple_microscope(
    experimental_data: MicroscopeData,
    probe_lightwave: OpticalWavefront,
    zoom_factor: ScalarFloat,
    aperture_diameter: ScalarFloat,
    travel_distance: ScalarFloat,
    camera_pixel_size: ScalarFloat,
    aperture_center: Optional[Float[Array, " 2"]] = None,
    padding: Optional[ScalarInteger] = None,
    regularization: float = 1e-6,
    seed: int = 42,
) -> PtychographyReconstruction:
    """Initialize sample by inverting the simple microscope forward model.

    Runs the microscope forward model in reverse to create an initial
    sample estimate from experimental diffraction patterns. This serves
    as iteration 0 of the reconstruction, returning a PtychographyReconstruction
    that can be passed to simple_microscope_ptychography for further
    optimization.

    For each pattern, the algorithm:

    1. Takes sqrt of intensity to recover amplitude, assigns random phase
    2. Propagates backwards via inverse scaled Fraunhofer propagation
    3. Divides by aperture mask (with regularization for zero regions)
    4. Applies inverse optical zoom to recover original pixel size
    5. Divides by probe to isolate sample contribution

    All patterns are processed in parallel using vmap. The individual
    sample estimates are then placed at their scan positions and combined
    using weighted averaging, where weights are based on probe intensity.
    Overlapping regions are averaged together. Regions with no coverage
    are filled with 1.0 (transparent). The final result is normalized
    to have mean amplitude near 1.0.

    Parameters
    ----------
    experimental_data : MicroscopeData
        Experimental diffraction patterns with positions in meters.
        Shape of image_data: (N, H, W) where N is number of positions.
    probe_lightwave : OpticalWavefront
        The probe/lightwave used in the experiment.
    zoom_factor : ScalarFloat
        Optical zoom factor for magnification.
    aperture_diameter : ScalarFloat
        Diameter of the aperture in meters.
    travel_distance : ScalarFloat
        Light propagation distance in meters.
    camera_pixel_size : ScalarFloat
        Physical size of camera pixels in meters.
    aperture_center : Float[Array, " 2"], optional
        Center position of the aperture (x, y) in meters. Default is None
        (centered at origin).
    padding : ScalarInteger, optional
        Additional padding in pixels around the scanned region.
        If None, defaults to half probe size.
    regularization : float, optional
        Small value for numerical stability in divisions. Default is 1e-6.
    seed : int, optional
        Random seed for initial phase assignment. Default is 42.

    Returns
    -------
    reconstruction : PtychographyReconstruction
        Initial reconstruction state containing:
        - sample: Initialized sample function
        - lightwave: The input probe lightwave
        - translated_positions: Positions in FOV coordinates
        - All optical parameters
        - intermediate_* arrays with shape [..., 1] for iteration 0
        - losses array with shape (1, 2) containing [0, nan]
    """
    zoom_factor_arr: Float[Array, " "] = jnp.asarray(
        zoom_factor, dtype=jnp.float64
    )
    aperture_diameter_arr: Float[Array, " "] = jnp.asarray(
        aperture_diameter, dtype=jnp.float64
    )
    travel_distance_arr: Float[Array, " "] = jnp.asarray(
        travel_distance, dtype=jnp.float64
    )

    fov_size_y: int
    fov_size_x: int
    translated_positions: Float[Array, " N 2"]
    sample_dx: Float[Array, " "]
    fov_size_y, fov_size_x, translated_positions, sample_dx = (
        compute_fov_and_positions(experimental_data, probe_lightwave, padding)
    )

    probe_size_y: int
    probe_size_x: int
    probe_size_y, probe_size_x = probe_lightwave.field.shape
    half_probe_x: int = probe_size_x // 2
    half_probe_y: int = probe_size_y // 2
    num_positions: int = experimental_data.image_data.shape[0]

    zoomed_dx: Float[Array, " "] = sample_dx * zoom_factor_arr
    aperture_mask: Float[Array, " H W"] = _get_aperture_mask(
        (probe_size_y, probe_size_x),
        zoomed_dx,
        aperture_diameter_arr,
        aperture_center,
    )

    probe_intensity: Float[Array, " H W"] = jnp.abs(probe_lightwave.field) ** 2
    probe_intensity_max: Float[Array, " "] = jnp.max(probe_intensity)

    key: jax.Array = jax.random.PRNGKey(seed)
    random_phases: Float[Array, " N H W"] = jax.random.uniform(
        key,
        (num_positions, probe_size_y, probe_size_x),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )

    def invert_single_pattern(
        diff_pattern: Float[Array, " H W"],
        random_phase: Float[Array, " H W"],
    ) -> Complex[Array, " H W"]:
        """Invert a single diffraction pattern to get sample estimate."""
        amplitude: Float[Array, " H W"] = jnp.sqrt(
            jnp.maximum(diff_pattern, 0.0)
        )
        field_at_camera: Complex[Array, " H W"] = amplitude * jnp.exp(
            1j * random_phase
        )

        camera_wavefront: OpticalWavefront = make_optical_wavefront(
            field=field_at_camera.astype(jnp.complex128),
            wavelength=probe_lightwave.wavelength,
            dx=camera_pixel_size,
            z_position=jnp.array(0.0),
        )

        after_prop: OpticalWavefront = _inverse_fraunhofer_prop_scaled(
            camera_wavefront,
            travel_distance_arr,
            zoomed_dx,
        )

        # Apply aperture: zero out regions outside the aperture rather than
        # dividing (which would amplify noise outside the aperture circle)
        after_aperture: Complex[Array, " H W"] = after_prop.field * aperture_mask

        unzoomed_wavefront: OpticalWavefront = make_optical_wavefront(
            field=after_aperture,
            wavelength=probe_lightwave.wavelength,
            dx=zoomed_dx,
            z_position=after_prop.z_position,
        )
        at_sample: OpticalWavefront = _inverse_optical_zoom(
            unzoomed_wavefront, zoom_factor_arr
        )

        probe_below_reg: Float[Array, " H W"] = (
            jnp.abs(probe_lightwave.field) < regularization
        )
        probe_safe: Complex[Array, " H W"] = (
            probe_lightwave.field + regularization * probe_below_reg
        )
        sample_estimate: Complex[Array, " H W"] = at_sample.field / probe_safe

        return sample_estimate

    sample_estimates: Complex[Array, " N H W"] = jax.vmap(
        invert_single_pattern
    )(experimental_data.image_data, random_phases)

    pixel_positions: Float[Array, " N 2"] = translated_positions / sample_dx

    sample_sum: Complex[Array, " H W"] = jnp.zeros(
        (fov_size_y, fov_size_x), dtype=jnp.complex128
    )
    weight_sum: Float[Array, " H W"] = jnp.zeros(
        (fov_size_y, fov_size_x), dtype=jnp.float64
    )

    def accumulate_sample(
        carry: Tuple[Complex[Array, " H W"], Float[Array, " H W"]],
        inputs: Tuple[Complex[Array, " h w"], Float[Array, " 2"]],
    ) -> Tuple[
        Tuple[Complex[Array, " H W"], Float[Array, " H W"]],
        None,
    ]:
        """Accumulate weighted sample estimate at scan position."""
        sample_acc: Complex[Array, " H W"]
        weight_acc: Float[Array, " H W"]
        sample_acc, weight_acc = carry

        estimate: Complex[Array, " h w"]
        pos: Float[Array, " 2"]
        estimate, pos = inputs

        pos_x: Int[Array, " "] = jnp.round(pos[0]).astype(int)
        pos_y: Int[Array, " "] = jnp.round(pos[1]).astype(int)

        start_y: Int[Array, " "] = pos_y - half_probe_y
        start_x: Int[Array, " "] = pos_x - half_probe_x

        weight_patch: Float[Array, " H W"] = probe_intensity / (
            probe_intensity_max + regularization
        )

        start_y_clamped: Int[Array, " "] = jnp.clip(
            start_y, 0, fov_size_y - probe_size_y
        )
        start_x_clamped: Int[Array, " "] = jnp.clip(
            start_x, 0, fov_size_x - probe_size_x
        )

        current_sample: Complex[Array, " h w"] = jax.lax.dynamic_slice(
            sample_acc,
            (start_y_clamped, start_x_clamped),
            (probe_size_y, probe_size_x),
        )
        current_weight: Float[Array, " h w"] = jax.lax.dynamic_slice(
            weight_acc,
            (start_y_clamped, start_x_clamped),
            (probe_size_y, probe_size_x),
        )

        new_sample: Complex[Array, " h w"] = (
            current_sample + estimate * weight_patch
        )
        new_weight: Float[Array, " h w"] = current_weight + weight_patch

        sample_acc = jax.lax.dynamic_update_slice(
            sample_acc, new_sample, (start_y_clamped, start_x_clamped)
        )
        weight_acc = jax.lax.dynamic_update_slice(
            weight_acc, new_weight, (start_y_clamped, start_x_clamped)
        )

        return (sample_acc, weight_acc), None

    (sample_sum, weight_sum), _ = jax.lax.scan(
        accumulate_sample,
        (sample_sum, weight_sum),
        (sample_estimates, pixel_positions),
    )

    weight_safe: Float[Array, " H W"] = weight_sum + regularization
    sample_field: Complex[Array, " H W"] = sample_sum / weight_safe

    no_data_mask: Float[Array, " H W"] = weight_sum < regularization
    sample_field = jnp.where(no_data_mask, 1.0 + 0j, sample_field)

    mean_amplitude: Float[Array, " "] = jnp.mean(
        jnp.abs(sample_field[~no_data_mask])
    )
    sample_field = jnp.where(
        no_data_mask,
        sample_field,
        sample_field / (mean_amplitude + regularization),
    )

    sample_function: SampleFunction = make_sample_function(
        sample=sample_field,
        dx=sample_dx,
    )

    probe_size_y: int
    probe_size_x: int
    probe_size_y, probe_size_x = probe_lightwave.field.shape

    fov_size_y: int = sample_field.shape[0]
    fov_size_x: int = sample_field.shape[1]

    intermediate_samples: Complex[Array, " H W 1"] = sample_field[
        :, :, jnp.newaxis
    ]
    intermediate_lightwaves: Complex[Array, " h w 1"] = probe_lightwave.field[
        :, :, jnp.newaxis
    ]
    intermediate_zoom_factors: Float[Array, " 1"] = zoom_factor_arr[
        jnp.newaxis
    ]
    intermediate_aperture_diameters: Float[Array, " 1"] = (
        aperture_diameter_arr[jnp.newaxis]
    )
    intermediate_aperture_centers: Float[Array, " 2 1"] = (
        jnp.zeros((2, 1))
        if aperture_center is None
        else aperture_center[:, jnp.newaxis]
    )
    intermediate_travel_distances: Float[Array, " 1"] = travel_distance_arr[
        jnp.newaxis
    ]

    # Compute initial MSE loss by running forward model
    camera_pixel_size_arr: Float[Array, " "] = jnp.asarray(
        camera_pixel_size, dtype=jnp.float64
    )
    simulated_data: MicroscopeData = simple_microscope(
        sample=sample_function,
        positions=translated_positions,
        lightwave=probe_lightwave,
        zoom_factor=zoom_factor_arr,
        aperture_diameter=aperture_diameter_arr,
        travel_distance=travel_distance_arr,
        camera_pixel_size=camera_pixel_size_arr,
        aperture_center=aperture_center,
    )
    initial_mse: Float[Array, " "] = jnp.mean(
        (simulated_data.image_data - experimental_data.image_data) ** 2
    )
    losses: Float[Array, " 1 2"] = jnp.array([[0.0, initial_mse]])

    reconstruction: PtychographyReconstruction = (
        make_ptychography_reconstruction(
            sample=sample_function,
            lightwave=probe_lightwave,
            translated_positions=translated_positions,
            zoom_factor=zoom_factor_arr,
            aperture_diameter=aperture_diameter_arr,
            aperture_center=aperture_center,
            travel_distance=travel_distance_arr,
            intermediate_samples=intermediate_samples,
            intermediate_lightwaves=intermediate_lightwaves,
            intermediate_zoom_factors=intermediate_zoom_factors,
            intermediate_aperture_diameters=intermediate_aperture_diameters,
            intermediate_aperture_centers=intermediate_aperture_centers,
            intermediate_travel_distances=intermediate_travel_distances,
            losses=losses,
        )
    )

    return reconstruction
