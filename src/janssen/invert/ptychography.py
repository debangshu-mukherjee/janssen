"""Ptychography algorithms and optimization.

Extended Summary
----------------
High-level ptychography reconstruction algorithms that combine
optimization strategies with forward models. Provides complete reconstruction
pipelines for recovering complex-valued sample functions from intensity
measurements.

Routine Listings
----------------
simple_microscope_ptychography : function
    Performs ptychography reconstruction using gradient-based optimization
simple_microscope_epie : function
    Performs ptychography reconstruction using extended PIE algorithm

Notes
-----
These functions provide complete reconstruction pipelines that can be
directly applied to experimental data. All functions support JAX
transformations and automatic differentiation for gradient-based optimization.
"""

import jax
import jax.numpy as jnp
import optax
from beartype import beartype
from beartype.typing import Callable, Tuple
from jax import lax
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from janssen.scopes import simple_microscope
from janssen.utils import (
    EpieData,
    MicroscopeData,
    OpticalWavefront,
    PtychographyParams,
    PtychographyReconstruction,
    SampleFunction,
    fourier_shift,
    make_epie_data,
    make_optical_wavefront,
    make_ptychography_reconstruction,
    make_sample_function,
)

from .initialization import init_simple_epie
from .loss_functions import create_loss_function

OPTIMIZERS: Tuple[
    optax.GradientTransformationExtraArgs,
    optax.GradientTransformationExtraArgs,
    optax.GradientTransformationExtraArgs,
    optax.GradientTransformationExtraArgs,
] = (
    optax.adam,
    optax.adagrad,
    optax.rmsprop,
    optax.sgd,
)

LOSS_TYPES: Tuple[str, str, str] = ("mse", "mae", "poisson")


@jaxtyped(typechecker=beartype)
def simple_microscope_ptychography(  # noqa: PLR0915
    experimental_data: MicroscopeData,
    reconstruction: PtychographyReconstruction,
    params: PtychographyParams,
) -> PtychographyReconstruction:
    """Continue ptychographic reconstruction from a previous state.

    Reconstructs a sample from experimental diffraction patterns using
    gradient-based optimization. Takes a PtychographyReconstruction
    (from init_simple_microscope or a previous call) and runs additional
    iterations, appending results to the intermediate arrays.

    This enables resumable reconstruction: run 20 iterations, save the
    result, then later resume from iteration 21. Uses jax.lax.scan for
    efficient iteration and full JAX compatibility.

    Parameters
    ----------
    experimental_data : MicroscopeData
        The experimental diffraction patterns collected at different
        positions. Positions should be in meters.
    reconstruction : PtychographyReconstruction
        Previous reconstruction state from init_simple_microscope or
        a previous call to this function. Contains sample, lightwave,
        positions, optical parameters, and intermediate history.
    params : PtychographyParams
        Optimization parameters including camera_pixel_size, num_iterations,
        learning_rate, loss_type, optimizer_type, and bounds for optical
        parameters.

    Returns
    -------
    reconstruction : PtychographyReconstruction
        Updated reconstruction with:
        - sample : Final optimized sample
        - lightwave : Final optimized probe/lightwave
        - translated_positions : Unchanged from input
        - Optical parameters (may be updated if bounds optimization enabled)
        - intermediate_* : Previous history + new iterations appended
        - losses : Previous history + new iterations appended

    See Also
    --------
    init_simple_microscope : Create initial reconstruction state.
    """
    guess_sample: SampleFunction = reconstruction.sample
    guess_lightwave: OpticalWavefront = reconstruction.lightwave
    translated_positions: Float[Array, " N 2"] = (
        reconstruction.translated_positions
    )
    zoom_factor: Float[Array, " "] = reconstruction.zoom_factor
    aperture_diameter: Float[Array, " "] = reconstruction.aperture_diameter
    travel_distance: Float[Array, " "] = reconstruction.travel_distance
    aperture_center: Float[Array, " 2"] = (
        jnp.zeros(2)
        if reconstruction.aperture_center is None
        else reconstruction.aperture_center
    )
    prev_intermediate_samples: Complex[Array, " H W S"] = (
        reconstruction.intermediate_samples
    )
    prev_intermediate_lightwaves: Complex[Array, " H W S"] = (
        reconstruction.intermediate_lightwaves
    )
    prev_intermediate_zoom_factors: Float[Array, " S"] = (
        reconstruction.intermediate_zoom_factors
    )
    prev_intermediate_aperture_diameters: Float[Array, " S"] = (
        reconstruction.intermediate_aperture_diameters
    )
    prev_intermediate_aperture_centers: Float[Array, " 2 S"] = (
        reconstruction.intermediate_aperture_centers
    )
    prev_intermediate_travel_distances: Float[Array, " S"] = (
        reconstruction.intermediate_travel_distances
    )
    prev_losses: Float[Array, " N 2"] = reconstruction.losses
    camera_pixel_size: Float[Array, " "] = params.camera_pixel_size
    num_iterations: Int[Array, " "] = params.num_iterations
    learning_rate: Float[Array, " "] = params.learning_rate
    loss_type: Int[Array, " "] = params.loss_type
    optimizer_type: Int[Array, " "] = params.optimizer_type
    zoom_factor_bounds: Float[Array, " 2"] = params.zoom_factor_bounds
    aperture_diameter_bounds: Float[Array, " 2"] = (
        params.aperture_diameter_bounds
    )
    travel_distance_bounds: Float[Array, " 2"] = params.travel_distance_bounds
    aperture_center_bounds: Float[Array, " 2 2"] = (
        params.aperture_center_bounds
    )
    start_iteration: Int[Array, " "] = jnp.array(
        prev_losses.shape[0], dtype=jnp.int64
    )
    num_iterations_int: int = int(num_iterations)
    sample_dx: Float[Array, " "] = guess_sample.dx
    guess_sample_field: Complex[Array, " H W"] = guess_sample.sample
    loss_type_str: str = LOSS_TYPES[int(loss_type)]

    def _forward_fn(
        sample_field: Complex[Array, " H W"],
        lightwave_field: Complex[Array, " H W"],
        zf: Float[Array, " "],
        ad: Float[Array, " "],
        td: Float[Array, " "],
        ac: Float[Array, " 2"],
    ) -> Float[Array, " N H W"]:
        sample: SampleFunction = make_sample_function(
            sample=sample_field, dx=sample_dx
        )
        lightwave: OpticalWavefront = make_optical_wavefront(
            field=lightwave_field,
            wavelength=guess_lightwave.wavelength,
            dx=guess_lightwave.dx,
            z_position=guess_lightwave.z_position,
        )
        simulated_data: MicroscopeData = simple_microscope(
            sample=sample,
            positions=translated_positions,
            lightwave=lightwave,
            zoom_factor=zf,
            aperture_diameter=ad,
            travel_distance=td,
            camera_pixel_size=camera_pixel_size,
            aperture_center=ac,
        )

        return simulated_data.image_data

    loss_func: Callable[..., Float[Array, " "]] = create_loss_function(
        _forward_fn, experimental_data.image_data, loss_type_str
    )

    def _compute_loss(
        sample_field: Complex[Array, " H W"],
        lightwave_field: Complex[Array, " H W"],
        zf: Float[Array, " "],
        ad: Float[Array, " "],
        td: Float[Array, " "],
        ac: Float[Array, " 2"],
    ) -> Float[Array, " "]:
        bounded_zf: Float[Array, " "] = jnp.clip(
            zf, zoom_factor_bounds[0], zoom_factor_bounds[1]
        )
        bounded_ad: Float[Array, " "] = jnp.clip(
            ad, aperture_diameter_bounds[0], aperture_diameter_bounds[1]
        )
        bounded_td: Float[Array, " "] = jnp.clip(
            td, travel_distance_bounds[0], travel_distance_bounds[1]
        )
        bounded_ac: Float[Array, " 2"] = jnp.clip(
            ac, aperture_center_bounds[0], aperture_center_bounds[1]
        )
        return loss_func(
            sample_field,
            lightwave_field,
            bounded_zf,
            bounded_ad,
            bounded_td,
            bounded_ac,
        )

    optimizer: optax.GradientTransformation = OPTIMIZERS[int(optimizer_type)](
        float(learning_rate)
    )
    sample_opt_state: optax.OptState = optimizer.init(guess_sample_field)
    sample_field: Complex[Array, " H W"] = guess_sample_field
    lightwave_field: Complex[Array, " H W"] = guess_lightwave.field

    def _scan_body(
        carry: Tuple[
            Complex[Array, " H W"],
            Complex[Array, " H W"],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " 2"],
            optax.OptState,
        ],
        _iteration: Int[Array, " "],
    ) -> Tuple[
        Tuple[
            Complex[Array, " H W"],
            Complex[Array, " H W"],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " 2"],
            optax.OptState,
        ],
        Tuple[
            Complex[Array, " H W"],
            Complex[Array, " H W"],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " 2"],
            Float[Array, " "],
        ],
    ]:
        sf, lf, zf, ad, td, ac, opt_state = carry
        loss_val, grad = jax.value_and_grad(_compute_loss, argnums=0)(
            sf, lf, zf, ad, td, ac
        )
        updates, new_opt_state = optimizer.update(grad, opt_state, sf)
        new_sf = optax.apply_updates(sf, updates)
        new_carry = (new_sf, lf, zf, ad, td, ac, new_opt_state)
        output = (new_sf, lf, zf, ad, td, ac, loss_val)
        return new_carry, output

    init_carry = (
        sample_field,
        lightwave_field,
        zoom_factor,
        aperture_diameter,
        travel_distance,
        aperture_center,
        sample_opt_state,
    )
    iterations: Int[Array, " N"] = jnp.arange(
        num_iterations_int, dtype=jnp.int64
    )
    final_carry, outputs = lax.scan(_scan_body, init_carry, iterations)
    (
        intermediate_samples_new,
        intermediate_lightwaves_new,
        intermediate_zoom_factors_new,
        intermediate_aperture_diameters_new,
        intermediate_travel_distances_new,
        intermediate_aperture_centers_new,
        losses_new,
    ) = outputs
    intermediate_samples: Complex[Array, " H W S"] = jnp.transpose(
        intermediate_samples_new, (1, 2, 0)
    )
    intermediate_lightwaves: Complex[Array, " H W S"] = jnp.transpose(
        intermediate_lightwaves_new, (1, 2, 0)
    )
    intermediate_aperture_centers: Float[Array, " 2 S"] = jnp.transpose(
        intermediate_aperture_centers_new, (1, 0)
    )
    iteration_numbers: Float[Array, " N"] = start_iteration + jnp.arange(
        num_iterations_int, dtype=jnp.float64
    )
    losses: Float[Array, " N 2"] = jnp.stack(
        [iteration_numbers, losses_new], axis=1
    )
    (
        final_sample_field,
        final_lightwave_field,
        current_zoom_factor,
        current_aperture_diameter,
        current_travel_distance,
        current_aperture_center,
        _,
    ) = final_carry
    final_sample: SampleFunction = make_sample_function(
        sample=final_sample_field, dx=sample_dx
    )
    final_lightwave: OpticalWavefront = make_optical_wavefront(
        field=final_lightwave_field,
        wavelength=guess_lightwave.wavelength,
        dx=guess_lightwave.dx,
        z_position=guess_lightwave.z_position,
    )
    combined_intermediate_samples: Complex[Array, " H W S"] = jnp.concatenate(
        [prev_intermediate_samples, intermediate_samples], axis=-1
    )
    combined_intermediate_lightwaves: Complex[Array, " H W S"] = (
        jnp.concatenate(
            [prev_intermediate_lightwaves, intermediate_lightwaves], axis=-1
        )
    )
    combined_intermediate_zoom_factors: Float[Array, " S"] = jnp.concatenate(
        [prev_intermediate_zoom_factors, intermediate_zoom_factors_new],
        axis=-1,
    )
    combined_intermediate_aperture_diameters: Float[Array, " S"] = (
        jnp.concatenate(
            [
                prev_intermediate_aperture_diameters,
                intermediate_aperture_diameters_new,
            ],
            axis=-1,
        )
    )
    combined_intermediate_aperture_centers: Float[Array, " 2 S"] = (
        jnp.concatenate(
            [
                prev_intermediate_aperture_centers,
                intermediate_aperture_centers,
            ],
            axis=-1,
        )
    )
    combined_intermediate_travel_distances: Float[Array, " S"] = (
        jnp.concatenate(
            [
                prev_intermediate_travel_distances,
                intermediate_travel_distances_new,
            ],
            axis=-1,
        )
    )
    combined_losses: Float[Array, " N 2"] = jnp.concatenate(
        [prev_losses, losses], axis=0
    )
    full_and_intermediate: PtychographyReconstruction = (
        make_ptychography_reconstruction(
            sample=final_sample,
            lightwave=final_lightwave,
            translated_positions=translated_positions,
            zoom_factor=current_zoom_factor,
            aperture_diameter=current_aperture_diameter,
            aperture_center=current_aperture_center,
            travel_distance=current_travel_distance,
            intermediate_samples=combined_intermediate_samples,
            intermediate_lightwaves=combined_intermediate_lightwaves,
            intermediate_zoom_factors=combined_intermediate_zoom_factors,
            intermediate_aperture_diameters=(
                combined_intermediate_aperture_diameters
            ),
            intermediate_aperture_centers=(
                combined_intermediate_aperture_centers
            ),
            intermediate_travel_distances=(
                combined_intermediate_travel_distances
            ),
            losses=combined_losses,
        )
    )
    return full_and_intermediate


def _sm_epie_core(
    epie_data: EpieData,
    iterations: Int[Array, " N"],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> EpieData:
    """FFT-based ePIE core algorithm with Fourier shifting.

    Pure JAX implementation of ePIE reconstruction using FFT-based
    position shifting. The object and probe have the same size as the
    diffraction patterns, and position shifts are applied via phase
    ramps in Fourier space for sub-pixel accuracy.

    Parameters
    ----------
    epie_data : EpieData
        Preprocessed data from init_simple_epie containing:

        - diffraction_patterns: Scaled camera images (N, H, W)
        - probe: Initial probe centered in the array (H, W)
        - sample: Initial sample estimate, same size as probe (H, W)
        - positions: Scan positions in pixels relative to center (0, 0)
    iterations : Int[Array, " N"]
        Array of iteration indices to scan over.
    alpha : float, optional
        ePIE step size for object update. Default is 1.0.
    beta : float, optional
        ePIE step size for probe update. Default is 1.0.
        Set to 0 to freeze probe and only update object.

    Returns
    -------
    EpieData
        Updated EpieData with reconstructed sample and probe.

    Notes
    -----
    **Sequential ePIE Algorithm with FFT Shifting**

    For each iteration, we loop through all scan positions sequentially.
    At each position (dx, dy) relative to center:

    1. Shift probe by (dx, dy) to the scan position
    2. exit_wave = object * shifted_probe
    3. detector = FFT(exit_wave)
    4. Replace amplitude: detector_new = detector * sqrt(I) / |detector|
    5. exit_wave_new = IFFT(detector_new)
    6. Update object and probe using ePIE formulas (in lab frame)
    7. Use updated object for next position

    Key insight: We shift the PROBE to each position rather than shifting
    the object. This keeps updates in the lab frame where they belong.
    """
    diffraction_patterns: Float[Array, " N H W"] = (
        epie_data.diffraction_patterns
    )
    sample_field: Complex[Array, " H W"] = epie_data.sample
    probe_field: Complex[Array, " H W"] = epie_data.probe
    positions: Float[Array, " N 2"] = epie_data.positions
    eps: float = 1e-8

    def _epie_single_position(
        carry: Tuple[Complex[Array, " H W"], Complex[Array, " H W"]],
        inputs: Tuple[Float[Array, " H W"], Float[Array, " 2"]],
    ) -> Tuple[
        Tuple[Complex[Array, " H W"], Complex[Array, " H W"]],
        None,
    ]:
        """Process one scan position, updating object and probe."""
        obj, probe = carry
        measurement, pos = inputs
        shift_x: Float[Array, " "] = pos[0]
        shift_y: Float[Array, " "] = pos[1]

        # Shift probe to scan position (probe moves, object stays fixed)
        probe_shifted: Complex[Array, " H W"] = fourier_shift(
            probe, shift_x, shift_y
        )

        # Forward model: exit wave in lab frame
        exit_wave: Complex[Array, " H W"] = obj * probe_shifted
        exit_wave_ft: Complex[Array, " H W"] = jnp.fft.fftshift(
            jnp.fft.fft2(exit_wave)
        )

        # Amplitude constraint
        measured_amplitude: Float[Array, " H W"] = jnp.sqrt(
            jnp.maximum(measurement, 0.0)
        )
        current_amplitude: Float[Array, " H W"] = jnp.abs(exit_wave_ft) + eps
        exit_wave_ft_updated: Complex[Array, " H W"] = (
            exit_wave_ft * measured_amplitude / current_amplitude
        )
        exit_wave_updated: Complex[Array, " H W"] = jnp.fft.ifft2(
            jnp.fft.ifftshift(exit_wave_ft_updated)
        )

        # Difference for updates
        diff: Complex[Array, " H W"] = exit_wave_updated - exit_wave

        # ePIE object update (in lab frame, where probe_shifted illuminates)
        # Standard ePIE: O += alpha * P* * diff / max(|P|^2)
        # This ensures uniform weighting across the illuminated region
        probe_conj: Complex[Array, " H W"] = jnp.conj(probe_shifted)
        probe_intensity: Float[Array, " H W"] = jnp.abs(probe_shifted) ** 2
        probe_max_intensity: Float[Array, " "] = jnp.max(probe_intensity)
        obj_update: Complex[Array, " H W"] = (
            alpha * probe_conj * diff / (probe_max_intensity + eps)
        )
        obj_new: Complex[Array, " H W"] = obj + obj_update

        # ePIE probe update (in shifted frame, then shift back)
        # Standard ePIE: P += beta * O* * diff / max(|O|^2)
        obj_conj: Complex[Array, " H W"] = jnp.conj(obj)
        obj_intensity: Float[Array, " H W"] = jnp.abs(obj) ** 2
        obj_max_intensity: Float[Array, " "] = jnp.max(obj_intensity)
        probe_update_shifted: Complex[Array, " H W"] = (
            beta * obj_conj * diff / (obj_max_intensity + eps)
        )
        # Shift probe update back to probe's reference frame (centered)
        probe_update: Complex[Array, " H W"] = fourier_shift(
            probe_update_shifted, -shift_x, -shift_y
        )
        probe_new: Complex[Array, " H W"] = probe + probe_update

        return (obj_new, probe_new), None

    def _epie_one_iteration(
        carry: Tuple[Complex[Array, " H W"], Complex[Array, " H W"]],
        _iter_idx: Int[Array, " "],
    ) -> Tuple[
        Tuple[Complex[Array, " H W"], Complex[Array, " H W"]],
        None,
    ]:
        """One ePIE iteration: sequential pass through all positions."""
        # Use lax.scan to process positions sequentially
        final_carry, _ = lax.scan(
            _epie_single_position,
            carry,
            (diffraction_patterns, positions),
        )
        return final_carry, None

    init_carry: Tuple[Complex[Array, " H W"], Complex[Array, " H W"]] = (
        sample_field,
        probe_field,
    )
    final_carry, _ = lax.scan(_epie_one_iteration, init_carry, iterations)
    final_sample_field: Complex[Array, " H W"]
    final_probe_field: Complex[Array, " H W"]
    final_sample_field, final_probe_field = final_carry
    result: EpieData = make_epie_data(
        diffraction_patterns=diffraction_patterns,
        probe=final_probe_field,
        sample=final_sample_field,
        positions=positions,
        effective_dx=epie_data.effective_dx,
        wavelength=epie_data.wavelength,
        original_camera_pixel_size=epie_data.original_camera_pixel_size,
        zoom_factor=epie_data.zoom_factor,
    )
    return result


@jaxtyped(typechecker=beartype)
def simple_microscope_epie(  # noqa: PLR0914, PLR0915
    experimental_data: MicroscopeData,
    reconstruction: PtychographyReconstruction,
    params: PtychographyParams,
) -> PtychographyReconstruction:
    """Ptychographic reconstruction using extended PIE algorithm.

    High-level orchestration function that preprocesses data, runs the
    FFT-based ePIE algorithm, and returns results in PtychographyReconstruction
    format. Supports resuming from previous reconstructions.

    Parameters
    ----------
    experimental_data : MicroscopeData
        Experimental diffraction patterns collected at different positions.
        Positions should be in meters.
    reconstruction : PtychographyReconstruction
        Previous reconstruction state from init_simple_microscope or a
        previous call. Contains sample, lightwave, positions, optical
        parameters, and intermediate history.
    params : PtychographyParams
        Optimization parameters:

        - learning_rate: Controls ePIE step size (alpha parameter)
        - num_iterations: Number of complete sweeps over all positions
        - camera_pixel_size: Physical size of camera pixels in meters

    Returns
    -------
    reconstruction : PtychographyReconstruction
        Updated reconstruction with:

        - sample: Final optimized sample
        - lightwave: Final optimized probe/lightwave
        - translated_positions: Unchanged from input
        - Optical parameters: Unchanged from input
        - intermediate_*: Previous history + new iterations appended
        - losses: Previous history + new iterations appended

    Notes
    -----
    **Workflow**

    1. Preprocess data using init_simple_epie (scales to FFT coordinates)
    2. If resuming, use previous sample/probe as starting point
    3. Run _sm_epie_core for the requested iterations
    4. Convert results back to PtychographyReconstruction format

    **Resume Support**

    When prev_losses has entries, the function uses the existing sample
    and probe from the reconstruction as the starting point instead of
    the freshly initialized values from init_simple_epie.

    See Also
    --------
    init_simple_microscope : Create initial reconstruction state.
    init_simple_epie : Preprocessing for FFT-compatible ePIE.
    _sm_epie_core : Core ePIE algorithm.
    simple_microscope_ptychography : Gradient-based reconstruction.
    """
    guess_sample: SampleFunction = reconstruction.sample
    guess_lightwave: OpticalWavefront = reconstruction.lightwave
    zoom_factor: Float[Array, " "] = reconstruction.zoom_factor
    aperture_diameter: Float[Array, " "] = reconstruction.aperture_diameter
    travel_distance: Float[Array, " "] = reconstruction.travel_distance
    aperture_center: Float[Array, " 2"] = (
        jnp.zeros(2)
        if reconstruction.aperture_center is None
        else reconstruction.aperture_center
    )
    prev_intermediate_samples: Complex[Array, " H W S"] = (
        reconstruction.intermediate_samples
    )
    prev_intermediate_lightwaves: Complex[Array, " H W S"] = (
        reconstruction.intermediate_lightwaves
    )
    prev_intermediate_zoom_factors: Float[Array, " S"] = (
        reconstruction.intermediate_zoom_factors
    )
    prev_intermediate_aperture_diameters: Float[Array, " S"] = (
        reconstruction.intermediate_aperture_diameters
    )
    prev_intermediate_aperture_centers: Float[Array, " 2 S"] = (
        reconstruction.intermediate_aperture_centers
    )
    prev_intermediate_travel_distances: Float[Array, " S"] = (
        reconstruction.intermediate_travel_distances
    )
    prev_losses: Float[Array, " N 2"] = reconstruction.losses
    camera_pixel_size: Float[Array, " "] = params.camera_pixel_size
    num_iterations: Int[Array, " "] = params.num_iterations
    alpha: Float[Array, " "] = params.learning_rate
    num_iterations_int: int = int(num_iterations)
    epie_data: EpieData = init_simple_epie(
        experimental_data=experimental_data,
        effective_dx=guess_sample.dx,
        wavelength=guess_lightwave.wavelength,
        zoom_factor=zoom_factor,
        aperture_diameter=aperture_diameter,
        travel_distance=travel_distance,
        camera_pixel_size=camera_pixel_size,
    )
    epie_sample_shape: tuple[int, int] = epie_data.sample.shape
    recon_sample_shape: tuple[int, int] = guess_sample.sample.shape
    is_epie_resume: bool = epie_sample_shape == recon_sample_shape

    if is_epie_resume:
        epie_data = make_epie_data(
            diffraction_patterns=epie_data.diffraction_patterns,
            probe=guess_lightwave.field,
            sample=guess_sample.sample,
            positions=epie_data.positions,
            effective_dx=epie_data.effective_dx,
            wavelength=epie_data.wavelength,
            original_camera_pixel_size=epie_data.original_camera_pixel_size,
            zoom_factor=epie_data.zoom_factor,
        )

    iterations: Int[Array, " N"] = jnp.arange(
        num_iterations_int, dtype=jnp.int64
    )
    result_epie: EpieData = _sm_epie_core(
        epie_data=epie_data,
        iterations=iterations,
        alpha=float(alpha),
    )
    final_sample_field: Complex[Array, " Hs Ws"] = result_epie.sample
    final_probe_field: Complex[Array, " H W"] = result_epie.probe
    output_dx: Float[Array, " "] = result_epie.effective_dx
    final_sample: SampleFunction = make_sample_function(
        sample=final_sample_field, dx=output_dx
    )
    final_lightwave: OpticalWavefront = make_optical_wavefront(
        field=final_probe_field,
        wavelength=guess_lightwave.wavelength,
        dx=output_dx,
        z_position=guess_lightwave.z_position,
    )
    loss_val: Float[Array, " "] = jnp.array(0.0)
    sample_shape: tuple[int, ...] = (
        *final_sample_field.shape,
        num_iterations_int,
    )
    probe_shape: tuple[int, ...] = (
        *final_probe_field.shape,
        num_iterations_int,
    )
    intermediate_samples: Complex[Array, " Hs Ws N"] = jnp.broadcast_to(
        final_sample_field[..., None], sample_shape
    )
    intermediate_lightwaves: Complex[Array, " H W N"] = jnp.broadcast_to(
        final_probe_field[..., None], probe_shape
    )
    intermediate_zoom_factors: Float[Array, " N"] = jnp.full(
        num_iterations_int, zoom_factor
    )
    intermediate_aperture_diameters: Float[Array, " N"] = jnp.full(
        num_iterations_int, aperture_diameter
    )
    intermediate_travel_distances: Float[Array, " N"] = jnp.full(
        num_iterations_int, travel_distance
    )
    intermediate_aperture_centers: Float[Array, " 2 N"] = jnp.broadcast_to(
        aperture_center[:, None], (2, num_iterations_int)
    )
    iteration_numbers: Float[Array, " N"] = jnp.arange(
        num_iterations_int, dtype=jnp.float64
    )
    losses_arr: Float[Array, " N"] = jnp.full(num_iterations_int, loss_val)
    losses: Float[Array, " N 2"] = jnp.stack(
        [iteration_numbers, losses_arr], axis=1
    )

    if is_epie_resume:
        intermediate_samples = jnp.concatenate(
            [prev_intermediate_samples, intermediate_samples], axis=-1
        )
        intermediate_lightwaves = jnp.concatenate(
            [prev_intermediate_lightwaves, intermediate_lightwaves], axis=-1
        )
        intermediate_zoom_factors = jnp.concatenate(
            [prev_intermediate_zoom_factors, intermediate_zoom_factors],
            axis=-1,
        )
        intermediate_aperture_diameters = jnp.concatenate(
            [
                prev_intermediate_aperture_diameters,
                intermediate_aperture_diameters,
            ],
            axis=-1,
        )
        intermediate_aperture_centers = jnp.concatenate(
            [
                prev_intermediate_aperture_centers,
                intermediate_aperture_centers,
            ],
            axis=-1,
        )
        intermediate_travel_distances = jnp.concatenate(
            [
                prev_intermediate_travel_distances,
                intermediate_travel_distances,
            ],
            axis=-1,
        )
        losses = jnp.concatenate([prev_losses, losses], axis=0)
    positions_meters: Float[Array, " N 2"] = epie_data.positions * output_dx
    result: PtychographyReconstruction = make_ptychography_reconstruction(
        sample=final_sample,
        lightwave=final_lightwave,
        translated_positions=positions_meters,
        zoom_factor=zoom_factor,
        aperture_diameter=aperture_diameter,
        aperture_center=aperture_center,
        travel_distance=travel_distance,
        intermediate_samples=intermediate_samples,
        intermediate_lightwaves=intermediate_lightwaves,
        intermediate_zoom_factors=intermediate_zoom_factors,
        intermediate_aperture_diameters=intermediate_aperture_diameters,
        intermediate_aperture_centers=intermediate_aperture_centers,
        intermediate_travel_distances=intermediate_travel_distances,
        losses=losses,
    )
    return result
