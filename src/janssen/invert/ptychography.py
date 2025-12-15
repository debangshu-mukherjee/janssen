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
            intermediate_aperture_centers=combined_intermediate_aperture_centers,
            intermediate_travel_distances=(
                combined_intermediate_travel_distances
            ),
            losses=combined_losses,
        )
    )
    return full_and_intermediate


def _sm_epie_core(  # noqa: PLR0915
    epie_data: EpieData,
    iterations: Int[Array, " N"],
    alpha: float = 1.0,
) -> EpieData:
    """FFT-based ePIE core algorithm using preprocessed data.

    Pure JAX implementation of ePIE reconstruction. This function is
    JIT-compatible and uses vmap for parallel updates across positions.

    Parameters
    ----------
    epie_data : EpieData
        Preprocessed data from init_simple_epie containing:

        - diffraction_patterns: Scaled camera images (N, H, W)
        - probe: Initial probe with aperture (H, W)
        - sample: Initial sample estimate (Hs, Ws)
        - positions: Scan positions in pixels
    iterations : Int[Array, " N"]
        Array of iteration indices to scan over. Length determines number
        of iterations. Must be created outside JIT context to avoid tracer
        issues with dynamically-sized arrays.
    alpha : float, optional
        ePIE step size parameter. Default is 1.0.

    Returns
    -------
    EpieData
        Updated EpieData with reconstructed sample and probe.
        All other fields are preserved from input.

    Notes
    -----
    **Algorithm Overview**

    For each iteration:

    1. vmap over all positions to compute updates in parallel
    2. Scatter-add object updates to full FOV using lax.scan
    3. Average object updates by overlap count
    4. Average probe updates across all positions
    5. Apply updates to object and probe

    Uses ``lax.scan`` over iterations for efficient JIT compilation.

    **ePIE Update Equations**

    For each position:

    1. ``exit_wave = object_patch * probe``
    2. ``detector = FFT(exit_wave)``
    3. ``detector_new = detector * sqrt(measured) / |detector|``
    4. ``exit_wave_new = IFFT(detector_new)``
    5. ``diff = exit_wave_new - exit_wave``
    6. ``obj_update = alpha * conj(probe) * diff / max(|probe|^2)``
    7. ``probe_update = alpha * conj(obj) * diff / max(|obj|^2)``

    **Scatter-Add Pattern**

    Object updates from overlapping positions are accumulated using
    ``lax.dynamic_slice`` and ``lax.dynamic_update_slice``, then averaged
    by dividing by the overlap count at each pixel.

    See Also
    --------
    init_simple_epie : Preprocess data for FFT-compatible ePIE.
    simple_microscope_epie : High-level orchestration function.
    """
    diffraction_patterns: Float[Array, " N H W"] = (
        epie_data.diffraction_patterns
    )
    sample_field: Complex[Array, " Hs Ws"] = epie_data.sample
    probe_field: Complex[Array, " H W"] = epie_data.probe
    positions: Float[Array, " N 2"] = epie_data.positions
    probe_size_y: int = probe_field.shape[0]
    probe_size_x: int = probe_field.shape[1]
    sample_size_y: int = sample_field.shape[0]
    sample_size_x: int = sample_field.shape[1]
    eps: float = 1e-8

    def _compute_single_update(
        obj: Complex[Array, " Hs Ws"],
        probe: Complex[Array, " H W"],
        measurement: Float[Array, " H W"],
        pos: Float[Array, " 2"],
    ) -> Tuple[
        Complex[Array, " H W"],
        Complex[Array, " H W"],
        Int[Array, " 2"],
    ]:
        """Compute ePIE update for single position (vmappable)."""
        start_x: Int[Array, " "] = jnp.floor(
            pos[0] - 0.5 * probe_size_x
        ).astype(jnp.int32)
        start_y: Int[Array, " "] = jnp.floor(
            pos[1] - 0.5 * probe_size_y
        ).astype(jnp.int32)
        obj_patch: Complex[Array, " H W"] = lax.dynamic_slice(
            obj, (start_y, start_x), (probe_size_y, probe_size_x)
        )
        exit_wave: Complex[Array, " H W"] = obj_patch * probe
        exit_wave_ft: Complex[Array, " H W"] = jnp.fft.fftshift(
            jnp.fft.fft2(exit_wave)
        )
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
        diff: Complex[Array, " H W"] = exit_wave_updated - exit_wave
        probe_conj: Complex[Array, " H W"] = jnp.conj(probe)
        probe_intensity: Float[Array, " H W"] = jnp.abs(probe) ** 2
        probe_max_intensity: Float[Array, " "] = jnp.max(probe_intensity)
        obj_update: Complex[Array, " H W"] = (
            alpha * probe_conj * diff / (probe_max_intensity + eps)
        )
        obj_conj: Complex[Array, " H W"] = jnp.conj(obj_patch)
        obj_intensity: Float[Array, " H W"] = jnp.abs(obj_patch) ** 2
        obj_max_intensity: Float[Array, " "] = jnp.max(obj_intensity)
        probe_update: Complex[Array, " H W"] = (
            alpha * obj_conj * diff / (obj_max_intensity + eps)
        )
        start_indices: Int[Array, " 2"] = jnp.array(
            [start_y, start_x], dtype=jnp.int32
        )
        return obj_update, probe_update, start_indices

    def _scatter_add_updates(
        carry: Tuple[Complex[Array, " Hs Ws"], Float[Array, " Hs Ws"]],
        inputs: Tuple[Complex[Array, " H W"], Int[Array, " 2"]],
    ) -> Tuple[
        Tuple[Complex[Array, " Hs Ws"], Float[Array, " Hs Ws"]],
        None,
    ]:
        """Scatter-add single update to accumulator with overlap counting."""
        acc_update, acc_count = carry
        update, idx = inputs
        current_patch: Complex[Array, " H W"] = lax.dynamic_slice(
            acc_update, (idx[0], idx[1]), (probe_size_y, probe_size_x)
        )
        acc_update = lax.dynamic_update_slice(
            acc_update, current_patch + update, (idx[0], idx[1])
        )
        ones_patch: Float[Array, " H W"] = jnp.ones(
            (probe_size_y, probe_size_x), dtype=jnp.float64
        )
        current_count: Float[Array, " H W"] = lax.dynamic_slice(
            acc_count, (idx[0], idx[1]), (probe_size_y, probe_size_x)
        )
        acc_count = lax.dynamic_update_slice(
            acc_count, current_count + ones_patch, (idx[0], idx[1])
        )
        return (acc_update, acc_count), None

    def _epie_one_iteration(
        carry: Tuple[Complex[Array, " Hs Ws"], Complex[Array, " H W"]],
        _iter_idx: Int[Array, " "],
    ) -> Tuple[
        Tuple[Complex[Array, " Hs Ws"], Complex[Array, " H W"]],
        None,
    ]:
        """One complete sweep over all positions using vmap."""
        obj, probe = carry
        obj_updates, probe_updates, start_indices = jax.vmap(
            lambda m, p: _compute_single_update(obj, probe, m, p)
        )(diffraction_patterns, positions)
        obj_update_full: Complex[Array, " Hs Ws"] = jnp.zeros(
            (sample_size_y, sample_size_x), dtype=jnp.complex128
        )
        obj_count: Float[Array, " Hs Ws"] = jnp.zeros(
            (sample_size_y, sample_size_x), dtype=jnp.float64
        )
        (obj_update_sum, obj_overlap_count), _ = lax.scan(
            _scatter_add_updates,
            (obj_update_full, obj_count),
            (obj_updates, start_indices),
        )
        obj_overlap_count_safe: Float[Array, " Hs Ws"] = jnp.maximum(
            obj_overlap_count, 1.0
        )
        obj_update_avg: Complex[Array, " Hs Ws"] = (
            obj_update_sum / obj_overlap_count_safe
        )
        probe_update_avg: Complex[Array, " H W"] = jnp.mean(
            probe_updates, axis=0
        )
        obj_new: Complex[Array, " Hs Ws"] = obj + obj_update_avg
        probe_new: Complex[Array, " H W"] = probe + probe_update_avg
        return (obj_new, probe_new), None

    init_carry: Tuple[Complex[Array, " Hs Ws"], Complex[Array, " H W"]] = (
        sample_field,
        probe_field,
    )
    final_carry, _ = lax.scan(_epie_one_iteration, init_carry, iterations)
    final_sample_field: Complex[Array, " Hs Ws"]
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
    alpha: Float[Array, " "] = params.learning_rate
    num_iterations_int: int = int(num_iterations)
    start_iteration: Int[Array, " "] = jnp.array(
        prev_losses.shape[0], dtype=jnp.int64
    )
    sample_dx: Float[Array, " "] = guess_sample.dx
    probe_size_y: int = guess_lightwave.field.shape[0]
    probe_size_x: int = guess_lightwave.field.shape[1]
    epie_data: EpieData = init_simple_epie(
        experimental_data=experimental_data,
        probe_size=(probe_size_y, probe_size_x),
        wavelength=guess_lightwave.wavelength,
        zoom_factor=zoom_factor,
        aperture_diameter=aperture_diameter,
        travel_distance=travel_distance,
        camera_pixel_size=camera_pixel_size,
    )
    is_resume: bool = int(prev_losses.shape[0]) > 0

    if is_resume:
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
    final_sample: SampleFunction = make_sample_function(
        sample=final_sample_field, dx=sample_dx
    )
    final_lightwave: OpticalWavefront = make_optical_wavefront(
        field=final_probe_field,
        wavelength=guess_lightwave.wavelength,
        dx=guess_lightwave.dx,
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
    intermediate_samples_new: Complex[Array, " Hs Ws N"] = jnp.broadcast_to(
        final_sample_field[..., None], sample_shape
    )
    intermediate_lightwaves_new: Complex[Array, " H W N"] = jnp.broadcast_to(
        final_probe_field[..., None], probe_shape
    )
    intermediate_zoom_factors_new: Float[Array, " N"] = jnp.full(
        num_iterations_int, zoom_factor
    )
    intermediate_aperture_diameters_new: Float[Array, " N"] = jnp.full(
        num_iterations_int, aperture_diameter
    )
    intermediate_travel_distances_new: Float[Array, " N"] = jnp.full(
        num_iterations_int, travel_distance
    )
    intermediate_aperture_centers_new: Float[Array, " 2 N"] = jnp.broadcast_to(
        aperture_center[:, None], (2, num_iterations_int)
    )

    iteration_numbers: Float[Array, " N"] = start_iteration + jnp.arange(
        num_iterations_int, dtype=jnp.float64
    )
    losses_new: Float[Array, " N"] = jnp.full(num_iterations_int, loss_val)
    losses: Float[Array, " N 2"] = jnp.stack(
        [iteration_numbers, losses_new], axis=1
    )

    combined_intermediate_samples: Complex[Array, " H W S"] = jnp.concatenate(
        [prev_intermediate_samples, intermediate_samples_new], axis=-1
    )
    combined_intermediate_lightwaves: Complex[Array, " H W S"] = (
        jnp.concatenate(
            [prev_intermediate_lightwaves, intermediate_lightwaves_new],
            axis=-1,
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
                intermediate_aperture_centers_new,
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
    result: PtychographyReconstruction = make_ptychography_reconstruction(
        sample=final_sample,
        lightwave=final_lightwave,
        translated_positions=translated_positions,
        zoom_factor=zoom_factor,
        aperture_diameter=aperture_diameter,
        aperture_center=aperture_center,
        travel_distance=travel_distance,
        intermediate_samples=combined_intermediate_samples,
        intermediate_lightwaves=combined_intermediate_lightwaves,
        intermediate_zoom_factors=combined_intermediate_zoom_factors,
        intermediate_aperture_diameters=combined_intermediate_aperture_diameters,
        intermediate_aperture_centers=combined_intermediate_aperture_centers,
        intermediate_travel_distances=combined_intermediate_travel_distances,
        losses=combined_losses,
    )
    return result
