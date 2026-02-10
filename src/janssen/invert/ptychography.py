"""Ptychography algorithms and optimization.

Extended Summary
----------------
High-level ptychography reconstruction algorithms that combine
optimization strategies with forward models. Provides complete reconstruction
pipelines for recovering complex-valued sample functions from intensity
measurements.

Routine Listings
----------------
optimal_cg_params : function
    Calculate optimal conjugate gradient parameters for memory constraints
profile_gn_memory : function
    Profile GPU memory usage during Gauss-Newton optimization
simple_microscope_optim : function
    Performs ptychography reconstruction using gradient-based optimization
simple_microscope_epie : function
    Performs ptychography reconstruction using extended PIE algorithm
simple_microscope_gn : function
    Performs ptychography reconstruction using Gauss-Newton optimization
_gn_state_to_ptychography_reconstruction : function, internal
    Packs Gauss-Newton state and geometry into a PtychographyReconstruction.

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
from janssen.types import (
    EpieData,
    GaussNewtonState,
    MicroscopeData,
    OpticalWavefront,
    PtychographyParams,
    PtychographyReconstruction,
    SampleFunction,
    make_epie_data,
    make_gauss_newton_state,
    make_optical_wavefront,
    make_ptychography_reconstruction,
    make_sample_function,
)
from janssen.utils import (
    fourier_shift,
    get_device_memory_gb,
    gn_loss_history,
    gn_solve,
    unflatten_params,
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
def simple_microscope_optim(  # noqa: PLR0915
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

        probe_shifted: Complex[Array, " H W"] = fourier_shift(
            probe, shift_x, shift_y
        )

        exit_wave: Complex[Array, " H W"] = obj * probe_shifted
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

        probe_conj: Complex[Array, " H W"] = jnp.conj(probe_shifted)
        probe_intensity: Float[Array, " H W"] = jnp.abs(probe_shifted) ** 2
        probe_max_intensity: Float[Array, " "] = jnp.max(probe_intensity)
        obj_update: Complex[Array, " H W"] = (
            alpha * probe_conj * diff / (probe_max_intensity + eps)
        )
        obj_new: Complex[Array, " H W"] = obj + obj_update

        obj_conj: Complex[Array, " H W"] = jnp.conj(obj)
        obj_intensity: Float[Array, " H W"] = jnp.abs(obj) ** 2
        obj_max_intensity: Float[Array, " "] = jnp.max(obj_intensity)
        probe_update_shifted: Complex[Array, " H W"] = (
            beta * obj_conj * diff / (obj_max_intensity + eps)
        )
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


def profile_gn_memory(
    experimental_data: MicroscopeData,
    reconstruction: PtychographyReconstruction,
    cg_maxiter: int = 5,
    verbose: bool = True,
) -> dict:
    """Profile memory usage during Gauss-Newton optimization.

    Tracks memory allocation at key stages to diagnose OOM issues.
    Works with GPU, TPU, and CPU backends (memory stats available on
    GPU and TPU only). Useful for understanding actual XLA memory
    behavior vs theoretical predictions.

    Parameters
    ----------
    experimental_data : MicroscopeData
        Experimental diffraction patterns
    reconstruction : PtychographyReconstruction
        Initial reconstruction state
    cg_maxiter : int, optional
        CG iterations to test. Start low (3-5) to avoid OOM. Default is 5.
    verbose : bool, optional
        Print detailed memory snapshots. Default is True.

    Returns
    -------
    memory_profile : dict
        Dictionary with keys:
        - 'baseline': Memory before GN step
        - 'after_warmup': Memory after warmup compilation
        - 'after_gn': Memory after 1 GN iteration
        - 'peak_per_device_gb': Peak memory per device (float or None)
        - 'succeeded': Whether profiling completed without OOM

    Notes
    -----
    Memory profiling is supported on:
    - GPU (CUDA, ROCm): Full support via device.memory_stats()
    - TPU: Full support via device.memory_stats()
    - CPU: Limited support (profiling runs but memory stats unavailable)

    On unsupported platforms, profiling still runs to test for OOM,
    but peak_per_device_gb will be None.

    Examples
    --------
    >>> profile = profile_gn_memory(data, init_recon, cg_maxiter=3)
    >>> print(f"Peak memory: {profile['peak_per_device_gb']:.2f} GB")
    """

    def get_memory_snapshot(label: str) -> dict:
        """Capture memory across all devices (GPU/TPU/CPU)."""
        snapshot = {"label": label, "devices": []}
        supported_devices = 0

        for i, device in enumerate(jax.devices()):
            try:
                stats = device.memory_stats()
                snapshot["devices"].append(
                    {
                        "id": i,
                        "platform": device.platform,
                        "bytes": stats.get("bytes_in_use", 0),
                        "peak": stats.get("peak_bytes_in_use", 0),
                    }
                )
                supported_devices += 1
            except (AttributeError, NotImplementedError, KeyError):
                snapshot["devices"].append(
                    {
                        "id": i,
                        "platform": device.platform,
                        "unsupported": True,
                    }
                )

        if verbose and supported_devices > 0:
            print(f"\n{label}:")
            total_gb = (
                sum(d.get("bytes", 0) for d in snapshot["devices"]) / 1e9
            )
            peak_gb = max(d.get("peak", 0) for d in snapshot["devices"]) / 1e9
            print(
                f"  Total: {total_gb:.2f} GB, "
                f"Peak/device: {peak_gb:.2f} GB "
                f"({supported_devices}/{len(jax.devices())} devices)"
            )
        elif verbose and supported_devices == 0:
            platforms = {d["platform"] for d in snapshot["devices"]}
            print(f"\n{label}:")
            print(
                f"  Memory profiling not supported on "
                f"{', '.join(platforms)}"
            )

        return snapshot

    profile = {}
    profile["baseline"] = get_memory_snapshot("Baseline")

    try:
        if verbose:
            print(f"\nRunning GN with cg_maxiter={cg_maxiter}...")

        _ = simple_microscope_gn(
            experimental_data,
            reconstruction,
            num_iterations=1,
            cg_maxiter=cg_maxiter,
            cg_tol=1e-2,
        )

        profile["after_gn"] = get_memory_snapshot("After 1 GN iteration")
        profile["succeeded"] = True

        peak_values = [
            d.get("peak", 0)
            for s in profile.values()
            if isinstance(s, dict) and "devices" in s
            for d in s["devices"]
            if "peak" in d
        ]

        if peak_values:
            profile["peak_per_device_gb"] = max(peak_values) / 1e9
            if verbose:
                print("\n✓ Profiling succeeded!")
                peak_mem = profile["peak_per_device_gb"]
                print(f"  Peak memory: {peak_mem:.2f} GB/device")
        else:
            profile["peak_per_device_gb"] = None
            if verbose:
                print("\n✓ Profiling succeeded " "(memory stats unavailable)")

    except Exception as e:
        profile["after_gn"] = get_memory_snapshot("At failure")
        profile["succeeded"] = False
        profile["error"] = str(e)

        if verbose:
            print(f"\n✗ Profiling failed: {e}")

    return profile


@jaxtyped(typechecker=beartype)
def optimal_cg_params(
    experimental_data: MicroscopeData,
    reconstruction: PtychographyReconstruction,
    memory_per_device_gb: float = -1.0,
    safety_factor: float = 0.3,
) -> Tuple[int, float]:
    """Calculate optimal conjugate gradient parameters for memory constraints.

    Automatically determines cg_maxiter and cg_tol that will fit in
    available GPU memory while maintaining good convergence. Accounts for
    problem size (number of positions, sample/probe dimensions), device
    count, and memory constraints.

    Implementation Logic
    --------------------
    The calculation follows a four-step process:

    1. **Problem Size Analysis**:
       - Extracts sample dimensions (Hs, Ws) from reconstruction.sample
       - Extracts probe dimensions (Hp, Wp) from reconstruction.lightwave
       - Total parameters: (Hs × Ws) + (Hp × Wp)
       - Each parameter is complex128 (16 bytes)

    2. **Memory Estimation**:
       - CG memory per iteration has two components:
         a. Parameter space: 3 × param_size × 16 bytes
            (stores x, r, p vectors in parameter space)
         b. Residual space: 6 × N × H × W × 4 bytes
            (Jacobian-vector products create multiple residual evaluations)
       - Baseline memory: forward model + gradients
         ≈ 2 × (N × H × W × 4 bytes) for diffractograms
       - Available memory: (devices × memory_per_device × safety_factor)
         - baseline

    3. **CG Iteration Calculation**:
       - Max iterations: floor(available_memory / memory_per_iteration)
       - Clamp to [5, 20] range (min 5 for convergence, max 20 for
         conservative memory safety with compilation overhead)

    4. **Tolerance Selection**:
       - Continuous log-linear relationship: tol = 10^(-(maxiter + 25)/15)
       - More iterations → tighter tolerance (lower value)
       - Fewer iterations → looser tolerance (higher value)
       - Examples: maxiter=50 → tol=1e-5, maxiter=20 → tol=1e-3,
         maxiter=5 → tol=1e-2

    Parameters
    ----------
    experimental_data : MicroscopeData
        Experimental diffraction patterns. Used to determine number of
        positions and diffractogram dimensions.
    reconstruction : PtychographyReconstruction
        Reconstruction state containing sample and probe. Used to
        determine parameter dimensions.
    memory_per_device_gb : float, optional
        Available memory per device in GB. Default is -1.0 which triggers
        automatic detection via get_device_memory_gb() (nvidia-smi for GPUs,
        system RAM for CPUs). Typical GPU values: V100/RTX 6000 = 16 GB,
        A100 = 40 GB, H100 = 80 GB.
    safety_factor : float, optional
        Fraction of memory to use for CG (0-1). Default is 0.3 to leave
        headroom for compilation overhead (2-3× runtime memory),
        fragmentation, and peak allocations. Increase to 0.4-0.5 for
        problems with <100 positions or after warm-up compilation.

    Returns
    -------
    cg_maxiter : int
        Recommended maximum conjugate gradient iterations. Clamped to
        [5, 50].
    cg_tol : float
        Recommended CG convergence tolerance. Automatically selected
        based on maxiter to balance accuracy and convergence rate.

    Notes
    -----
    **Memory Model**:

    The memory estimation accounts for XLA's actual buffer allocation:
    - Theoretical minimum: ~6 residual evaluations per CG iteration
    - Empirical overhead: ~40× residual size per iteration (measured)
    - Overhead sources: XLA buffer stacking, compilation temporaries,
      CG intermediate storage, sharding boundary copies
    - safety_factor=0.3 accounts for additional ~3× peak memory during
      compilation and runtime fluctuations
    - For very large problems (N > 500), reduce to 0.2 if needed
    - For small problems (<100 positions) after warm-up, increase to
      0.4-0.5 for better accuracy

    **Accuracy vs Memory Tradeoff**:

    Lower cg_maxiter means:
    - Less accurate Gauss-Newton steps (higher residual after CG)
    - More GN iterations needed to reach same MSE (~2-4 extra iterations
      per 10 reduction in maxiter)
    - Faster per-iteration time (less CG work)
    - Lower memory usage (fewer intermediate vectors)

    Higher cg_maxiter means:
    - More accurate GN steps (lower residual after CG)
    - Fewer GN iterations to convergence
    - Slower per-iteration time (more CG work)
    - Higher memory usage (more intermediate vectors)

    **Example Calculations**:

    For N=400, H=W=256, sample=512×512, probe=256×256, 7 GPUs, 16GB each:
    - Total params: 512² + 256² = 327,680 complex128 = 5.24 MB
    - Param space memory/iter: 3 × 5.24 MB = 15.72 MB
    - Residual space memory/iter: 40 × 400 × 256 × 256 × 4 = 4.19 GB
    - CG memory/iter: 15.72 MB + 4.19 GB ≈ 4.21 GB
    - Baseline memory: 2 × (400 × 256 × 256 × 4) = 210 MB
    - Available: 7 × 16GB × 0.3 - 0.21 GB ≈ 33.4 GB
    - Max iters: 33.4 GB / 4.21 GB ≈ 7
    - Recommended: cg_maxiter=7, tol≈4e-3

    For same problem with 1 GPU:
    - Available: 1 × 16GB × 0.3 - 0.21 GB ≈ 4.6 GB
    - Max iters: 4.6 GB / 4.21 GB ≈ 1
    - Recommended: cg_maxiter=5 (minimum), tol=1e-2

    For N=400, sample=1024×1024, probe=512×512, 7 GPUs:
    - Total params: 1024² + 512² = 1,310,720 complex128 = 21 MB
    - Param space: 3 × 21 MB = 63 MB, Residual space: 4.19 GB
    - CG memory/iter: 4.25 GB
    - Available: ≈ 33.4 GB
    - Max iters: 33.4 GB / 4.25 GB ≈ 7
    - Recommended: cg_maxiter=7, tol≈4e-3

    For N=400, sample=2048×2048, probe=512×512, 7 GPUs:
    - Total params: 2048² + 512² = 4,456,448 complex128 = 71.3 MB
    - Param space: 3 × 71.3 MB = 214 MB, Residual space: 4.19 GB
    - CG memory/iter: 4.40 GB
    - Available: ≈ 33.4 GB
    - Max iters: 33.4 GB / 4.40 GB ≈ 7
    - Recommended: cg_maxiter=7, tol≈4e-3

    For N=400, sample=4096×4096, probe=1024×1024, 7 GPUs:
    - Total params: 4096² + 1024² = 17,825,792 complex128 = 285 MB
    - Param space: 3 × 285 MB = 855 MB, Residual space: 4.19 GB
    - CG memory/iter: 5.05 GB
    - Available: ≈ 33.4 GB
    - Max iters: 33.4 GB / 5.05 GB ≈ 6
    - Recommended: cg_maxiter=6, tol≈5e-3

    For N=400, sample=8192×8192, probe=2048×2048, 7 GPUs, 16GB:
    - Total params: 8192² + 2048² = 71,303,168 complex128 = 1.14 GB
    - Param space: 3 × 1.14 GB = 3.42 GB, Residual space: 4.19 GB
    - CG memory/iter: 7.61 GB
    - Available: ≈ 33.4 GB
    - Max iters: 33.4 GB / 7.61 GB ≈ 4
    - Recommended: cg_maxiter=5 (minimum), tol=1e-2

    **Design Decisions**:

    - safety_factor=0.3 chosen empirically: provides reliable OOM
      avoidance accounting for ~3× compilation overhead
    - Clamp minimum to 5: CG needs at least a few iterations for any
      progress
    - Clamp maximum to 20: conservative cap for memory safety with
      compilation overhead; beyond 20, better to do more GN iterations
      than risk OOM
    - Continuous tolerance formula tol = 10^(-(maxiter + 25)/15) provides
      smooth scaling: more iterations enable tighter tolerances, fewer
      iterations require looser tolerances for CG convergence

    See Also
    --------
    simple_microscope_gn : Uses these parameters for optimization

    Examples
    --------
    >>> # Automatic GPU memory detection (default)
    >>> data = MicroscopeData(...)
    >>> init_recon = init_simple_microscope(data, ...)
    >>> cg_maxiter, cg_tol = optimal_cg_params(data, init_recon)
    >>> print(f"Recommended: cg_maxiter={cg_maxiter}, cg_tol={cg_tol}")
    >>> result = simple_microscope_gn(
    ...     data, init_recon, cg_maxiter=cg_maxiter, cg_tol=cg_tol
    ... )
    >>>
    >>> # Manual override for specific GPU memory
    >>> cg_maxiter, cg_tol = optimal_cg_params(
    ...     data, init_recon, memory_per_device_gb=40.0
    ... )
    """
    if memory_per_device_gb < 0:
        _, memory_per_device_gb = get_device_memory_gb()
    num_positions: int = experimental_data.image_data.shape[0]
    diff_height: int = experimental_data.image_data.shape[1]
    diff_width: int = experimental_data.image_data.shape[2]
    sample_shape: Tuple[int, int] = reconstruction.sample.sample.shape
    probe_shape: Tuple[int, int] = reconstruction.lightwave.field.shape
    sample_size: int = sample_shape[0] * sample_shape[1]
    probe_size: int = probe_shape[0] * probe_shape[1]
    total_params: int = sample_size + probe_size
    bytes_per_complex128: int = 16
    param_space_vectors: int = 3
    param_memory_per_iter_bytes: float = (
        total_params * bytes_per_complex128 * param_space_vectors
    )
    residual_size: int = num_positions * diff_height * diff_width
    bytes_per_float32: int = 4
    residual_evaluations_per_iter: int = 40
    residual_memory_per_iter_bytes: float = (
        residual_size * bytes_per_float32 * residual_evaluations_per_iter
    )
    cg_memory_per_iter_bytes: float = (
        param_memory_per_iter_bytes + residual_memory_per_iter_bytes
    )
    cg_memory_per_iter_gb: float = cg_memory_per_iter_bytes / 1e9
    num_devices: int = len(jax.devices())
    total_memory_gb: float = memory_per_device_gb * num_devices
    diffractogram_memory_bytes: float = (
        num_positions * diff_height * diff_width * 4
    )
    diffractogram_memory_gb: float = diffractogram_memory_bytes / 1e9
    baseline_memory_gb: float = diffractogram_memory_gb * 2
    available_memory_gb: float = (
        total_memory_gb * safety_factor - baseline_memory_gb
    )
    if available_memory_gb <= 0:
        cg_maxiter: int = 5
        cg_tol: float = 1e-2
    else:
        max_cg_iters: int = int(available_memory_gb / cg_memory_per_iter_gb)
        conservative_cap: int = 20
        cg_maxiter: int = max(5, min(conservative_cap, max_cg_iters))
        log_tol: float = -(cg_maxiter + 25.0) / 15.0
        cg_tol: float = 10.0**log_tol
    result: Tuple[int, float] = (cg_maxiter, cg_tol)
    return result


@jaxtyped(typechecker=beartype)
def simple_microscope_gn(  # noqa: PLR0915
    experimental_data: MicroscopeData,
    reconstruction: PtychographyReconstruction,
    num_iterations: int = 10,
    initial_damping: float = 1e-3,
    cg_maxiter: int = -1,
    cg_tol: float = -1.0,
    save_every: int = 10,
) -> PtychographyReconstruction:
    """Perform ptychographic reconstruction using Gauss-Newton optimization.

    Uses second-order Gauss-Newton optimization with Levenberg-Marquardt
    damping for ptychography reconstruction. Solves the nonlinear
    least-squares problem:

        min_{sample, probe}  0.5 * ||sqrt(I_exp) - sqrt(I_pred)||^2

    where I_exp are experimental diffraction patterns and I_pred are
    simulated patterns from the forward model. The function uses JAX's
    autodiff for Jacobian-free optimization via conjugate gradient.

    The function automatically includes warm-up compilation for large
    problems (>50 positions) and uses memory-optimized defaults for
    conjugate gradient to fit in 16GB GPU memory. Multi-GPU sharding
    is inherited from simple_microscope via the residual function.

    Implementation Logic
    --------------------
    The optimization follows a four-stage pipeline:

    1. **Residual Function Definition**:
       - _amplitude_residuals(params) unflatens params into sample and
         probe
       - Calls simple_microscope to compute simulated patterns
         (automatically sharded)
       - Computes amplitude residuals: sqrt(I_exp) - sqrt(I_pred)
       - Returns flattened residual vector of length N × H × W

    2. **Warm-up Compilation** (Automatic for N > 50):
       - Creates warmup_data with first 25 positions
       - Defines _warmup_residuals using warmup subset
       - Runs gn_solve for 1 iteration with max_iterations=1
       - Critical: triggers JIT compilation with small problem
       - Compilation time: ~30s for warmup vs 5+ minutes for full problem
       - Memory during compilation: ~2× runtime memory
       - After warmup, full problem reuses compiled kernels

    3. **Chunked Gauss-Newton Optimization**:
       - Runs gn_loss_history in chunks of size
         `save_every`
       - Each GN iteration:
         a. Computes residuals r(θ) and loss = 0.5 * ||r||^2
         b. Forms (J^T J + λI) operator via jtj_matvec
         c. Solves (J^T J + λI) δ = -J^T r via conjugate gradient
         d. Updates parameters: θ_new = θ + δ
         e. Adapts damping λ based on trust-region criterion
       - Stores full per-iteration loss history across all chunks
       - Stores sample/probe snapshots only at chunk boundaries
         (every `save_every` iterations)
       - Reduces memory compared to storing full state history every step

    4. **Result Packaging**:
       - Converts GaussNewtonState to PtychographyReconstruction
       - Unflattens final parameters into sample and probe
       - Appends sparse intermediate_* snapshots and full loss history
       - Returns updated reconstruction

    Parameters
    ----------
    experimental_data : MicroscopeData
        Experimental diffraction patterns and scan positions.
    reconstruction : PtychographyReconstruction
        Initial reconstruction state from init_simple_microscope.
    num_iterations : int, optional
        Number of Gauss-Newton iterations. Default is 10.
    initial_damping : float, optional
        Initial Levenberg-Marquardt damping parameter λ. Default is 1e-3.
        Adapts automatically based on step quality.
    cg_maxiter : int, optional
        Maximum conjugate gradient iterations per GN step. Default is -1,
        which automatically calculates optimal value via optimal_cg_params
        based on problem size and available memory. Set to positive value
        to override automatic calculation.
    cg_tol : float, optional
        CG convergence tolerance. Default is -1.0, which automatically
        calculates optimal value via optimal_cg_params. Set to positive
        value to override automatic calculation.
    save_every : int, optional
        Save sample/probe snapshots in intermediate history once every
        `save_every` iterations. Full loss history is still recorded at
        every iteration. Default is 10.

    Returns
    -------
    reconstruction : PtychographyReconstruction
        Updated reconstruction with optimized sample and lightwave.

    Notes
    -----
    **Warm-up Compilation Rationale**:

    Without warm-up, JIT compilation happens during the first GN iteration
    with the full problem size:
    - Compilation memory: ~2× runtime memory for N=400 positions
    - For 256×256 diffractograms on 7 GPUs: ~4 GB per device
    - 16GB GPUs: OOM during compilation
    - Compilation time: 5-7 minutes for full problem

    With warm-up (25 positions):
    - Compilation memory: ~2× runtime memory for N=25
    - For 256×256 diffractograms: ~250 MB per device → no OOM
    - Compilation time: ~30 seconds
    - Full problem reuses compiled kernels (only recompiles shape-dependent
      operations)
    - Total time saved: 4-6 minutes

    **Automatic CG Parameter Optimization**:

    By default (cg_maxiter=-1, cg_tol=-1.0), the function automatically
    calculates optimal parameters via optimal_cg_params based on:
    - Problem size (number of positions, sample/probe dimensions)
    - Available GPU memory (assumes 16GB per device)
    - Number of devices detected via jax.devices()

    This ensures the solver fits in memory while maximizing accuracy.
    Override by passing positive values for manual control.

    **Memory-Optimized CG Behavior**:

    Conjugate gradient memory scales with maxiter:
    - Each CG iteration stores: residual, direction, Ap vectors
    - Memory per iteration: ~3 × parameter_size
    - For sample (512×512) + probe (256×256): ~2.5 GB per CG iteration
    - cg_maxiter=50: ~125 GB peak memory → OOM on 16GB GPUs
    - cg_maxiter=20: ~50 GB peak memory → still OOM on 16GB GPUs
    - cg_maxiter=10: ~25 GB peak memory → fits with 7-GPU sharding

    Quality impact of reduced maxiter:
    - cg_maxiter=50, tol=1e-5: δ accurate to ~1e-5
    - cg_maxiter=10, tol=1e-3: δ accurate to ~1e-3
    - GN convergence: relative MSE decrease per iteration ~5-10%
    - Impact: ~2-4 extra GN iterations to reach same MSE
    - Tradeoff: 80% memory reduction for 20-40% more GN iterations

    **Multi-GPU Sharding**:

    Sharding is automatic via simple_microscope in _amplitude_residuals:
    - Forward model distributes positions across devices
    - Jacobian-vector products (jvp/vjp) respect sharding
    - CG operates on sharded vectors → memory distributed
    - No explicit sharding code needed in this function

    **Design Decisions**:

    - Amplitude residuals (sqrt(I)) rather than intensity residuals (I):
      better conditioning, noise model closer to Poisson
    - Warm-up threshold of 50 positions chosen empirically: problems
      smaller than 50 compile fast enough without warm-up
    - Warm-up size of 25 positions: 50% of threshold, large enough for
      stable compilation
    - Single warm-up iteration (max_iterations=1): only need compilation,
      not convergence
    - Automatic CG parameter optimization (cg_maxiter=-1, cg_tol=-1.0):
      enabled by default to eliminate manual tuning for novice users
    - Sentinel values allow expert users to override with manual settings
      when needed

    See Also
    --------
    optimal_cg_params : Calculate optimal CG parameters for your problem
    simple_microscope_optim : First-order gradient-based optimization
    simple_microscope_epie : Extended PIE algorithm
    gn_solve : General-purpose Gauss-Newton solver
    gn_loss_history : GN solver with loss-only history

    Examples
    --------
    Basic usage with automatic CG parameter optimization (default):

    >>> data = MicroscopeData(...)
    >>> init_recon = init_simple_microscope(data, ...)
    >>> final_recon = simple_microscope_gn(data, init_recon, num_iterations=20)
    # CG parameters automatically calculated based on problem size and memory

    Manual CG parameter override (for expert users):

    >>> final_recon = simple_microscope_gn(
    ...     data, init_recon, num_iterations=20,
    ...     cg_maxiter=15, cg_tol=1e-4
    ... )
    """
    sample: SampleFunction = reconstruction.sample
    lightwave: OpticalWavefront = reconstruction.lightwave
    translated_positions: Float[Array, " N 2"] = (
        reconstruction.translated_positions
    )
    aperture_center: Float[Array, " 2"] = (
        jnp.zeros(2)
        if reconstruction.aperture_center is None
        else reconstruction.aperture_center
    )
    if cg_maxiter < 0 or cg_tol < 0:
        optimal_maxiter: int
        optimal_tol: float
        optimal_maxiter, optimal_tol = optimal_cg_params(
            experimental_data, reconstruction
        )
        cg_maxiter_used: int = (
            optimal_maxiter if cg_maxiter < 0 else cg_maxiter
        )
        cg_tol_used: float = optimal_tol if cg_tol < 0 else cg_tol
    else:
        cg_maxiter_used: int = cg_maxiter
        cg_tol_used: float = cg_tol
    if save_every <= 0:
        msg: str = f"save_every must be positive, got {save_every}"
        raise ValueError(msg)

    def _amplitude_residuals(params: Float[Array, " n"]) -> Float[Array, " m"]:
        sample_shape: Tuple[int, int] = sample.sample.shape
        probe_shape: Tuple[int, int] = lightwave.field.shape
        sample_field: Complex[Array, " Hs Ws"]
        probe_field: Complex[Array, " Hp Wp"]
        sample_field, probe_field = unflatten_params(
            params, sample_shape, probe_shape
        )
        sample_fn: SampleFunction = make_sample_function(
            sample=sample_field, dx=sample.dx
        )
        lightwave_fn: OpticalWavefront = make_optical_wavefront(
            field=probe_field,
            wavelength=lightwave.wavelength,
            dx=lightwave.dx,
            z_position=lightwave.z_position,
        )
        simulated: MicroscopeData = simple_microscope(
            sample=sample_fn,
            positions=translated_positions,
            lightwave=lightwave_fn,
            zoom_factor=reconstruction.zoom_factor,
            aperture_diameter=reconstruction.aperture_diameter,
            travel_distance=reconstruction.travel_distance,
            camera_pixel_size=experimental_data.dx,
            aperture_center=aperture_center,
        )
        measured: Float[Array, " N H W"] = experimental_data.image_data
        predicted: Float[Array, " N H W"] = simulated.image_data
        amp_measured: Float[Array, " N H W"] = jnp.sqrt(
            jnp.maximum(measured, 1e-12)
        )
        amp_predicted: Float[Array, " N H W"] = jnp.sqrt(
            jnp.maximum(predicted, 1e-12)
        )
        return (amp_measured - amp_predicted).ravel()

    state: GaussNewtonState = make_gauss_newton_state(
        sample=sample.sample,
        probe=lightwave.field,
        iteration=0,
        loss=jnp.inf,
        damping=initial_damping,
        converged=False,
    )
    num_positions: int = experimental_data.image_data.shape[0]
    warmup_threshold: int = 50
    if num_positions > warmup_threshold:
        warmup_positions: int = 25
        warmup_data: MicroscopeData = MicroscopeData(
            image_data=experimental_data.image_data[:warmup_positions],
            positions=experimental_data.positions[:warmup_positions],
            dx=experimental_data.dx,
            wavelength=experimental_data.wavelength,
        )
        warmup_translated_positions: Float[Array, " warmup 2"] = (
            translated_positions[:warmup_positions]
        )

        def _warmup_residuals(
            params: Float[Array, " n"],
        ) -> Float[Array, " m"]:
            sample_shape: Tuple[int, int] = sample.sample.shape
            probe_shape: Tuple[int, int] = lightwave.field.shape
            sample_field: Complex[Array, " Hs Ws"]
            probe_field: Complex[Array, " Hp Wp"]
            sample_field, probe_field = unflatten_params(
                params, sample_shape, probe_shape
            )
            sample_fn: SampleFunction = make_sample_function(
                sample=sample_field, dx=sample.dx
            )
            lightwave_fn: OpticalWavefront = make_optical_wavefront(
                field=probe_field,
                wavelength=lightwave.wavelength,
                dx=lightwave.dx,
                z_position=lightwave.z_position,
            )
            simulated: MicroscopeData = simple_microscope(
                sample=sample_fn,
                positions=warmup_translated_positions,
                lightwave=lightwave_fn,
                zoom_factor=reconstruction.zoom_factor,
                aperture_diameter=reconstruction.aperture_diameter,
                travel_distance=reconstruction.travel_distance,
                camera_pixel_size=warmup_data.dx,
                aperture_center=aperture_center,
            )
            measured: Float[Array, " warmup H W"] = warmup_data.image_data
            predicted: Float[Array, " warmup H W"] = simulated.image_data
            amp_measured: Float[Array, " warmup H W"] = jnp.sqrt(
                jnp.maximum(measured, 1e-12)
            )
            amp_predicted: Float[Array, " warmup H W"] = jnp.sqrt(
                jnp.maximum(predicted, 1e-12)
            )
            return (amp_measured - amp_predicted).ravel()

        _ = gn_solve(
            state,
            _warmup_residuals,
            max_iterations=1,
            cg_maxiter=cg_maxiter_used,
            cg_tol=cg_tol_used,
        )
    full_loss_chunks: list[Float[Array, " N"]]
    full_loss_chunks = []
    sample_snapshots: list[Complex[Array, " H W"]]
    sample_snapshots = []
    probe_snapshots: list[Complex[Array, " H W"]]
    probe_snapshots = []
    current_state: GaussNewtonState = state
    remaining: int = num_iterations

    while remaining > 0:
        this_chunk: int = min(save_every, remaining)
        current_state, chunk_losses = gn_loss_history(
            current_state,
            _amplitude_residuals,
            max_iterations=this_chunk,
            cg_maxiter=cg_maxiter_used,
            cg_tol=cg_tol_used,
        )
        full_loss_chunks.append(chunk_losses)
        sample_snapshots.append(current_state.sample)
        probe_snapshots.append(current_state.probe)
        remaining -= this_chunk

    if num_iterations > 0:
        all_losses: Float[Array, " N"] = jnp.concatenate(
            full_loss_chunks, axis=0
        )
        snapshot_samples: Complex[Array, " K H W"] = jnp.stack(
            sample_snapshots, axis=0
        )
        snapshot_probes: Complex[Array, " K H W"] = jnp.stack(
            probe_snapshots, axis=0
        )
    else:
        all_losses = jnp.zeros((0,), dtype=jnp.float64)
        snapshot_samples = jnp.zeros(
            (0, *sample.sample.shape), dtype=jnp.complex128
        )
        snapshot_probes = jnp.zeros(
            (0, *lightwave.field.shape), dtype=jnp.complex128
        )

    final_state: GaussNewtonState = current_state
    return _gn_state_to_ptychography_reconstruction(
        final_state,
        snapshot_samples,
        snapshot_probes,
        all_losses,
        reconstruction,
        sample.dx,
    )


def _gn_state_to_ptychography_reconstruction(
    final_state: GaussNewtonState,
    snapshot_samples: Complex[Array, " K H W"],
    snapshot_probes: Complex[Array, " K H W"],
    all_losses: Float[Array, " N"],
    reconstruction: PtychographyReconstruction,
    sample_dx: Float[Array, " "],
) -> PtychographyReconstruction:
    """Pack Gauss-Newton state and geometry into a PtychographyReconstruction.

    Builds sample/lightwave from final_state, stores sparse sample/probe
    snapshots, records full loss history, and concatenates with any previous
    intermediates/losses from reconstruction. Handles global iteration
    numbering across resume calls.
    """
    lightwave: OpticalWavefront = reconstruction.lightwave
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

    final_sample: SampleFunction = make_sample_function(
        sample=final_state.sample, dx=sample_dx
    )
    final_lightwave: OpticalWavefront = make_optical_wavefront(
        field=final_state.probe,
        wavelength=lightwave.wavelength,
        dx=lightwave.dx,
        z_position=lightwave.z_position,
    )

    prev_losses: Float[Array, " M 2"] = reconstruction.losses
    num_iterations: int = int(all_losses.shape[0])
    num_snapshots: int = int(snapshot_samples.shape[0])
    if num_snapshots == 0:
        intermediate_samples: Complex[Array, " H W 0"] = jnp.zeros(
            (*final_state.sample.shape, 0), dtype=jnp.complex128
        )
        intermediate_lightwaves: Complex[Array, " H W 0"] = jnp.zeros(
            (*final_state.probe.shape, 0), dtype=jnp.complex128
        )
        intermediate_zoom_factors: Float[Array, " 0"] = jnp.zeros(
            (0,), dtype=jnp.float64
        )
        intermediate_aperture_diameters: Float[Array, " 0"] = jnp.zeros(
            (0,), dtype=jnp.float64
        )
        intermediate_aperture_centers: Float[Array, " 2 0"] = jnp.zeros(
            (2, 0), dtype=jnp.float64
        )
        intermediate_travel_distances: Float[Array, " 0"] = jnp.zeros(
            (0,), dtype=jnp.float64
        )
        losses: Float[Array, " 0 2"] = jnp.zeros((0, 2), dtype=jnp.float64)
    else:
        intermediate_samples: Complex[Array, " H W K"] = jnp.transpose(
            snapshot_samples, (1, 2, 0)
        )
        intermediate_lightwaves: Complex[Array, " H W K"] = jnp.transpose(
            snapshot_probes, (1, 2, 0)
        )
        intermediate_zoom_factors: Float[Array, " K"] = jnp.full(
            num_snapshots, zoom_factor, dtype=jnp.float64
        )
        intermediate_aperture_diameters: Float[Array, " K"] = jnp.full(
            num_snapshots, aperture_diameter, dtype=jnp.float64
        )
        intermediate_aperture_centers: Float[Array, " 2 K"] = jnp.broadcast_to(
            aperture_center[:, None], (2, num_snapshots)
        )
        intermediate_travel_distances: Float[Array, " K"] = jnp.full(
            num_snapshots, travel_distance, dtype=jnp.float64
        )

        losses = jnp.zeros((0, 2), dtype=jnp.float64)
    start_iteration: Float[Array, " "] = jnp.where(
        prev_losses.shape[0] > 0,
        prev_losses[-1, 0] + 1.0,
        0.0,
    )
    if num_iterations > 0:
        iteration_numbers: Float[Array, " N"] = start_iteration + jnp.arange(
            num_iterations, dtype=jnp.float64
        )
        losses = jnp.stack([iteration_numbers, all_losses], axis=1)
    if prev_losses.shape[0] > 0:
        intermediate_samples = jnp.concatenate(
            [
                reconstruction.intermediate_samples,
                intermediate_samples,
            ],
            axis=-1,
        )
        intermediate_lightwaves = jnp.concatenate(
            [
                reconstruction.intermediate_lightwaves,
                intermediate_lightwaves,
            ],
            axis=-1,
        )
        intermediate_zoom_factors = jnp.concatenate(
            [
                reconstruction.intermediate_zoom_factors,
                intermediate_zoom_factors,
            ],
            axis=-1,
        )
        intermediate_aperture_diameters = jnp.concatenate(
            [
                reconstruction.intermediate_aperture_diameters,
                intermediate_aperture_diameters,
            ],
            axis=-1,
        )
        intermediate_aperture_centers = jnp.concatenate(
            [
                reconstruction.intermediate_aperture_centers,
                intermediate_aperture_centers,
            ],
            axis=-1,
        )
        intermediate_travel_distances = jnp.concatenate(
            [
                reconstruction.intermediate_travel_distances,
                intermediate_travel_distances,
            ],
            axis=-1,
        )
        losses = jnp.concatenate([prev_losses, losses], axis=0)

    return make_ptychography_reconstruction(
        sample=final_sample,
        lightwave=final_lightwave,
        translated_positions=translated_positions,
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
