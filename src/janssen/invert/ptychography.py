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
    MicroscopeData,
    OpticalWavefront,
    PtychographyParams,
    PtychographyReconstruction,
    SampleFunction,
    make_optical_wavefront,
    make_ptychography_reconstruction,
    make_sample_function,
)

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
    # Extract state from reconstruction
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

    # Extract previous history for appending
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

    # Extract parameters from params
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

    # Get starting iteration number from previous history
    start_iteration: Int[Array, " "] = jnp.array(
        prev_losses.shape[0], dtype=jnp.int64
    )
    num_iterations_int: int = int(num_iterations)

    sample_dx: Float[Array, " "] = guess_sample.dx
    guess_sample_field: Complex[Array, " H W"] = guess_sample.sample

    # Get loss type string
    loss_type_str: str = LOSS_TYPES[int(loss_type)]

    # Define the forward model function for the loss calculation
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

    # Create loss function
    loss_func: Callable[..., Float[Array, " "]] = create_loss_function(
        _forward_fn, experimental_data.image_data, loss_type_str
    )

    # Define function to compute loss and gradients
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

    # Create optax optimizer using integer index
    optimizer: optax.GradientTransformation = OPTIMIZERS[int(optimizer_type)](
        float(learning_rate)
    )

    # Initialize optimizer state for sample
    sample_opt_state: optax.OptState = optimizer.init(guess_sample_field)

    # Initial parameters
    sample_field: Complex[Array, " H W"] = guess_sample_field
    lightwave_field: Complex[Array, " H W"] = guess_lightwave.field

    # Define the scan body function
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

        # Compute loss and gradients (only for sample)
        loss_val, grad = jax.value_and_grad(_compute_loss, argnums=0)(
            sf, lf, zf, ad, td, ac
        )

        # Update sample
        updates, new_opt_state = optimizer.update(grad, opt_state, sf)
        new_sf = optax.apply_updates(sf, updates)

        # New carry
        new_carry = (new_sf, lf, zf, ad, td, ac, new_opt_state)

        # Output to stack (sample, lightwave, params, loss)
        output = (new_sf, lf, zf, ad, td, ac, loss_val)

        return new_carry, output

    # Initial carry
    init_carry = (
        sample_field,
        lightwave_field,
        zoom_factor,
        aperture_diameter,
        travel_distance,
        aperture_center,
        sample_opt_state,
    )

    # Run scan over iterations
    iterations: Int[Array, " N"] = jnp.arange(
        num_iterations_int, dtype=jnp.int64
    )

    final_carry, outputs = lax.scan(_scan_body, init_carry, iterations)

    # Unpack outputs - each has shape (num_iterations, ...)
    (
        intermediate_samples_new,
        intermediate_lightwaves_new,
        intermediate_zoom_factors_new,
        intermediate_aperture_diameters_new,
        intermediate_travel_distances_new,
        intermediate_aperture_centers_new,
        losses_new,
    ) = outputs

    # Transpose samples and lightwaves from (N, H, W) to (H, W, N)
    intermediate_samples: Complex[Array, " H W S"] = jnp.transpose(
        intermediate_samples_new, (1, 2, 0)
    )
    intermediate_lightwaves: Complex[Array, " H W S"] = jnp.transpose(
        intermediate_lightwaves_new, (1, 2, 0)
    )

    # Transpose aperture centers from (N, 2) to (2, N)
    intermediate_aperture_centers: Float[Array, " 2 S"] = jnp.transpose(
        intermediate_aperture_centers_new, (1, 0)
    )

    # Create iteration numbers for loss array
    iteration_numbers: Float[Array, " N"] = start_iteration + jnp.arange(
        num_iterations_int, dtype=jnp.float64
    )
    losses: Float[Array, " N 2"] = jnp.stack(
        [iteration_numbers, losses_new], axis=1
    )

    # Get final values from carry
    (
        final_sample_field,
        final_lightwave_field,
        current_zoom_factor,
        current_aperture_diameter,
        current_travel_distance,
        current_aperture_center,
        _,
    ) = final_carry

    # Create final objects
    final_sample: SampleFunction = make_sample_function(
        sample=final_sample_field, dx=sample_dx
    )

    final_lightwave: OpticalWavefront = make_optical_wavefront(
        field=final_lightwave_field,
        wavelength=guess_lightwave.wavelength,
        dx=guess_lightwave.dx,
        z_position=guess_lightwave.z_position,
    )

    # Concatenate previous history with new results
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

    # Return PtychographyReconstruction PyTree with combined history
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


@jaxtyped(typechecker=beartype)
def simple_microscope_epie(  # noqa: PLR0915
    experimental_data: MicroscopeData,
    reconstruction: PtychographyReconstruction,
    params: PtychographyParams,
) -> PtychographyReconstruction:
    """Ptychographic reconstruction using extended PIE algorithm.

    Reconstructs a sample from experimental diffraction patterns using
    the extended Ptychographic Iterative Engine (ePIE) algorithm. Takes a
    PtychographyReconstruction (from init_simple_microscope or a previous call)
    and runs additional iterations, appending results to intermediate arrays.

    ePIE is a sequential algorithm that updates both the sample (object) and
    probe at each scan position. Unlike gradient-based methods, it uses
    multiplicative updates based on the Fourier constraint.

    Parameters
    ----------
    experimental_data : MicroscopeData
        The experimental diffraction patterns collected at different
        positions. Positions should be in meters.
    reconstruction : PtychographyReconstruction
        Previous reconstruction state from init_simple_microscope or
        a previous call. Contains sample, lightwave, positions, optical
        parameters, and intermediate history.
    params : PtychographyParams
        Optimization parameters. For ePIE, learning_rate controls the
        update step size (alpha parameter). num_iterations is the number
        of complete sweeps over all positions.

    Returns
    -------
    reconstruction : PtychographyReconstruction
        Updated reconstruction with:
        - sample : Final optimized sample
        - lightwave : Final optimized probe/lightwave
        - translated_positions : Unchanged from input
        - Optical parameters : Unchanged from input
        - intermediate_* : Previous history + new iterations appended
        - losses : Previous history + new iterations appended

    See Also
    --------
    init_simple_microscope : Create initial reconstruction state.
    simple_microscope_ptychography : Gradient-based reconstruction.
    """
    # Extract state from reconstruction
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

    # Extract previous history
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

    # Extract parameters
    camera_pixel_size: Float[Array, " "] = params.camera_pixel_size
    num_iterations: Int[Array, " "] = params.num_iterations
    alpha: Float[Array, " "] = params.learning_rate  # ePIE step size
    num_iterations_int: int = int(num_iterations)

    # Get starting iteration number
    start_iteration: Int[Array, " "] = jnp.array(
        prev_losses.shape[0], dtype=jnp.int64
    )

    sample_dx: Float[Array, " "] = guess_sample.dx
    sample_field: Complex[Array, " H W"] = guess_sample.sample
    probe_field: Complex[Array, " h w"] = guess_lightwave.field

    probe_size_y: int = probe_field.shape[0]
    probe_size_x: int = probe_field.shape[1]

    # Convert positions to pixels
    pixel_positions: Float[Array, " N 2"] = translated_positions / sample_dx

    # Regularization for numerical stability
    eps: float = 1e-8

    def _epie_single_position(
        carry: Tuple[Complex[Array, " H W"], Complex[Array, " h w"]],
        inputs: Tuple[Float[Array, " h w"], Float[Array, " 2"]],
    ) -> Tuple[
        Tuple[Complex[Array, " H W"], Complex[Array, " h w"]],
        None,
    ]:
        """Update sample and probe for a single scan position."""
        obj, probe = carry
        measurement, pos = inputs

        # Get position indices - must match forward model convention
        # pos[0] is x (column), pos[1] is y (row)
        # Forward model uses: floor(pos - 0.5*size) for start index
        start_x: Int[Array, " "] = jnp.floor(
            pos[0] - 0.5 * probe_size_x
        ).astype(jnp.int32)
        start_y: Int[Array, " "] = jnp.floor(
            pos[1] - 0.5 * probe_size_y
        ).astype(jnp.int32)

        # Extract object patch at this position
        obj_patch: Complex[Array, " h w"] = lax.dynamic_slice(
            obj, (start_y, start_x), (probe_size_y, probe_size_x)
        )

        # Exit wave: object * probe
        exit_wave: Complex[Array, " h w"] = obj_patch * probe

        # Propagate to detector (simple Fourier transform for far-field)
        exit_wave_ft: Complex[Array, " h w"] = jnp.fft.fftshift(
            jnp.fft.fft2(exit_wave)
        )

        # Apply Fourier constraint: replace amplitude with measured amplitude
        measured_amplitude: Float[Array, " h w"] = jnp.sqrt(
            jnp.maximum(measurement, 0.0)
        )
        current_amplitude: Float[Array, " h w"] = jnp.abs(exit_wave_ft) + eps
        exit_wave_ft_updated: Complex[Array, " h w"] = (
            exit_wave_ft * measured_amplitude / current_amplitude
        )

        # Propagate back
        exit_wave_updated: Complex[Array, " h w"] = jnp.fft.ifft2(
            jnp.fft.ifftshift(exit_wave_ft_updated)
        )

        # Compute update difference
        diff: Complex[Array, " h w"] = exit_wave_updated - exit_wave

        # ePIE object update
        probe_conj: Complex[Array, " h w"] = jnp.conj(probe)
        probe_intensity: Float[Array, " h w"] = jnp.abs(probe) ** 2
        probe_max_intensity: Float[Array, " "] = jnp.max(probe_intensity)
        obj_update: Complex[Array, " h w"] = (
            alpha * probe_conj * diff / (probe_max_intensity + eps)
        )
        obj_patch_new: Complex[Array, " h w"] = obj_patch + obj_update

        # Update object in full array
        obj_new: Complex[Array, " H W"] = lax.dynamic_update_slice(
            obj, obj_patch_new, (start_y, start_x)
        )

        # ePIE probe update
        obj_conj: Complex[Array, " h w"] = jnp.conj(obj_patch)
        obj_intensity: Float[Array, " h w"] = jnp.abs(obj_patch) ** 2
        obj_max_intensity: Float[Array, " "] = jnp.max(obj_intensity)
        probe_update: Complex[Array, " h w"] = (
            alpha * obj_conj * diff / (obj_max_intensity + eps)
        )
        probe_new: Complex[Array, " h w"] = probe + probe_update

        return (obj_new, probe_new), None

    def _epie_one_iteration(
        carry: Tuple[Complex[Array, " H W"], Complex[Array, " h w"]],
        _iter_idx: Int[Array, " "],
    ) -> Tuple[
        Tuple[Complex[Array, " H W"], Complex[Array, " h w"]],
        Tuple[
            Complex[Array, " H W"],
            Complex[Array, " h w"],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " 2"],
            Float[Array, " "],
        ],
    ]:
        """One complete sweep over all positions."""
        obj, probe = carry

        # Sweep over all positions sequentially
        (obj_updated, probe_updated), _ = lax.scan(
            _epie_single_position,
            (obj, probe),
            (experimental_data.image_data, pixel_positions),
        )

        # Compute loss (MSE) after this iteration
        # Run forward model with current estimate
        sample_current: SampleFunction = make_sample_function(
            sample=obj_updated, dx=sample_dx
        )
        lightwave_current: OpticalWavefront = make_optical_wavefront(
            field=probe_updated,
            wavelength=guess_lightwave.wavelength,
            dx=guess_lightwave.dx,
            z_position=guess_lightwave.z_position,
        )
        simulated: MicroscopeData = simple_microscope(
            sample=sample_current,
            positions=translated_positions,
            lightwave=lightwave_current,
            zoom_factor=zoom_factor,
            aperture_diameter=aperture_diameter,
            travel_distance=travel_distance,
            camera_pixel_size=camera_pixel_size,
            aperture_center=aperture_center,
        )
        loss_val: Float[Array, " "] = jnp.mean(
            (simulated.image_data - experimental_data.image_data) ** 2
        )

        new_carry = (obj_updated, probe_updated)
        output = (
            obj_updated,
            probe_updated,
            zoom_factor,
            aperture_diameter,
            travel_distance,
            aperture_center,
            loss_val,
        )

        return new_carry, output

    # Initial carry
    init_carry = (sample_field, probe_field)

    # Run scan over iterations
    iterations: Int[Array, " N"] = jnp.arange(
        num_iterations_int, dtype=jnp.int64
    )

    final_carry, outputs = lax.scan(
        _epie_one_iteration, init_carry, iterations
    )

    # Unpack outputs
    (
        intermediate_samples_new,
        intermediate_lightwaves_new,
        intermediate_zoom_factors_new,
        intermediate_aperture_diameters_new,
        intermediate_travel_distances_new,
        intermediate_aperture_centers_new,
        losses_new,
    ) = outputs

    # Transpose from (N, H, W) to (H, W, N)
    intermediate_samples: Complex[Array, " H W S"] = jnp.transpose(
        intermediate_samples_new, (1, 2, 0)
    )
    intermediate_lightwaves: Complex[Array, " H W S"] = jnp.transpose(
        intermediate_lightwaves_new, (1, 2, 0)
    )
    intermediate_aperture_centers: Float[Array, " 2 S"] = jnp.transpose(
        intermediate_aperture_centers_new, (1, 0)
    )

    # Create iteration numbers
    iteration_numbers: Float[Array, " N"] = start_iteration + jnp.arange(
        num_iterations_int, dtype=jnp.float64
    )
    losses: Float[Array, " N 2"] = jnp.stack(
        [iteration_numbers, losses_new], axis=1
    )

    # Get final values
    final_sample_field, final_probe_field = final_carry

    # Create final objects
    final_sample: SampleFunction = make_sample_function(
        sample=final_sample_field, dx=sample_dx
    )
    final_lightwave: OpticalWavefront = make_optical_wavefront(
        field=final_probe_field,
        wavelength=guess_lightwave.wavelength,
        dx=guess_lightwave.dx,
        z_position=guess_lightwave.z_position,
    )

    # Concatenate histories
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

    # Return updated reconstruction
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
