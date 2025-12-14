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
    Performs ptychography reconstruction using a simple microscope model

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
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from janssen.scopes import simple_microscope
from janssen.utils import (
    MicroscopeData,
    OpticalWavefront,
    PtychographyParams,
    SampleFunction,
    ScalarFloat,
    ScalarInteger,
    make_optical_wavefront,
    make_sample_function,
)

from .loss_functions import create_loss_function

jax.config.update("jax_enable_x64", True)

OPTIMIZERS = {
    "adam": optax.adam,
    "adagrad": optax.adagrad,
    "rmsprop": optax.rmsprop,
    "sgd": optax.sgd,
}


@jaxtyped(typechecker=beartype)
def simple_microscope_ptychography(
    experimental_data: MicroscopeData,
    guess_lightwave: OpticalWavefront,
    params: PtychographyParams,
    save_every: Optional[ScalarInteger] = 10,
    loss_type: Optional[str] = "mse",
    optimizer_name: Optional[str] = "adam",
    zoom_factor_bounds: Optional[Tuple[ScalarFloat, ScalarFloat]] = None,
    aperture_diameter_bounds: Optional[Tuple[ScalarFloat, ScalarFloat]] = None,
    travel_distance_bounds: Optional[Tuple[ScalarFloat, ScalarFloat]] = None,
    aperture_center_bounds: Optional[
        Tuple[Float[Array, " 2"], Float[Array, " 2"]]
    ] = None,
    padding: Optional[ScalarInteger] = None,
) -> Tuple[
    Tuple[
        SampleFunction,  # final_sample
        OpticalWavefront,  # final_lightwave
        ScalarFloat,  # final_zoom_factor
        ScalarFloat,  # final_aperture_diameter
        Optional[Float[Array, " 2"]],  # final_aperture_center
        ScalarFloat,  # final_travel_distance
    ],
    Tuple[
        Complex[Array, " H W S"],  # intermediate_samples
        Complex[Array, " H W S"],  # intermediate_lightwaves
        Float[Array, " S"],  # intermediate_zoom_factors
        Float[Array, " S"],  # intermediate_aperture_diameters
        Float[Array, " 2 S"],  # intermediate_aperture_centers
        Float[Array, " S"],  # intermediate_travel_distances
    ],
]:
    """Solve the optical ptychography inverse problem.

    Here experimental diffraction patterns are used to reconstruct a
    sample, lightwave, and optical system parameters. The reconstructed
    sample covers the full field of view spanned by the scan positions
    plus padding.

    Parameters
    ----------
    experimental_data : MicroscopeData
        The experimental diffraction patterns collected at different
        positions. Positions should be in meters.
    guess_lightwave : OpticalWavefront
        Initial guess for the lightwave (probe). The probe size determines
        the interaction region at each scan position.
    params : PtychographyParams
        Ptychography parameters including:
        - zoom_factor: Optical zoom factor for magnification
        - aperture_diameter: Diameter of the aperture in meters
        - travel_distance: Light propagation distance in meters
        - aperture_center: Center position of the aperture (x, y)
        - camera_pixel_size: Camera pixel size in meters
        - learning_rate: Learning rate for optimization
        - num_iterations: Number of optimization iterations
    save_every : ScalarInteger, optional
        Save intermediate results every n iterations. Default is 10.
    loss_type : str, optional
        Type of loss function to use. Default is "mse".
    optimizer_name : str, optional
        Name of the optimizer to use ("adam", "adagrad", "rmsprop", "sgd").
        Default is "adam".
    zoom_factor_bounds : Tuple[ScalarFloat, ScalarFloat], optional
        Lower and upper bounds for zoom factor optimization.
    aperture_diameter_bounds :
        Tuple[ScalarFloat, ScalarFloat], optional
        Lower and upper bounds for aperture diameter optimization.
    travel_distance_bounds : Tuple[ScalarFloat, ScalarFloat], optional
        Lower and upper bounds for travel distance optimization.
    aperture_center_bounds :
        Tuple[Float[Array, " 2"], Float[Array, " 2"]], optional
        Lower and upper bounds for aperture center optimization.
    padding : ScalarInteger, optional
        Additional padding in pixels around the scanned region. If None,
        defaults to FOV/4 + probe/2. Only modify if you understand the
        position normalization logic.

    Returns
    -------
    Tuple[Tuple[...], Tuple[...]]
        Tuple containing:
        - Final results tuple:
            - final_sample : SampleFunction
                Optimized sample covering the scanned FOV.
            - final_lightwave : OpticalWavefront
                Optimized lightwave (probe).
            - final_zoom_factor : ScalarFloat
                Optimized zoom factor.
            - final_aperture_diameter : ScalarFloat
                Optimized aperture diameter.
            - final_aperture_center : Float[Array, " 2"] or None
                Optimized aperture center.
            - final_travel_distance : ScalarFloat
                Optimized travel distance.
        - Intermediate results tuple:
            - intermediate_samples : Complex[Array, " H W S"]
                Intermediate samples during optimization.
            - intermediate_lightwaves : Complex[Array, " H W S"]
                Intermediate lightwaves during optimization.
            - intermediate_zoom_factors : Float[Array, " S"]
                Intermediate zoom factors during optimization.
            - intermediate_aperture_diameters : Float[Array, " S"]
                Intermediate aperture diameters during optimization.
            - intermediate_aperture_centers : Float[Array, " 2 S"]
                Intermediate aperture centers during optimization.
            - intermediate_travel_distances : Float[Array, " S"]
                Intermediate travel distances during optimization.

    Notes
    -----
    Position normalization:
    - Input positions can start from any value (negative or high positive)
    - Positions are normalized by subtracting the minimum x and y values
    - Padding is then added so positions start from the pad value

    The reconstruction FOV is computed as:
    - Range of normalized scan positions (in pixels)
    - plus half the probe size on each side
    - plus the specified padding on each side

    Default padding is a quarter of the scan FOV size.
    """
    # Extract parameters from PtychographyParams
    zoom_factor = params.zoom_factor
    aperture_diameter = params.aperture_diameter
    travel_distance = params.travel_distance
    aperture_center = params.aperture_center
    camera_pixel_size = params.camera_pixel_size
    learning_rate = params.learning_rate
    num_iterations = params.num_iterations

    # Get probe size from lightwave
    probe_size_y, probe_size_x = guess_lightwave.field.shape

    # Convert positions from meters to pixels
    pixel_positions = experimental_data.positions / guess_lightwave.dx

    # Compute the scan FOV (range of positions) before normalization
    min_pos_x = jnp.min(pixel_positions[:, 0])
    max_pos_x = jnp.max(pixel_positions[:, 0])
    min_pos_y = jnp.min(pixel_positions[:, 1])
    max_pos_y = jnp.max(pixel_positions[:, 1])

    # Scan FOV is the range of positions (before adding probe size)
    scan_fov_x = max_pos_x - min_pos_x
    scan_fov_y = max_pos_y - min_pos_y

    # Half probe sizes (positions point to probe center, extraction needs space)
    half_probe_x = probe_size_x // 2
    half_probe_y = probe_size_y // 2

    # Default padding is FOV/4 + probe/2
    if padding is None:
        scan_fov = int(jnp.maximum(scan_fov_x, scan_fov_y))
        half_probe = max(half_probe_x, half_probe_y)
        padding = scan_fov // 4 + half_probe

    # Normalize positions: subtract minimum so they start from padding
    normalized_positions_x = pixel_positions[:, 0] - min_pos_x + padding
    normalized_positions_y = pixel_positions[:, 1] - min_pos_y + padding

    # FOV size: scan range + probe size + padding on both sides
    fov_size_x = int(jnp.ceil(scan_fov_x)) + probe_size_x + 2 * padding
    fov_size_y = int(jnp.ceil(scan_fov_y)) + probe_size_y + 2 * padding

    # Convert normalized pixel positions back to meters for the forward model
    sample_dx = guess_lightwave.dx
    translated_positions = jnp.stack(
        [normalized_positions_x, normalized_positions_y], axis=1
    ) * sample_dx

    # Create initial guess sample covering the full FOV
    guess_sample_field = jnp.ones(
        (int(fov_size_y), int(fov_size_x)), dtype=jnp.complex128
    )

    print(f"Scan FOV: {scan_fov_y:.1f} x {scan_fov_x:.1f} pixels")
    print(f"Padding: {padding} pixels")
    print(f"Reconstruction FOV: {int(fov_size_y)} x {int(fov_size_x)} pixels")
    print(f"Probe size: {probe_size_y} x {probe_size_x} pixels")

    # Define bound enforcement functions
    def enforce_bounds(
        param: Float[Array, " S"],
        param_bounds: Optional[Tuple[ScalarFloat, ScalarFloat]] = None,
    ) -> Float[Array, " S"]:
        if param_bounds is None:
            return param
        lower, upper = param_bounds
        return jnp.clip(param, lower, upper)

    def enforce_bounds_2d(
        param: Float[Array, " 2 S"],
        param_bounds: Optional[Tuple[ScalarFloat, ScalarFloat]] = None,
    ) -> Float[Array, " 2 S"]:
        if param_bounds is None:
            return param
        lower, upper = param_bounds
        return jnp.clip(param, lower, upper)

    # Define the forward model function for the loss calculation
    def forward_fn(
        sample_field: Complex[Array, " H W S"],
        lightwave_field: Complex[Array, " H W S"],
        zoom_factor: Float[Array, " S"],
        aperture_diameter: Float[Array, " S"],
        travel_distance: Float[Array, " S"],
        aperture_center: Float[Array, " 2 S"],
    ) -> MicroscopeData:
        # Reconstruct PyTree objects from arrays
        sample: SampleFunction = make_sample_function(
            sample=sample_field, dx=sample_dx
        )

        lightwave: OpticalWavefront = make_optical_wavefront(
            field=lightwave_field,
            wavelength=guess_lightwave.wavelength,
            dx=guess_lightwave.dx,
            z_position=guess_lightwave.z_position,
        )

        # Generate the microscope data using the forward model
        # Use translated positions so they index correctly into the FOV
        simulated_data: MicroscopeData = simple_microscope(
            sample=sample,
            positions=translated_positions,
            lightwave=lightwave,
            zoom_factor=zoom_factor,
            aperture_diameter=aperture_diameter,
            travel_distance=travel_distance,
            camera_pixel_size=camera_pixel_size,
            aperture_center=aperture_center,
        )

        return simulated_data.image_data

    # Create loss function using the tools module
    loss_func = create_loss_function(
        forward_fn, experimental_data.image_data, loss_type
    )

    # Define function to compute loss and gradients
    @jax.jit
    def loss_and_grad(
        sample_field,
        lightwave_field,
        zoom_factor,
        aperture_diameter,
        travel_distance,
        aperture_center,
    ):
        def loss_wrapped(
            sample_field,
            lightwave_field,
            zoom_factor,
            aperture_diameter,
            travel_distance,
            aperture_center,
        ):
            # Enforce bounds before calculating loss
            bounded_zoom_factor = enforce_bounds(
                zoom_factor, zoom_factor_bounds
            )
            bounded_aperture_diameter = enforce_bounds(
                aperture_diameter, aperture_diameter_bounds
            )
            bounded_travel_distance = enforce_bounds(
                travel_distance, travel_distance_bounds
            )
            bounded_aperture_center = enforce_bounds_2d(
                aperture_center, aperture_center_bounds
            )

            return loss_func(
                sample_field,
                lightwave_field,
                bounded_zoom_factor,
                bounded_aperture_diameter,
                bounded_travel_distance,
                bounded_aperture_center,
            )

        loss, grads = jax.value_and_grad(
            loss_wrapped, argnums=(0, 1, 2, 3, 4, 5)
        )(
            sample_field,
            lightwave_field,
            zoom_factor,
            aperture_diameter,
            travel_distance,
            aperture_center,
        )

        return loss, {
            "sample": grads[0],
            "lightwave": grads[1],
            "zoom_factor": grads[2],
            "aperture_diameter": grads[3],
            "travel_distance": grads[4],
            "aperture_center": grads[5],
        }

    # Create optax optimizer
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Available: {list(OPTIMIZERS.keys())}"
        )
    optimizer = OPTIMIZERS[optimizer_name](learning_rate)

    # Initialize optimizer states for each parameter
    sample_opt_state = optimizer.init(guess_sample_field)
    lightwave_opt_state = optimizer.init(guess_lightwave.field)
    zoom_factor_opt_state = optimizer.init(zoom_factor)
    aperture_diameter_opt_state = optimizer.init(aperture_diameter)
    travel_distance_opt_state = optimizer.init(travel_distance)
    aperture_center_opt_state = optimizer.init(
        jnp.zeros(2) if aperture_center is None else aperture_center
    )

    # Initialize parameters
    sample_field = guess_sample_field
    lightwave_field = guess_lightwave.field
    current_zoom_factor = zoom_factor
    current_aperture_diameter = aperture_diameter
    current_travel_distance = travel_distance
    current_aperture_center = (
        jnp.zeros(2) if aperture_center is None else aperture_center
    )

    # Set up intermediate result storage
    num_saves = jnp.floor(num_iterations / save_every).astype(int)

    intermediate_samples = jnp.zeros(
        (sample_field.shape[0], sample_field.shape[1], num_saves),
        dtype=sample_field.dtype,
    )

    intermediate_lightwaves = jnp.zeros(
        (lightwave_field.shape[0], lightwave_field.shape[1], num_saves),
        dtype=lightwave_field.dtype,
    )

    intermediate_zoom_factors = jnp.zeros(num_saves, dtype=jnp.float64)
    intermediate_aperture_diameters = jnp.zeros(num_saves, dtype=jnp.float64)
    intermediate_travel_distances = jnp.zeros(num_saves, dtype=jnp.float64)
    intermediate_aperture_centers = jnp.zeros(
        (2, num_saves), dtype=jnp.float64
    )

    @jax.jit
    def update_step(
        sample_field,
        lightwave_field,
        zoom_factor,
        aperture_diameter,
        travel_distance,
        aperture_center,
        sample_opt_state,
        lightwave_opt_state,
        zoom_factor_opt_state,
        aperture_diameter_opt_state,
        travel_distance_opt_state,
        aperture_center_opt_state,
    ):
        loss, grads = loss_and_grad(
            sample_field,
            lightwave_field,
            zoom_factor,
            aperture_diameter,
            travel_distance,
            aperture_center,
        )

        # Update sample
        sample_updates, sample_opt_state = optimizer.update(
            grads["sample"], sample_opt_state, sample_field
        )
        sample_field = optax.apply_updates(sample_field, sample_updates)

        # Update lightwave
        lightwave_updates, lightwave_opt_state = optimizer.update(
            grads["lightwave"], lightwave_opt_state, lightwave_field
        )
        lightwave_field = optax.apply_updates(lightwave_field, lightwave_updates)

        # Update zoom factor
        zoom_updates, zoom_factor_opt_state = optimizer.update(
            grads["zoom_factor"], zoom_factor_opt_state, zoom_factor
        )
        zoom_factor = optax.apply_updates(zoom_factor, zoom_updates)
        zoom_factor = enforce_bounds(zoom_factor, zoom_factor_bounds)

        # Update aperture diameter
        aperture_updates, aperture_diameter_opt_state = optimizer.update(
            grads["aperture_diameter"],
            aperture_diameter_opt_state,
            aperture_diameter,
        )
        aperture_diameter = optax.apply_updates(
            aperture_diameter, aperture_updates
        )
        aperture_diameter = enforce_bounds(
            aperture_diameter, aperture_diameter_bounds
        )

        # Update travel distance
        travel_updates, travel_distance_opt_state = optimizer.update(
            grads["travel_distance"],
            travel_distance_opt_state,
            travel_distance,
        )
        travel_distance = optax.apply_updates(travel_distance, travel_updates)
        travel_distance = enforce_bounds(
            travel_distance, travel_distance_bounds
        )

        # Update aperture center
        center_updates, aperture_center_opt_state = optimizer.update(
            grads["aperture_center"],
            aperture_center_opt_state,
            aperture_center,
        )
        aperture_center = optax.apply_updates(aperture_center, center_updates)
        aperture_center = enforce_bounds_2d(
            aperture_center, aperture_center_bounds
        )

        return (
            sample_field,
            lightwave_field,
            zoom_factor,
            aperture_diameter,
            travel_distance,
            aperture_center,
            sample_opt_state,
            lightwave_opt_state,
            zoom_factor_opt_state,
            aperture_diameter_opt_state,
            travel_distance_opt_state,
            aperture_center_opt_state,
            loss,
        )

    # Run optimization loop
    for ii in range(num_iterations):
        (
            sample_field,
            lightwave_field,
            current_zoom_factor,
            current_aperture_diameter,
            current_travel_distance,
            current_aperture_center,
            sample_opt_state,
            lightwave_opt_state,
            zoom_factor_opt_state,
            aperture_diameter_opt_state,
            travel_distance_opt_state,
            aperture_center_opt_state,
            loss,
        ) = update_step(
            sample_field,
            lightwave_field,
            current_zoom_factor,
            current_aperture_diameter,
            current_travel_distance,
            current_aperture_center,
            sample_opt_state,
            lightwave_opt_state,
            zoom_factor_opt_state,
            aperture_diameter_opt_state,
            travel_distance_opt_state,
            aperture_center_opt_state,
        )

        # Save intermediate results
        if ii % save_every == 0:
            print(f"Iteration {ii}, Loss: {loss}")
            save_idx = ii // save_every
            if save_idx < num_saves:
                intermediate_samples = intermediate_samples.at[
                    :, :, save_idx
                ].set(sample_field)
                intermediate_lightwaves = intermediate_lightwaves.at[
                    :, :, save_idx
                ].set(lightwave_field)
                intermediate_zoom_factors = intermediate_zoom_factors.at[
                    save_idx
                ].set(current_zoom_factor)
                intermediate_aperture_diameters = (
                    intermediate_aperture_diameters.at[save_idx].set(
                        current_aperture_diameter
                    )
                )
                intermediate_travel_distances = (
                    intermediate_travel_distances.at[save_idx].set(
                        current_travel_distance
                    )
                )
                intermediate_aperture_centers = (
                    intermediate_aperture_centers.at[:, save_idx].set(
                        current_aperture_center
                    )
                )

    # Create final objects
    final_sample: SampleFunction = make_sample_function(
        sample=sample_field, dx=sample_dx
    )

    final_lightwave: OpticalWavefront = make_optical_wavefront(
        field=lightwave_field,
        wavelength=guess_lightwave.wavelength,
        dx=guess_lightwave.dx,
        z_position=guess_lightwave.z_position,
    )

    # Create final values tuple
    final_values = (
        final_sample,
        final_lightwave,
        current_zoom_factor,
        current_aperture_diameter,
        current_aperture_center,
        current_travel_distance,
    )

    # Create intermediate values tuple
    intermediate_values = (
        intermediate_samples,
        intermediate_lightwaves,
        intermediate_zoom_factors,
        intermediate_aperture_diameters,
        intermediate_aperture_centers,
        intermediate_travel_distances,
    )

    # Return both tuples as a single tuple of tuples
    return (final_values, intermediate_values)
