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
from beartype.typing import Callable, Dict, Optional, Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from janssen.scopes import simple_microscope
from janssen.utils import (
    MicroscopeData,
    OpticalWavefront,
    PtychographyParams,
    PtychographyReconstruction,
    SampleFunction,
    ScalarFloat,
    ScalarInteger,
    make_optical_wavefront,
    make_ptychography_reconstruction,
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
) -> PtychographyReconstruction:
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
        Default is ±10% of the initial value.
    aperture_diameter_bounds :
        Tuple[ScalarFloat, ScalarFloat], optional
        Lower and upper bounds for aperture diameter optimization.
        Default is ±10% of the initial value.
    travel_distance_bounds : Tuple[ScalarFloat, ScalarFloat], optional
        Lower and upper bounds for travel distance optimization.
        Default is ±10% of the initial value.
    aperture_center_bounds :
        Tuple[Float[Array, " 2"], Float[Array, " 2"]], optional
        Lower and upper bounds for aperture center optimization.
        Default is ±10% of the initial value (element-wise).
    padding : ScalarInteger, optional
        Additional padding in pixels around the scanned region. If None,
        defaults to FOV/4 + probe/2. Only modify if you understand the
        position normalization logic.

    Returns
    -------
    full_and_intermediate : PtychographyReconstruction
        PyTree containing all reconstruction results:
        - sample : SampleFunction
            Final optimized sample covering the scanned FOV.
        - lightwave : OpticalWavefront
            Final optimized probe/lightwave.
        - zoom_factor : Float[Array, " "]
            Final optimized zoom factor.
        - aperture_diameter : Float[Array, " "]
            Final optimized aperture diameter.
        - aperture_center : Float[Array, " 2"] or None
            Final optimized aperture center.
        - travel_distance : Float[Array, " "]
            Final optimized travel distance.
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
    zoom_factor: Float[Array, " "] = params.zoom_factor
    aperture_diameter: Float[Array, " "] = params.aperture_diameter
    travel_distance: Float[Array, " "] = params.travel_distance
    aperture_center: Optional[Float[Array, " 2"]] = params.aperture_center
    camera_pixel_size: Float[Array, " "] = params.camera_pixel_size
    learning_rate: Float[Array, " "] = params.learning_rate
    num_iterations: int = int(params.num_iterations)

    # Set default bounds to ±10% of initial values if not provided
    if zoom_factor_bounds is None:
        zoom_factor_bounds = (0.9 * zoom_factor, 1.1 * zoom_factor)
    if aperture_diameter_bounds is None:
        aperture_diameter_bounds = (
            0.9 * aperture_diameter,
            1.1 * aperture_diameter,
        )
    if travel_distance_bounds is None:
        travel_distance_bounds = (
            0.9 * travel_distance,
            1.1 * travel_distance,
        )
    if aperture_center_bounds is None and aperture_center is not None:
        aperture_center_bounds = (
            0.9 * aperture_center,
            1.1 * aperture_center,
        )

    # Get probe size from lightwave
    probe_size_y: int
    probe_size_x: int
    probe_size_y, probe_size_x = guess_lightwave.field.shape

    # Convert positions from meters to pixels
    pixel_positions: Float[Array, " N 2"] = (
        experimental_data.positions / guess_lightwave.dx
    )

    # Compute the scan FOV (range of positions) before normalization
    min_pos_x: Float[Array, " "] = jnp.min(pixel_positions[:, 0])
    max_pos_x: Float[Array, " "] = jnp.max(pixel_positions[:, 0])
    min_pos_y: Float[Array, " "] = jnp.min(pixel_positions[:, 1])
    max_pos_y: Float[Array, " "] = jnp.max(pixel_positions[:, 1])

    # Scan FOV is the range of positions (before adding probe size)
    scan_fov_x: Float[Array, " "] = max_pos_x - min_pos_x
    scan_fov_y: Float[Array, " "] = max_pos_y - min_pos_y

    # Half probe sizes (positions point to probe center, extraction needs space)
    half_probe_x: int = probe_size_x // 2
    half_probe_y: int = probe_size_y // 2

    # Default padding is FOV/4 + probe/2
    if padding is None:
        scan_fov: int = int(jnp.maximum(scan_fov_x, scan_fov_y))
        half_probe: int = max(half_probe_x, half_probe_y)
        padding = scan_fov // 4 + half_probe

    # Normalize positions: subtract minimum so they start from padding
    normalized_positions_x: Float[Array, " N"] = (
        pixel_positions[:, 0] - min_pos_x + padding
    )
    normalized_positions_y: Float[Array, " N"] = (
        pixel_positions[:, 1] - min_pos_y + padding
    )

    # FOV size: scan range + probe size + padding on both sides
    fov_size_x: int = int(jnp.ceil(scan_fov_x)) + probe_size_x + 2 * padding
    fov_size_y: int = int(jnp.ceil(scan_fov_y)) + probe_size_y + 2 * padding

    # Convert normalized pixel positions back to meters for the forward model
    sample_dx: Float[Array, " "] = guess_lightwave.dx
    translated_positions: Float[Array, " N 2"] = (
        jnp.stack([normalized_positions_x, normalized_positions_y], axis=1)
        * sample_dx
    )

    # Create initial guess sample covering the full FOV
    guess_sample_field: Complex[Array, " H W"] = jnp.ones(
        (fov_size_y, fov_size_x), dtype=jnp.complex128
    )

    print(f"Scan FOV: {scan_fov_y:.1f} x {scan_fov_x:.1f} pixels")
    print(f"Padding: {padding} pixels")
    print(f"Reconstruction FOV: {fov_size_y} x {fov_size_x} pixels")
    print(f"Probe size: {probe_size_y} x {probe_size_x} pixels")

    # Define bound enforcement functions
    def _enforce_bounds(
        param: Float[Array, " "],
        param_bounds: Optional[Tuple[ScalarFloat, ScalarFloat]] = None,
    ) -> Float[Array, " "]:
        if param_bounds is None:
            return param
        lower: ScalarFloat
        upper: ScalarFloat
        lower, upper = param_bounds
        return jnp.clip(param, lower, upper)

    def _enforce_bounds_2d(
        param: Float[Array, " 2"],
        param_bounds: Optional[
            Tuple[Float[Array, " 2"], Float[Array, " 2"]]
        ] = None,
    ) -> Float[Array, " 2"]:
        if param_bounds is None:
            return param
        lower: Float[Array, " 2"]
        upper: Float[Array, " 2"]
        lower, upper = param_bounds
        return jnp.clip(param, lower, upper)

    # Define the forward model function for the loss calculation
    def _forward_fn(
        sample_field: Complex[Array, " H W"],
        lightwave_field: Complex[Array, " H W"],
        zoom_factor: Float[Array, " "],
        aperture_diameter: Float[Array, " "],
        travel_distance: Float[Array, " "],
        aperture_center: Float[Array, " 2"],
    ) -> Float[Array, " N H W"]:
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
    loss_func: Callable[..., Float[Array, " "]] = create_loss_function(
        _forward_fn, experimental_data.image_data, loss_type
    )

    # Define function to compute loss and gradients
    @jax.jit
    def _loss_and_grad(
        sample_field: Complex[Array, " H W"],
        lightwave_field: Complex[Array, " H W"],
        zoom_factor: Float[Array, " "],
        aperture_diameter: Float[Array, " "],
        travel_distance: Float[Array, " "],
        aperture_center: Float[Array, " 2"],
    ) -> Tuple[Float[Array, " "], Dict[str, Array]]:
        def _loss_wrapped(
            sample_field: Complex[Array, " H W"],
            lightwave_field: Complex[Array, " H W"],
            zoom_factor: Float[Array, " "],
            aperture_diameter: Float[Array, " "],
            travel_distance: Float[Array, " "],
            aperture_center: Float[Array, " 2"],
        ) -> Float[Array, " "]:
            # Enforce bounds before calculating loss
            bounded_zoom_factor: Float[Array, " "] = _enforce_bounds(
                zoom_factor, zoom_factor_bounds
            )
            bounded_aperture_diameter: Float[Array, " "] = _enforce_bounds(
                aperture_diameter, aperture_diameter_bounds
            )
            bounded_travel_distance: Float[Array, " "] = _enforce_bounds(
                travel_distance, travel_distance_bounds
            )
            bounded_aperture_center: Float[Array, " 2"] = _enforce_bounds_2d(
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

        loss: Float[Array, " "]
        grads: Tuple[Array, ...]
        loss, grads = jax.value_and_grad(
            _loss_wrapped, argnums=(0, 1, 2, 3, 4, 5)
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
    optimizer: optax.GradientTransformation = OPTIMIZERS[optimizer_name](
        learning_rate
    )

    # Initialize optimizer states for each parameter
    sample_opt_state: optax.OptState = optimizer.init(guess_sample_field)
    lightwave_opt_state: optax.OptState = optimizer.init(guess_lightwave.field)
    zoom_factor_opt_state: optax.OptState = optimizer.init(zoom_factor)
    aperture_diameter_opt_state: optax.OptState = optimizer.init(
        aperture_diameter
    )
    travel_distance_opt_state: optax.OptState = optimizer.init(travel_distance)
    aperture_center_opt_state: optax.OptState = optimizer.init(
        jnp.zeros(2) if aperture_center is None else aperture_center
    )

    # Initialize parameters
    sample_field: Complex[Array, " H W"] = guess_sample_field
    lightwave_field: Complex[Array, " H W"] = guess_lightwave.field
    current_zoom_factor: Float[Array, " "] = zoom_factor
    current_aperture_diameter: Float[Array, " "] = aperture_diameter
    current_travel_distance: Float[Array, " "] = travel_distance
    current_aperture_center: Float[Array, " 2"] = (
        jnp.zeros(2) if aperture_center is None else aperture_center
    )

    # Set up intermediate result storage
    num_saves: int = int(jnp.floor(num_iterations / save_every))

    intermediate_samples: Complex[Array, " H W S"] = jnp.zeros(
        (sample_field.shape[0], sample_field.shape[1], num_saves),
        dtype=sample_field.dtype,
    )

    intermediate_lightwaves: Complex[Array, " H W S"] = jnp.zeros(
        (lightwave_field.shape[0], lightwave_field.shape[1], num_saves),
        dtype=lightwave_field.dtype,
    )

    intermediate_zoom_factors: Float[Array, " S"] = jnp.zeros(
        num_saves, dtype=jnp.float64
    )
    intermediate_aperture_diameters: Float[Array, " S"] = jnp.zeros(
        num_saves, dtype=jnp.float64
    )
    intermediate_travel_distances: Float[Array, " S"] = jnp.zeros(
        num_saves, dtype=jnp.float64
    )
    intermediate_aperture_centers: Float[Array, " 2 S"] = jnp.zeros(
        (2, num_saves), dtype=jnp.float64
    )

    @jax.jit
    def _update_step(
        sample_field: Complex[Array, " H W"],
        lightwave_field: Complex[Array, " H W"],
        zoom_factor: Float[Array, " "],
        aperture_diameter: Float[Array, " "],
        travel_distance: Float[Array, " "],
        aperture_center: Float[Array, " 2"],
        sample_opt_state: optax.OptState,
        lightwave_opt_state: optax.OptState,
        zoom_factor_opt_state: optax.OptState,
        aperture_diameter_opt_state: optax.OptState,
        travel_distance_opt_state: optax.OptState,
        aperture_center_opt_state: optax.OptState,
    ) -> Tuple[
        Complex[Array, " H W"],
        Complex[Array, " H W"],
        Float[Array, " "],
        Float[Array, " "],
        Float[Array, " "],
        Float[Array, " 2"],
        optax.OptState,
        optax.OptState,
        optax.OptState,
        optax.OptState,
        optax.OptState,
        optax.OptState,
        Float[Array, " "],
    ]:
        loss: Float[Array, " "]
        grads: Dict[str, Array]
        loss, grads = _loss_and_grad(
            sample_field,
            lightwave_field,
            zoom_factor,
            aperture_diameter,
            travel_distance,
            aperture_center,
        )

        # Update sample
        sample_updates: Array
        sample_updates, sample_opt_state = optimizer.update(
            grads["sample"], sample_opt_state, sample_field
        )
        sample_field = optax.apply_updates(sample_field, sample_updates)

        # Update lightwave
        lightwave_updates: Array
        lightwave_updates, lightwave_opt_state = optimizer.update(
            grads["lightwave"], lightwave_opt_state, lightwave_field
        )
        lightwave_field = optax.apply_updates(
            lightwave_field, lightwave_updates
        )

        # Update zoom factor
        zoom_updates: Array
        zoom_updates, zoom_factor_opt_state = optimizer.update(
            grads["zoom_factor"], zoom_factor_opt_state, zoom_factor
        )
        zoom_factor = optax.apply_updates(zoom_factor, zoom_updates)
        zoom_factor = _enforce_bounds(zoom_factor, zoom_factor_bounds)

        # Update aperture diameter
        aperture_updates: Array
        aperture_updates, aperture_diameter_opt_state = optimizer.update(
            grads["aperture_diameter"],
            aperture_diameter_opt_state,
            aperture_diameter,
        )
        aperture_diameter = optax.apply_updates(
            aperture_diameter, aperture_updates
        )
        aperture_diameter = _enforce_bounds(
            aperture_diameter, aperture_diameter_bounds
        )

        # Update travel distance
        travel_updates: Array
        travel_updates, travel_distance_opt_state = optimizer.update(
            grads["travel_distance"],
            travel_distance_opt_state,
            travel_distance,
        )
        travel_distance = optax.apply_updates(travel_distance, travel_updates)
        travel_distance = _enforce_bounds(
            travel_distance, travel_distance_bounds
        )

        # Update aperture center
        center_updates: Array
        center_updates, aperture_center_opt_state = optimizer.update(
            grads["aperture_center"],
            aperture_center_opt_state,
            aperture_center,
        )
        aperture_center = optax.apply_updates(aperture_center, center_updates)
        aperture_center = _enforce_bounds_2d(
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
    loss: Float[Array, " "]
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
        ) = _update_step(
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
            save_idx: int = ii // save_every
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

    # Return PtychographyReconstruction PyTree
    full_and_intermediate: PtychographyReconstruction = (
        make_ptychography_reconstruction(
            sample=final_sample,
            lightwave=final_lightwave,
            zoom_factor=current_zoom_factor,
            aperture_diameter=current_aperture_diameter,
            aperture_center=current_aperture_center,
            travel_distance=current_travel_distance,
            intermediate_samples=intermediate_samples,
            intermediate_lightwaves=intermediate_lightwaves,
            intermediate_zoom_factors=intermediate_zoom_factors,
            intermediate_aperture_diameters=intermediate_aperture_diameters,
            intermediate_aperture_centers=intermediate_aperture_centers,
            intermediate_travel_distances=intermediate_travel_distances,
        )
    )
    return full_and_intermediate
