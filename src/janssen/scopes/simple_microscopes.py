"""Codes for optical propagation through lenses and optical elements.

Extended Summary
----------------
Microscope forward models for simulating image formation in optical
microscopy. Includes functions for computing diffraction patterns and
modeling light-sample interactions.

Routine Listings
----------------
lens_propagation : function
    Propagates an optical wavefront through a lens
linear_interaction : function
    Propagates an optical wavefront through a sample using linear
    interaction.
diffractogram_noscale : function
    Calculates the diffractogram of a sample using a simple model
    without scaling the pixel size of the camera image.
simple_diffractogram : function
    Calculates the diffractogram of a sample using a simple model
simple_microscope : function
    Calculates 3D diffractograms at all pixel positions in parallel

Notes
-----
These functions provide complete forward models for optical microscopy
and are designed for use in inverse problems and ptychography
reconstruction.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Complex, Float, Int, Num, jaxtyped

from janssen.optics.apertures import circular_aperture
from janssen.optics.helper import field_intensity
from janssen.prop import fraunhofer_prop, fraunhofer_prop_scaled, optical_zoom
from janssen.types import (
    Diffractogram,
    MicroscopeData,
    OpticalWavefront,
    SampleFunction,
    ScalarFloat,
    make_diffractogram,
    make_microscope_data,
    make_optical_wavefront,
    make_sample_function,
)


@jaxtyped(typechecker=beartype)
def linear_interaction(
    sample: SampleFunction,
    light: OpticalWavefront,
) -> OpticalWavefront:
    """Propagate an optical wavefront through a sample.

    The sample is modeled as a complex function that modifies
    the incoming wavefront. Using linear interaction.

    Parameters
    ----------
    sample : SampleFunction
        The sample function representing the optical properties of the
        sample
    light : OpticalWavefront
        The incoming optical wavefront

    Returns
    -------
    OpticalWavefront
        The propagated optical wavefront after passing through the
        sample
    """
    new_field: Complex[Array, " H W"] = sample.sample * light.field
    interacted: OpticalWavefront = make_optical_wavefront(
        field=new_field,
        wavelength=light.wavelength,
        dx=light.dx,
        z_position=light.z_position,
    )
    return interacted


@jaxtyped(typechecker=beartype)
def diffractogram_noscale(
    sample_cut: SampleFunction,
    lightwave: OpticalWavefront,
    zoom_factor: ScalarFloat,
    aperture_diameter: ScalarFloat,
    travel_distance: ScalarFloat,
    aperture_center: Optional[Float[Array, " 2"]] = None,
) -> OpticalWavefront:
    """Calculate the diffractogram of a sample using a simple model.

    The lightwave interacts with the sample linearly, and is then
    zoomed optically. Following this it interacts with a circular
    aperture before propagating to the camera plane.
    The camera image is then scaled to the pixel size of the camera.
    The diffractogram is created from the camera image.

    Parameters
    ----------
    sample_cut : SampleFunction
        The sample function representing the optical properties of the
        sample
    lightwave : OpticalWavefront
        The incoming optical wavefront
    zoom_factor : ScalarFloat
        The zoom factor for the optical system
    aperture_diameter : ScalarFloat
        The diameter of the aperture in meters
    travel_distance : ScalarFloat
        The distance traveled by the light in meters
    camera_pixel_size : ScalarFloat
        The pixel size of the camera in meters
    aperture_center : Optional[Float[Array, " 2"]], optional
        The center of the aperture in meters

    Returns
    -------
    at_camera : OpticalWavefront
        The calculated optical wavefront at the camera plane.

    Notes
    -----
    Algorithm:

    - Propagate the lightwave through the sample using linear interaction
    - Apply optical zoom to the wavefront
    - Apply a circular aperture to the zoomed wavefront
    - Propagate the wavefront to the camera plane using Fraunhofer propagation
    """
    at_sample_plane: OpticalWavefront = linear_interaction(
        sample=sample_cut,
        light=lightwave,
    )
    zoomed_wave: OpticalWavefront = optical_zoom(at_sample_plane, zoom_factor)
    center_to_use = aperture_center if aperture_center is not None else 0.0
    after_aperture: OpticalWavefront = circular_aperture(
        zoomed_wave, aperture_diameter, center_to_use
    )
    at_camera: OpticalWavefront = fraunhofer_prop(
        after_aperture, travel_distance
    )
    return at_camera


@jaxtyped(typechecker=beartype)
def simple_diffractogram(
    sample_cut: SampleFunction,
    lightwave: OpticalWavefront,
    zoom_factor: ScalarFloat,
    aperture_diameter: ScalarFloat,
    travel_distance: ScalarFloat,
    camera_pixel_size: ScalarFloat,
    aperture_center: Optional[Float[Array, " 2"]] = None,
) -> Diffractogram:
    """Calculate the diffractogram of a sample using a simple model.

    The lightwave interacts with the sample linearly, and is then
    zoomed optically. Following this it interacts with a circular
    aperture before propagating to the camera plane using scaled
    Fraunhofer propagation to match the detector pixel size.

    Parameters
    ----------
    sample_cut : SampleFunction
        The sample function representing the optical properties of the
        sample
    lightwave : OpticalWavefront
        The incoming optical wavefront
    zoom_factor : ScalarFloat
        The zoom factor for the optical system
    aperture_diameter : ScalarFloat
        The diameter of the aperture in meters
    travel_distance : ScalarFloat
        The distance traveled by the light in meters
    camera_pixel_size : ScalarFloat
        The pixel size of the detector/camera in meters
    aperture_center : Optional[Float[Array, " 2"]], optional
        The center of the aperture in meters

    Returns
    -------
    Diffractogram
        The calculated diffractogram of the sample. Output shape
        matches the input lightwave field shape.

    Notes
    -----
    Algorithm:

    - Propagate the lightwave through the sample using linear
      interaction
    - Apply optical zoom to the wavefront
    - Apply a circular aperture to the zoomed wavefront
    - Propagate to the camera plane using scaled Fraunhofer propagation
      which outputs at the specified camera pixel size
    - Calculate the field intensity of the camera image
    - Create a diffractogram from the camera image
    """
    at_sample_plane: OpticalWavefront = linear_interaction(
        sample=sample_cut,
        light=lightwave,
    )
    zoomed_wave: OpticalWavefront = optical_zoom(at_sample_plane, zoom_factor)
    center_to_use = aperture_center if aperture_center is not None else 0.0
    after_aperture: OpticalWavefront = circular_aperture(
        zoomed_wave, aperture_diameter, center_to_use
    )

    at_camera: OpticalWavefront = fraunhofer_prop_scaled(
        after_aperture, travel_distance, camera_pixel_size
    )

    camera_image: Float[Array, " H W"] = field_intensity(at_camera.field)
    diffractogram: Diffractogram = make_diffractogram(
        image=camera_image,
        wavelength=at_camera.wavelength,
        dx=at_camera.dx,
    )
    return diffractogram


@jaxtyped(typechecker=beartype)
def simple_microscope(
    sample: SampleFunction,
    positions: Num[Array, " n 2"],
    lightwave: OpticalWavefront,
    zoom_factor: ScalarFloat,
    aperture_diameter: ScalarFloat,
    travel_distance: ScalarFloat,
    camera_pixel_size: ScalarFloat,
    aperture_center: Optional[Float[Array, " 2"]] = None,
) -> MicroscopeData:
    """Calculate the 3D diffractograms of the entire imaging.

    This cuts the sample, and then generates a diffractogram with the
    desired camera pixel size - all done in parallel across all available
    GPUs. Done at every pixel positions.

    The function automatically detects available GPUs and distributes
    computation across devices using JAX's modern sharding API. For
    ptychography experiments with hundreds of scan positions, this
    provides significant speedup on multi-GPU systems.

    Implementation Logic
    --------------------
    The function orchestrates parallel diffractogram computation through
    a four-stage pipeline:

    1. **Position Preparation**:
       - Converts physical positions to pixel coordinates via positions/dx
       - Detects available devices (GPUs/TPUs) via jax.devices()
       - Pads position array to nearest multiple of device count for
         even distribution
       - Example: 400 positions on 7 GPUs → pad to 406 (7 × 58)

    2. **Data Sharding** (Automatic Multi-GPU Distribution):
       - Creates Mesh with all devices along "data" axis
       - Defines NamedSharding(P("data", None)) for position dimension
       - Uses jax.device_put to explicitly distribute positions across
         devices
       - Each device receives n_positions // n_devices positions
       - Critical for memory: sharding prevents OOM by distributing both
         input positions and output diffractograms

    3. **Parallel Computation**:
       - jax.vmap over positions dimension with in_axes=(None, 0)
         broadcasts sample to all positions
       - For each position: dynamic_slice extracts sample cutout,
         simple_diffractogram computes forward model
       - JAX's SPMD compiler ensures each device processes only its
         shard
       - Outputs remain sharded: Float[Array, "padded hh ww"] distributed
         across devices

    4. **Result Assembly**:
       - Trims padded diffractograms back to original count via [:n]
       - Creates MicroscopeData with trimmed images and original positions
       - JAX automatically manages cross-device communication

    Parameters
    ----------
    sample : SampleFunction
        The sample function representing the optical properties of the
        sample
    positions : Num[Array, " n 2"]
        The positions in the sample plane where the diffractograms are
        calculated. Physical coordinates in meters.
    lightwave : OpticalWavefront
        The incoming optical wavefront
    zoom_factor : ScalarFloat
        The zoom factor for the optical system
    aperture_diameter : ScalarFloat
        The diameter of the aperture in meters
    travel_distance : ScalarFloat
        The distance traveled by the light in meters
    camera_pixel_size : ScalarFloat
        The pixel size of the detector/camera in meters
    aperture_center : Optional[Float[Array, " 2"]], optional
        The center of the aperture in meters

    Returns
    -------
    MicroscopeData
        The calculated diffractograms of the sample at the specified
        positions. Output diffractogram shape matches the lightwave
        field shape.

    Notes
    -----
    **Multi-GPU Performance**:

    - Sharding overhead is negligible (<100ms) compared to computation
      time (minutes for hundreds of positions)
    - Speedup is near-linear with device count for large n (>100
      positions)
    - Memory per device: ~(n_total / n_devices) × diffractogram_size
    - Padding waste: at most (n_devices - 1) extra computations,
      typically <2% overhead

    **Memory Characteristics**:

    For n=400 positions, 256×256 diffractograms, float32:
    - Single GPU: 400 × 256 × 256 × 4 bytes ≈ 100 MB (tractable)
    - Output sharding prevents accumulation on single device
    - Forward model peak memory: ~2-3× output size during vmap execution

    **When Sharding Helps**:

    - Multi-GPU systems: Always beneficial for n > 50
    - Single GPU: No overhead, padding waste is minimal
    - CPU: Sharding across multiple CPUs provides moderate speedup

    **Design Decisions**:

    - Padding strategy ensures compatibility with any device count
      (avoids "not divisible by" errors)
    - NamedSharding chosen over legacy PositionalSharding for JAX 0.7.1+
      compatibility
    - P("data", None) shards only position dimension, broadcasts sample
      (sample is read-only and fits in device memory)
    - Explicit jax.device_put ensures sharding happens before vmap, not
      during (avoids runtime sharding overhead)
    """
    interaction_size: Tuple[int, int] = lightwave.field.shape
    pixel_positions: Float[Array, " n 2"] = positions / lightwave.dx
    num_positions: int = pixel_positions.shape[0]
    devices: list = jax.devices()
    num_devices: int = len(devices)
    padded_size: int = (
        (num_positions + num_devices - 1) // num_devices
    ) * num_devices
    padding_needed: int = padded_size - num_positions
    if padding_needed > 0:
        padding: Float[Array, " pad 2"] = jnp.zeros(
            (padding_needed, 2), dtype=pixel_positions.dtype
        )
        padded_positions: Float[Array, " padded 2"] = jnp.concatenate(
            [pixel_positions, padding], axis=0
        )
    else:
        padded_positions: Float[Array, " n 2"] = pixel_positions
    mesh: Mesh = Mesh(devices, axis_names=("data",))
    positions_sharding: NamedSharding = NamedSharding(mesh, P("data", None))
    sharded_pixel_positions: Float[Array, " padded 2"] = jax.device_put(
        padded_positions, positions_sharding
    )

    def diffractogram_at_position(
        sample: SampleFunction, this_position: Num[Array, " 2"]
    ) -> Diffractogram:
        start_cut_x: Int[Array, " "] = jnp.floor(
            this_position[0] - (0.5 * interaction_size[1])
        ).astype(int)
        start_cut_y: Int[Array, " "] = jnp.floor(
            this_position[1] - (0.5 * interaction_size[0])
        ).astype(int)
        cutout_sample: Complex[Array, " hh ww"] = jax.lax.dynamic_slice(
            sample.sample,
            (start_cut_y, start_cut_x),
            (interaction_size[0], interaction_size[1]),
        )
        this_sample: SampleFunction = make_sample_function(
            sample=cutout_sample,
            dx=sample.dx,
        )
        this_diffractogram: Diffractogram = simple_diffractogram(
            sample_cut=this_sample,
            lightwave=lightwave,
            zoom_factor=zoom_factor,
            aperture_diameter=aperture_diameter,
            travel_distance=travel_distance,
            camera_pixel_size=camera_pixel_size,
            aperture_center=aperture_center,
        )
        return this_diffractogram.image

    diffraction_images_padded: Float[Array, " padded hh ww"] = jax.vmap(
        diffractogram_at_position, in_axes=(None, 0)
    )(sample, sharded_pixel_positions)
    diffraction_images: Float[Array, " n hh ww"] = diffraction_images_padded[
        :num_positions
    ]
    combined_data: MicroscopeData = make_microscope_data(
        image_data=diffraction_images,
        positions=positions,
        wavelength=lightwave.wavelength,
        dx=camera_pixel_size,
    )
    return combined_data
