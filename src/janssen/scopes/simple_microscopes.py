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
from jaxtyping import Array, Complex, Float, Int, Num, jaxtyped

from janssen.optics.apertures import circular_aperture
from janssen.optics.helper import field_intensity
from janssen.prop import fraunhofer_prop, optical_zoom
from janssen.utils import (
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

jax.config.update("jax_enable_x64", True)


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
        The center of the aperture in pixels

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
    detector_shape: Tuple[int, int] = (256, 256),
    aperture_center: Optional[Float[Array, " 2"]] = None,
) -> Diffractogram:
    """Calculate the diffractogram of a sample using a simple model.

    The lightwave interacts with the sample linearly, and is then
    zoomed optically. Following this it interacts with a circular
    aperture before propagating to the camera plane.
    The wavefront is then resized to the detector shape and pixel size.
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
    detector_shape : Tuple[int, int], optional
        The shape of the detector in pixels (height, width).
        Defaults to (256, 256).
    aperture_center : Optional[Float[Array, " 2"]], optional
        The center of the aperture in pixels

    Returns
    -------
    Diffractogram
        The calculated diffractogram of the sample

    Notes
    -----
    Algorithm:

    - Propagate the lightwave through the sample using linear
    interaction
    - Apply optical zoom to the wavefront
    - Apply a circular aperture to the zoomed wavefront
    - Propagate the wavefront to the camera plane using Fraunhofer
    propagation
    - Resize to detector shape and pixel size
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
    at_camera: OpticalWavefront = fraunhofer_prop(
        after_aperture, travel_distance
    )

    # Fit wavefront to detector shape (JAX-safe: always pad then crop)
    # 1. Pad field to at least detector size (handles detector > field)
    # 2. Crop to exact detector size (handles detector < field)
    field_h, field_w = at_camera.field.shape
    det_h, det_w = detector_shape

    # Calculate padding needed (0 if field already large enough)
    pad_h = max(0, (det_h - field_h + 1) // 2)
    pad_w = max(0, (det_w - field_w + 1) // 2)
    padded_field: Complex[Array, " H W"] = jnp.pad(
        at_camera.field,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0.0,
    )

    # Crop to exact detector size from center
    padded_h, padded_w = field_h + 2 * pad_h, field_w + 2 * pad_w
    start_h = (padded_h - det_h) // 2
    start_w = (padded_w - det_w) // 2
    resized_field: Complex[Array, " H W"] = jax.lax.dynamic_slice(
        padded_field,
        (start_h, start_w),
        (det_h, det_w),
    )

    at_camera_resized: OpticalWavefront = make_optical_wavefront(
        field=resized_field,
        wavelength=at_camera.wavelength,
        dx=camera_pixel_size,
        z_position=at_camera.z_position,
    )

    scaled_camera_image: Float[Array, " H W"] = field_intensity(
        at_camera_resized.field
    )
    diffractogram: Diffractogram = make_diffractogram(
        image=scaled_camera_image,
        wavelength=at_camera_resized.wavelength,
        dx=at_camera_resized.dx,
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
    detector_shape: Tuple[int, int] = (256, 256),
    aperture_center: Optional[Float[Array, " 2"]] = None,
) -> MicroscopeData:
    """Calculate the 3D diffractograms of the entire imaging.

    This cuts the sample, and then generates a diffractogram with the
    desired camera pixel size - all done in parallel.
    Done at every pixel positions.

    Parameters
    ----------
    sample : SampleFunction
        The sample function representing the optical properties of the
        sample
    positions : Num[Array, " n 2"]
        The positions in the sample plane where the diffractograms are
        calculated.
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
    detector_shape : Tuple[int, int], optional
        The shape of the detector in pixels (height, width).
        Defaults to (256, 256).
    aperture_center : Optional[Float[Array, " 2"]], optional
        The center of the aperture in pixels

    Returns
    -------
    MicroscopeData
        The calculated diffractograms of the sample at the specified
        positions

    Notes
    -----
    Algorithm:

    - Get the size of the lightwave field
    - Calculate the pixel positions in the sample plane
    - For each position, cut out the sample and calculate the
    diffractogram
    - Combine the diffractograms into a single MicroscopeData object
    - Return the MicroscopeData object
    """
    interaction_size: Tuple[int, int] = lightwave.field.shape
    pixel_positions: Float[Array, " n 2"] = positions / lightwave.dx

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
            detector_shape=detector_shape,
            aperture_center=aperture_center,
        )
        return this_diffractogram.image

    diffraction_images: Float[Array, " n hh ww"] = jax.vmap(
        diffractogram_at_position, in_axes=(None, 0)
    )(sample, pixel_positions)
    combined_data: MicroscopeData = make_microscope_data(
        image_data=diffraction_images,
        positions=positions,
        wavelength=lightwave.wavelength,
        dx=camera_pixel_size,
    )
    return combined_data
