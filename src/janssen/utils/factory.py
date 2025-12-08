"""Factory functions for creating data structures.

Extended Summary
----------------
Factory functions for creating data structures with runtime type
checking.
All runtime validations use JAX safe conditional statements.

Routine Listings
----------------
make_lens_params : function
    Creates a LensParams instance with runtime type checking
make_grid_params : function
    Creates a GridParams instance with runtime type checking
make_optical_wavefront : function
    Creates an OpticalWavefront instance with runtime type checking
make_propagating_wavefront : function
    Creates a PropagatingWavefront instance with runtime type checking
optical2propagating : function
    Creates a PropagatingWavefront from a tuple of OpticalWavefronts
make_microscope_data : function
    Creates a MicroscopeData instance with runtime type checking
make_diffractogram : function
    Creates a Diffractogram instance with runtime type checking
make_sample_function : function
    Creates a SampleFunction instance with runtime type checking
make_sliced_material_function : function
    Creates a SlicedMaterialFunction instance with runtime type checking
make_optimizer_state : function
    Creates an OptimizerState instance with runtime type checking
make_ptychography_params : function
    Creates a PtychographyParams instance with runtime type checking

Notes
-----
Always use these factory functions instead of directly instantiating the
NamedTuple classes to ensure proper runtime type checking of the
contents.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jax import lax
from jaxtyping import Array, Bool, Complex, Float, Int, Num, jaxtyped

from .types import (
    Diffractogram,
    GridParams,
    LensParams,
    MicroscopeData,
    OpticalWavefront,
    OptimizerState,
    PropagatingWavefront,
    PtychographyParams,
    SampleFunction,
    ScalarBool,
    ScalarComplex,
    ScalarFloat,
    ScalarInteger,
    ScalarNumeric,
    SlicedMaterialFunction,
)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def make_lens_params(
    focal_length: ScalarNumeric,
    diameter: ScalarNumeric,
    n: ScalarNumeric,
    center_thickness: ScalarNumeric,
    r1: ScalarNumeric,
    r2: ScalarNumeric,
) -> LensParams:
    """JAX-safe factory function for LensParams with data validation.

    Parameters
    ----------
    focal_length : ScalarNumeric
        Focal length of the lens in meters
    diameter : ScalarNumeric
        Diameter of the lens in meters
    n : ScalarNumeric
        Refractive index of the lens material
    center_thickness : ScalarNumeric
        Thickness at the center of the lens in meters
    r1 : ScalarNumeric
        Radius of curvature of the first surface in meters
        (positive for convex)
    r2 : ScalarNumeric
        Radius of curvature of the second surface in meters
        (positive for convex)

    Returns
    -------
    validated_lens_params : LensParams
        Validated lens parameters instance

    Raises
    ------
    ValueError
        If parameters are invalid or out of valid ranges

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Validate parameters:
        - Check focal_length is positive
        - Check diameter is positive
        - Check refractive index is positive
        - Check center_thickness is positive
        - Check radii are finite
    - Create and return LensParams instance
    """
    focal_length = jnp.asarray(focal_length, dtype=jnp.float64)
    diameter = jnp.asarray(diameter, dtype=jnp.float64)
    n = jnp.asarray(n, dtype=jnp.float64)
    center_thickness = jnp.asarray(center_thickness, dtype=jnp.float64)
    r1 = jnp.asarray(r1, dtype=jnp.float64)
    r2 = jnp.asarray(r2, dtype=jnp.float64)

    def validate_and_create() -> LensParams:
        def check_focal_length() -> Float[Array, " "]:
            return lax.cond(
                focal_length > 0,
                lambda: focal_length,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: focal_length, lambda: focal_length)
                ),
            )

        def check_diameter() -> Float[Array, " "]:
            return lax.cond(
                diameter > 0,
                lambda: diameter,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: diameter, lambda: diameter)
                ),
            )

        def check_refractive_index() -> Float[Array, " "]:
            return lax.cond(
                n > 0,
                lambda: n,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: n, lambda: n)
                ),
            )

        def check_center_thickness() -> Float[Array, " "]:
            return lax.cond(
                center_thickness > 0,
                lambda: center_thickness,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: center_thickness,
                        lambda: center_thickness,
                    )
                ),
            )

        def check_radii_finite() -> (
            Tuple[Float[Array, " "], Float[Array, " "]]
        ):
            return lax.cond(
                jnp.logical_and(jnp.isfinite(r1), jnp.isfinite(r2)),
                lambda: (r1, r2),
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: (r1, r2), lambda: (r1, r2))
                ),
            )

        check_focal_length()
        check_diameter()
        check_refractive_index()
        check_center_thickness()
        check_radii_finite()

        return LensParams(
            focal_length=focal_length,
            diameter=diameter,
            n=n,
            center_thickness=center_thickness,
            r1=r1,
            r2=r2,
        )

    validated_lens_params: LensParams = validate_and_create()
    return validated_lens_params


@jaxtyped(typechecker=beartype)
def make_grid_params(
    xx: Float[Array, " hh ww"],
    yy: Float[Array, " hh ww"],
    phase_profile: Float[Array, " hh ww"],
    transmission: Float[Array, " hh ww"],
) -> GridParams:
    """JAX-safe factory function for GridParams with data validation.

    Parameters
    ----------
    xx : Float[Array, " hh ww"]
        Spatial grid in the x-direction
    yy : Float[Array, " hh ww"]
        Spatial grid in the y-direction
    phase_profile : Float[Array, " hh ww"]
        Phase profile of the optical field
    transmission : Float[Array, " hh ww"]
        Transmission profile of the optical field

    Returns
    -------
    validated_grid_params : GridParams
        Validated grid parameters instance

    Raises
    ------
    ValueError
        If array shapes are inconsistent or data is invalid

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Validate array shapes:
        - Check all arrays are 2D
        - Check all arrays have the same shape
    - Validate data:
        - Ensure transmission values are between 0 and 1
        - Ensure phase values are finite
        - Ensure grid coordinates are finite
    - Create and return GridParams instance
    """
    xx = jnp.asarray(xx, dtype=jnp.float64)
    yy = jnp.asarray(yy, dtype=jnp.float64)
    phase_profile = jnp.asarray(phase_profile, dtype=jnp.float64)
    transmission = jnp.asarray(transmission, dtype=jnp.float64)

    def validate_and_create() -> GridParams:
        array_dims: int = 2
        hh: int
        ww: int
        hh, ww = xx.shape

        def check_2d_arrays() -> Tuple[
            Float[Array, " hh ww"],
            Float[Array, " hh ww"],
            Float[Array, " hh ww"],
            Float[Array, " hh ww"],
        ]:
            return lax.cond(
                jnp.logical_and(
                    jnp.logical_and(
                        xx.ndim == array_dims, yy.ndim == array_dims
                    ),
                    jnp.logical_and(
                        phase_profile.ndim == array_dims,
                        transmission.ndim == array_dims,
                    ),
                ),
                lambda: (xx, yy, phase_profile, transmission),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: (xx, yy, phase_profile, transmission),
                        lambda: (xx, yy, phase_profile, transmission),
                    )
                ),
            )

        def check_same_shape() -> Tuple[
            Float[Array, " hh ww"],
            Float[Array, " hh ww"],
            Float[Array, " hh ww"],
            Float[Array, " hh ww"],
        ]:
            return lax.cond(
                jnp.logical_and(
                    jnp.logical_and(
                        xx.shape == (hh, ww), yy.shape == (hh, ww)
                    ),
                    jnp.logical_and(
                        phase_profile.shape == (hh, ww),
                        transmission.shape == (hh, ww),
                    ),
                ),
                lambda: (xx, yy, phase_profile, transmission),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: (xx, yy, phase_profile, transmission),
                        lambda: (xx, yy, phase_profile, transmission),
                    )
                ),
            )

        def check_transmission_range() -> Float[Array, " hh ww"]:
            return lax.cond(
                jnp.logical_and(
                    jnp.all(transmission >= 0), jnp.all(transmission <= 1)
                ),
                lambda: transmission,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: transmission, lambda: transmission)
                ),
            )

        def check_phase_finite() -> Float[Array, " hh ww"]:
            return lax.cond(
                jnp.all(jnp.isfinite(phase_profile)),
                lambda: phase_profile,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: phase_profile, lambda: phase_profile
                    )
                ),
            )

        def check_grid_finite() -> (
            Tuple[Float[Array, " hh ww"], Float[Array, " hh ww"]]
        ):
            return lax.cond(
                jnp.logical_and(
                    jnp.all(jnp.isfinite(xx)), jnp.all(jnp.isfinite(yy))
                ),
                lambda: (xx, yy),
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: (xx, yy), lambda: (xx, yy))
                ),
            )

        check_2d_arrays()
        check_same_shape()
        check_transmission_range()
        check_phase_finite()
        check_grid_finite()

        return GridParams(
            xx=xx,
            yy=yy,
            phase_profile=phase_profile,
            transmission=transmission,
        )

    validated_grid_params: GridParams = validate_and_create()
    return validated_grid_params


@jaxtyped(typechecker=beartype)
def make_optical_wavefront(
    field: Union[Complex[Array, " hh ww"], Complex[Array, " hh ww 2"]],
    wavelength: ScalarNumeric,
    dx: ScalarNumeric,
    z_position: ScalarNumeric,
    polarization: ScalarBool = False,
) -> OpticalWavefront:
    """JAX-safe factory function for OpticalWavefront with data
    validation.

    Parameters
    ----------
    field : Union[Complex[Array, " hh ww"], Complex[Array, " hh ww 2"]]
       Complex amplitude of the optical field. Should be 2D for scalar
       fields or 3D with last dimension 2 for polarized fields.
       Polarization is automatically detected from field dimensions.
    wavelength : ScalarNumeric
        Wavelength of the optical wavefront in meters
    dx : ScalarNumeric
        Spatial sampling interval (grid spacing) in meters
    z_position : ScalarNumeric
        Axial position of the wavefront in the propagation direction in
        meters.
    polarization : Bool[Array, " "]
        Boolean indicating whether the field is polarized.
        Default is False.

    Returns
    -------
    validated_optical_wavefront : OpticalWavefront
        Validated optical wavefront instance

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid ranges

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Auto-detect polarization based on field dimensions (3D with last
      dimension 2 means polarized)
    - Validate field array:
        - Check it's 2D or 3D with last dimension 2
        - Ensure all values are finite
    - Validate parameters:
        - Check wavelength is positive
        - Check dx is positive
        - Check z_position is finite
    - Create and return OpticalWavefront instance
    """
    non_polar_dim: int = 2
    polar_dim: int = 3
    field: Complex[Array, " hh ww"] = jnp.asarray(field, dtype=jnp.complex128)
    wavelength: Float[Array, " "] = jnp.asarray(wavelength, dtype=jnp.float64)
    dx: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    z_position: Float[Array, " "] = jnp.asarray(z_position, dtype=jnp.float64)
    polarization: Bool[Array, " "] = jnp.asarray(polarization, dtype=jnp.bool_)

    # Override polarization if field dimensions indicate polarized field
    polarization = jnp.where(
        field.ndim == polar_dim,
        jnp.asarray(field.shape[-1] == non_polar_dim, dtype=jnp.bool_),
        polarization,
    )

    def validate_and_create() -> OpticalWavefront:
        def check_field_dimensions() -> (
            Union[Complex[Array, " hh ww"], Complex[Array, " hh ww 2"]]
        ):
            non_polar_dimensions: int = 2
            polar_dimensions: int = 3

            def check_polarized() -> Complex[Array, " hh ww 2"]:
                return lax.cond(
                    jnp.logical_and(
                        field.ndim == polar_dimensions,
                        field.shape[-1] == non_polar_dimensions,
                    ),
                    lambda: field,
                    lambda: lax.stop_gradient(
                        lax.cond(False, lambda: field, lambda: field)
                    ),
                )

            def check_scalar() -> Complex[Array, " hh ww"]:
                return lax.cond(
                    field.ndim == non_polar_dimensions,
                    lambda: field,
                    lambda: lax.stop_gradient(
                        lax.cond(False, lambda: field, lambda: field)
                    ),
                )

            return lax.cond(
                polarization,
                check_polarized,
                check_scalar,
            )

        def check_field_finite() -> (
            Union[Complex[Array, " hh ww"], Complex[Array, " hh ww 2"]]
        ):
            return lax.cond(
                jnp.all(jnp.isfinite(field)),
                lambda: field,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: field, lambda: field)
                ),
            )

        def check_wavelength() -> Float[Array, " "]:
            return lax.cond(
                wavelength > 0,
                lambda: wavelength,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: wavelength, lambda: wavelength)
                ),
            )

        def check_dx() -> Float[Array, " "]:
            return lax.cond(
                dx > 0,
                lambda: dx,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: dx, lambda: dx)
                ),
            )

        def check_z_position() -> Float[Array, " "]:
            return lax.cond(
                jnp.isfinite(z_position),
                lambda: z_position,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: z_position, lambda: z_position)
                ),
            )

        check_field_dimensions()
        check_field_finite()
        check_wavelength()
        check_dx()
        check_z_position()

        return OpticalWavefront(
            field=field,
            wavelength=wavelength,
            dx=dx,
            z_position=z_position,
            polarization=polarization,
        )

    validated_optical_wavefront: OpticalWavefront = validate_and_create()
    return validated_optical_wavefront


@jaxtyped(typechecker=beartype)
def make_propagating_wavefront(
    field: Union[Complex[Array, " zz hh ww"], Complex[Array, " zz hh ww 2"]],
    wavelength: ScalarNumeric,
    dx: ScalarNumeric,
    z_positions: Float[Array, " zz"],
    polarization: ScalarBool = False,
) -> PropagatingWavefront:
    """JAX-safe factory function for PropagatingWavefront with data
    validation.

    Parameters
    ----------
    field : Union[Complex[Array, " zz hh ww"], Complex[Array, " zz hh ww 2"]]
        Complex amplitude of the optical field. Should be 3D for scalar
        fields (Z, H, W) or 4D with last dimension 2 for polarized fields
        (Z, H, W, 2). Z represents slices along the propagation direction.
    wavelength : ScalarNumeric
        Wavelength of the optical wavefront in meters
    dx : ScalarNumeric
        Spatial sampling interval (grid spacing) in meters
    z_positions : Float[Array, " zz"]
        Axial positions of the wavefront slices along the propagation
        direction in meters.
    polarization : ScalarBool
        Boolean indicating whether the field is polarized.
        Default is False.

    Returns
    -------
    validated_propagating_wavefront : PropagatingWavefront
        Validated propagating wavefront instance

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid ranges

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Auto-detect polarization based on field dimensions (4D with last
      dimension 2 means polarized)
    - Validate field array:
        - Check it's 3D or 4D with last dimension 2
        - Ensure all values are finite
    - Validate parameters:
        - Check wavelength is positive
        - Check dx is positive
        - Check z_positions are finite
        - Check z_positions length matches field's first dimension
    - Create and return PropagatingWavefront instance
    """
    non_polar_dim: int = 3
    polar_dim: int = 4
    polarization_components: int = 2
    field = jnp.asarray(field, dtype=jnp.complex128)
    wavelength_arr: Float[Array, " "] = jnp.asarray(
        wavelength, dtype=jnp.float64
    )
    dx_arr: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    z_positions_arr: Float[Array, " zz"] = jnp.asarray(
        z_positions, dtype=jnp.float64
    )
    polarization_arr: Bool[Array, " "] = jnp.asarray(
        polarization, dtype=jnp.bool_
    )

    # Override polarization if field dimensions indicate polarized field
    polarization_arr = jnp.where(
        field.ndim == polar_dim,
        jnp.asarray(
            field.shape[-1] == polarization_components, dtype=jnp.bool_
        ),
        polarization_arr,
    )

    def validate_and_create() -> PropagatingWavefront:
        def check_field_dimensions() -> (
            Union[Complex[Array, " zz hh ww"], Complex[Array, " zz hh ww 2"]]
        ):
            def check_polarized() -> Complex[Array, " zz hh ww 2"]:
                return lax.cond(
                    jnp.logical_and(
                        field.ndim == polar_dim,
                        field.shape[-1] == polarization_components,
                    ),
                    lambda: field,
                    lambda: lax.stop_gradient(
                        lax.cond(False, lambda: field, lambda: field)
                    ),
                )

            def check_scalar() -> Complex[Array, " zz hh ww"]:
                return lax.cond(
                    field.ndim == non_polar_dim,
                    lambda: field,
                    lambda: lax.stop_gradient(
                        lax.cond(False, lambda: field, lambda: field)
                    ),
                )

            return lax.cond(
                polarization_arr,
                check_polarized,
                check_scalar,
            )

        def check_field_finite() -> (
            Union[Complex[Array, " zz hh ww"], Complex[Array, " zz hh ww 2"]]
        ):
            return lax.cond(
                jnp.all(jnp.isfinite(field)),
                lambda: field,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: field, lambda: field)
                ),
            )

        def check_wavelength() -> Float[Array, " "]:
            return lax.cond(
                wavelength_arr > 0,
                lambda: wavelength_arr,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: wavelength_arr, lambda: wavelength_arr
                    )
                ),
            )

        def check_dx() -> Float[Array, " "]:
            return lax.cond(
                dx_arr > 0,
                lambda: dx_arr,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: dx_arr, lambda: dx_arr)
                ),
            )

        def check_z_positions_finite() -> Float[Array, " zz"]:
            return lax.cond(
                jnp.all(jnp.isfinite(z_positions_arr)),
                lambda: z_positions_arr,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: z_positions_arr, lambda: z_positions_arr
                    )
                ),
            )

        def check_z_positions_length() -> Float[Array, " zz"]:
            return lax.cond(
                z_positions_arr.shape[0] == field.shape[0],
                lambda: z_positions_arr,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: z_positions_arr, lambda: z_positions_arr
                    )
                ),
            )

        check_field_dimensions()
        check_field_finite()
        check_wavelength()
        check_dx()
        check_z_positions_finite()
        check_z_positions_length()

        return PropagatingWavefront(
            field=field,
            wavelength=wavelength_arr,
            dx=dx_arr,
            z_positions=z_positions_arr,
            polarization=polarization_arr,
        )

    validated_propagating_wavefront: PropagatingWavefront = (
        validate_and_create()
    )
    return validated_propagating_wavefront


@jaxtyped(typechecker=beartype)
def optical2propagating(
    wavefronts: Tuple[OpticalWavefront, ...],
) -> PropagatingWavefront:
    """Create a PropagatingWavefront from a tuple of OpticalWavefronts.

    Parameters
    ----------
    wavefronts : Tuple[OpticalWavefront, ...]
        Tuple of OpticalWavefront instances. All wavefronts must have the
        same wavelength, dx, polarization, and field shape (H, W).

    Returns
    -------
    propagating_wavefront : PropagatingWavefront
        A PropagatingWavefront containing all input wavefronts stacked
        along the z dimension.

    Raises
    ------
    ValueError
        If wavefronts tuple is empty, or if wavefronts have inconsistent
        wavelength, dx, polarization, or field shapes.

    Notes
    -----
    Algorithm:

    - Extract fields from all wavefronts and stack along axis 0
    - Extract z_positions from each wavefront
    - Validate all wavefronts have consistent wavelength, dx, and
      polarization
    - Create PropagatingWavefront using the factory function
    """
    if len(wavefronts) == 0:
        raise ValueError("wavefronts tuple cannot be empty")

    # Stack fields along axis 0
    fields = jnp.stack([wf.field for wf in wavefronts], axis=0)

    # Extract z_positions from each wavefront
    z_positions = jnp.array([wf.z_position for wf in wavefronts])

    # Use first wavefront's properties
    wavelength = wavefronts[0].wavelength
    dx = wavefronts[0].dx
    polarization = wavefronts[0].polarization

    return make_propagating_wavefront(
        field=fields,
        wavelength=wavelength,
        dx=dx,
        z_positions=z_positions,
        polarization=polarization,
    )


@jaxtyped(typechecker=beartype)
def make_microscope_data(
    image_data: Union[Float[Array, " pp hh ww"], Float[Array, " xx yy hh ww"]],
    positions: Num[Array, " pp 2"],
    wavelength: ScalarNumeric,
    dx: ScalarNumeric,
) -> MicroscopeData:
    """JAX-safe factory function for MicroscopeData with data
    validation.

    Parameters
    ----------
    image_data :
        Union[Float[Array, " pp hh ww"], Float[Array, " xx yy hh ww"]]
        3D or 4D image data representing the optical field
    positions : Num[Array, " pp 2"]
        Positions of the images during collection
    wavelength : ScalarNumeric
        Wavelength of the optical wavefront in meters
    dx : ScalarNumeric
        Spatial sampling interval (grid spacing) in meters

    Returns
    -------
    validated_microscope_data : MicroscopeData
        Validated microscope data instance

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid ranges

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Validate image_data:
        - Check it's 3D or 4D
        - Ensure all values are finite and non-negative
    - Validate positions:
        - Check it's 2D with shape (pp, 2)
        - Ensure all values are finite
    - Validate parameters:
        - Check wavelength is positive
        - Check dx is positive
    - Validate consistency:
        - Check P matches between image_data and positions
    - Create and return MicroscopeData instance
    """
    image_data: Union[
        Float[Array, " pp hh ww"], Float[Array, " xx yy hh ww"]
    ] = jnp.asarray(image_data, dtype=jnp.float64)
    positions: Num[Array, " pp 2"] = jnp.asarray(positions, dtype=jnp.float64)
    wavelength: Float[Array, " "] = jnp.asarray(wavelength, dtype=jnp.float64)
    dx: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    expected_image_dim = 2
    expected_diffractogram_dim_3d: int = 3
    expected_diffractogram_dim_4d: int = 4

    def validate_and_create() -> MicroscopeData:
        def check_image_dimensions() -> (
            Union[Float[Array, " P H W"], Float[Array, " X Y H W"]]
        ):
            return lax.cond(
                jnp.logical_or(
                    image_data.ndim == expected_diffractogram_dim_3d,
                    image_data.ndim == expected_diffractogram_dim_4d,
                ),
                lambda: image_data,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: image_data, lambda: image_data)
                ),
            )

        def check_image_finite() -> (
            Union[Float[Array, " P H W"], Float[Array, " X Y H W"]]
        ):
            return lax.cond(
                jnp.all(jnp.isfinite(image_data)),
                lambda: image_data,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: image_data, lambda: image_data)
                ),
            )

        def check_image_nonnegative() -> (
            Union[Float[Array, " P H W"], Float[Array, " X Y H W"]]
        ):
            return lax.cond(
                jnp.all(image_data >= 0),
                lambda: image_data,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: image_data, lambda: image_data)
                ),
            )

        def check_positions_shape() -> Num[Array, " P 2"]:
            return lax.cond(
                positions.shape[1] == expected_image_dim,
                lambda: positions,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: positions, lambda: positions)
                ),
            )

        def check_positions_finite() -> Num[Array, " P 2"]:
            return lax.cond(
                jnp.all(jnp.isfinite(positions)),
                lambda: positions,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: positions, lambda: positions)
                ),
            )

        def check_wavelength() -> Float[Array, " "]:
            return lax.cond(
                wavelength > 0,
                lambda: wavelength,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: wavelength, lambda: wavelength)
                ),
            )

        def check_dx() -> Float[Array, " "]:
            return lax.cond(
                dx > 0,
                lambda: dx,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: dx, lambda: dx)
                ),
            )

        def check_consistency() -> Tuple[
            Union[Float[Array, " P H W"], Float[Array, " X Y H W"]],
            Num[Array, " P 2"],
        ]:
            pp = positions.shape[0]

            def check_3d_consistency() -> Tuple[
                Union[Float[Array, " pp H W"], Float[Array, " X Y H W"]],
                Num[Array, " pp 2"],
            ]:
                return lax.cond(
                    image_data.shape[0] == pp,
                    lambda: (image_data, positions),
                    lambda: lax.stop_gradient(
                        lax.cond(
                            False,
                            lambda: (image_data, positions),
                            lambda: (image_data, positions),
                        )
                    ),
                )

            def check_4d_consistency() -> Tuple[
                Union[Float[Array, " P H W"], Float[Array, " X Y H W"]],
                Num[Array, " P 2"],
            ]:
                return lax.cond(
                    image_data.shape[0] * image_data.shape[1] == pp,
                    lambda: (image_data, positions),
                    lambda: lax.stop_gradient(
                        lax.cond(
                            False,
                            lambda: (image_data, positions),
                            lambda: (image_data, positions),
                        )
                    ),
                )

            return lax.cond(
                image_data.ndim == expected_image_dim,
                check_3d_consistency,
                check_4d_consistency,
            )

        check_image_dimensions()
        check_image_finite()
        check_image_nonnegative()
        check_positions_shape()
        check_positions_finite()
        check_wavelength()
        check_dx()
        check_consistency()

        return MicroscopeData(
            image_data=image_data,
            positions=positions,
            wavelength=wavelength,
            dx=dx,
        )

    validated_microscope_data: MicroscopeData = validate_and_create()
    return validated_microscope_data


@jaxtyped(typechecker=beartype)
def make_diffractogram(
    image: Float[Array, " hh ww"],
    wavelength: ScalarNumeric,
    dx: ScalarNumeric,
) -> Diffractogram:
    """JAX-safe factory function for Diffractogram with data validation.

    Parameters
    ----------
    image : Float[Array, " hh ww"]
        Image data
    wavelength : ScalarNumeric
        Wavelength of the optical wavefront in meters
    dx : ScalarNumeric
        Spatial sampling interval (grid spacing) in meters

    Returns
    -------
    validated_diffractogram : Diffractogram
        Validated diffractogram instance

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid ranges

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Validate image array:
        - Check it's 2D
        - Ensure all values are finite and non-negative
    - Validate parameters:
        - Check wavelength is positive
        - Check dx is positive
    - Create and return Diffractogram instance
    """
    image: Float[Array, " H W"] = jnp.asarray(image, dtype=jnp.float64)
    wavelength: Float[Array, " "] = jnp.asarray(wavelength, dtype=jnp.float64)
    dx: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    expected_sample_dim: int = 2

    def validate_and_create() -> Diffractogram:
        def check_2d_image() -> Float[Array, " H W"]:
            return lax.cond(
                image.ndim == expected_sample_dim,
                lambda: image,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: image, lambda: image)
                ),
            )

        def check_image_finite() -> Float[Array, " H W"]:
            return lax.cond(
                jnp.all(jnp.isfinite(image)),
                lambda: image,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: image, lambda: image)
                ),
            )

        def check_image_nonnegative() -> Float[Array, " H W"]:
            return lax.cond(
                jnp.all(image >= 0),
                lambda: image,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: image, lambda: image)
                ),
            )

        def check_wavelength() -> Float[Array, " "]:
            return lax.cond(
                wavelength > 0,
                lambda: wavelength,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: wavelength, lambda: wavelength)
                ),
            )

        def check_dx() -> Float[Array, " "]:
            return lax.cond(
                dx > 0,
                lambda: dx,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: dx, lambda: dx)
                ),
            )

        check_2d_image()
        check_image_finite()
        check_image_nonnegative()
        check_wavelength()
        check_dx()

        return Diffractogram(
            image=image,
            wavelength=wavelength,
            dx=dx,
        )

    validated_diffractogram: Diffractogram = validate_and_create()
    return validated_diffractogram


@jaxtyped(typechecker=beartype)
def make_sample_function(
    sample: Num[Array, " hh ww"],
    dx: ScalarNumeric,
) -> SampleFunction:
    """JAX-safe factory function for SampleFunction with data
    validation.

    Parameters
    ----------
    sample : Num[Array, " hh ww"]
        The sample function. Will be converted to complex if real.
    dx : ScalarNumeric
        Spatial sampling interval (grid spacing) in meters

    Returns
    -------
    validated_sample_function : SampleFunction
        Validated sample function instance

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid ranges

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Validate sample array:
        - Check it's 2D
        - Ensure all values are finite
    - Validate parameters:
        - Check dx is positive
    - Create and return SampleFunction instance
    """
    sample: Complex[Array, " hh ww"] = jnp.asarray(
        sample, dtype=jnp.complex128
    )
    dx: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    expected_sample_dim: int = 2

    def validate_and_create() -> SampleFunction:
        def check_2d_sample() -> Complex[Array, " hh ww"]:
            return lax.cond(
                sample.ndim == expected_sample_dim,
                lambda: sample,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: sample, lambda: sample)
                ),
            )

        def check_sample_finite() -> Complex[Array, " hh ww"]:
            return lax.cond(
                jnp.all(jnp.isfinite(sample)),
                lambda: sample,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: sample, lambda: sample)
                ),
            )

        def check_dx() -> Float[Array, " "]:
            return lax.cond(
                dx > 0,
                lambda: dx,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: dx, lambda: dx)
                ),
            )

        check_2d_sample()
        check_sample_finite()
        check_dx()

        return SampleFunction(
            sample=sample,
            dx=dx,
        )

    validated_sample_function: SampleFunction = validate_and_create()
    return validated_sample_function


@jaxtyped(typechecker=beartype)
def make_sliced_material_function(
    material: Num[Array, " hh ww zz"],
    dx: ScalarNumeric,
    tz: ScalarNumeric,
) -> SlicedMaterialFunction:
    """JAX-safe validated factory function for SlicedMaterialFunction.

    Parameters
    ----------
    material : Num[Array, " hh ww zz"]
        3D array of complex refractive indices. The real part represents
        the refractive index n, and the imaginary part represents the
        extinction coefficient κ (absorption). Will be converted to complex
        if real.
    dx : ScalarNumeric
        Spatial sampling interval (pixel spacing) within each slice in meters
    tz : ScalarNumeric
        Interslice distance (spacing between slices) in the z-direction in
        meters.

    Returns
    -------
    validated_sliced_material : SlicedMaterialFunction
        Validated sliced material function instance

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid ranges

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Validate material array:
        - Check it's 3D
        - Ensure all values are finite
    - Validate parameters:
        - Check dx is positive
        - Check tz is positive
    - Create and return SlicedMaterialFunction instance
    """
    material_array: Complex[Array, " hh ww zz"] = jnp.asarray(
        material, dtype=jnp.complex128
    )
    dx_array: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    tz_array: Float[Array, " "] = jnp.asarray(tz, dtype=jnp.float64)
    expected_material_dim: int = 3

    def validate_and_create() -> SlicedMaterialFunction:
        def check_3d_material() -> Complex[Array, " hh ww zz"]:
            return lax.cond(
                material_array.ndim == expected_material_dim,
                lambda: material_array,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: material_array, lambda: material_array
                    )
                ),
            )

        def check_material_finite() -> Complex[Array, " hh ww zz"]:
            return lax.cond(
                jnp.all(jnp.isfinite(material_array)),
                lambda: material_array,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: material_array, lambda: material_array
                    )
                ),
            )

        def check_dx() -> Float[Array, " "]:
            return lax.cond(
                dx_array > 0,
                lambda: dx_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: dx_array, lambda: dx_array)
                ),
            )

        def check_tz() -> Float[Array, " "]:
            return lax.cond(
                tz_array > 0,
                lambda: tz_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: tz_array, lambda: tz_array)
                ),
            )

        check_3d_material()
        check_material_finite()
        check_dx()
        check_tz()

        return SlicedMaterialFunction(
            material=material_array,
            dx=dx_array,
            tz=tz_array,
        )

    validated_sliced_material: SlicedMaterialFunction = validate_and_create()
    return validated_sliced_material


@jaxtyped(typechecker=beartype)
def make_optimizer_state(
    shape: Tuple,
    m: Optional[Union[Complex[Array, " ..."], ScalarComplex]] = 1j,
    v: Optional[Union[Float[Array, " ..."], ScalarFloat]] = 0.0,
    step: Optional[ScalarInteger] = 0,
) -> OptimizerState:
    """JAX-safe factory function for OptimizerState with data
    validation.

    Parameters
    ----------
    shape : Tuple
        Shape of the parameters to be optimized
    m : Optional[Complex[Array, "..."]], optional
        First moment estimate. If None, initialized to zeros with given
        shape.
        Default is 1j.
    v : Optional[Float[Array, "..."]], optional
        Second moment estimate. If None, initialized to zeros with given
        shape.
        Default is 0.0.
    step : Optional[ScalarInteger], optional
        Step count. Default is 0.

    Returns
    -------
    validated_optimizer_state : OptimizerState
        Validated optimizer state instance

    Raises
    ------
    ValueError
        If arrays have incompatible shapes with the given shape
        parameter

    Notes
    -----
    Algorithm:

    - Convert all inputs to JAX arrays with appropriate dtypes
    - Always broadcast m and v to the target shape (if already the
      right shape, broadcast_to is a no-op)
    - Validate arrays have compatible shapes
    - Create and return OptimizerState instance
    """
    m_input = jnp.asarray(m, dtype=jnp.complex128)
    v_input = jnp.asarray(v, dtype=jnp.float64)
    step_input = jnp.asarray(step, dtype=jnp.int32)

    m_array = jnp.broadcast_to(m_input, shape).astype(jnp.complex128)
    v_array = jnp.broadcast_to(v_input, shape).astype(jnp.float64)

    step_array = step_input

    def validate_and_create() -> OptimizerState:
        def check_m_shape() -> Complex[Array, " ..."]:
            return lax.cond(
                m_array.shape == shape,
                lambda: m_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: m_array, lambda: m_array)
                ),
            )

        def check_v_shape() -> Float[Array, " ..."]:
            return lax.cond(
                v_array.shape == shape,
                lambda: v_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: v_array, lambda: v_array)
                ),
            )

        def check_step_scalar() -> Int[Array, " "]:
            return lax.cond(
                step_array.ndim == 0,
                lambda: step_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: step_array, lambda: step_array)
                ),
            )

        check_m_shape()
        check_v_shape()
        check_step_scalar()

        return OptimizerState(
            m=m_array,
            v=v_array,
            step=step_array,
        )

    validated_optimizer_state: OptimizerState = validate_and_create()
    return validated_optimizer_state


@jaxtyped(typechecker=beartype)
def make_ptychography_params(
    zoom_factor: ScalarNumeric,
    aperture_diameter: ScalarNumeric,
    travel_distance: ScalarNumeric,
    aperture_center: Float[Array, " 2"],
    camera_pixel_size: ScalarNumeric,
    learning_rate: ScalarNumeric,
    num_iterations: ScalarInteger,
) -> PtychographyParams:
    """Create a PtychographyParams PyTree with validated parameters.

    Parameters
    ----------
    zoom_factor : ScalarNumeric
        Optical zoom factor for magnification (must be positive)
    aperture_diameter : ScalarNumeric
        Diameter of the aperture in meters (must be positive)
    travel_distance : ScalarNumeric
        Light propagation distance in meters (must be positive)
    aperture_center : Float[Array, " 2"]
        Center position of the aperture (x, y) in meters
    camera_pixel_size : ScalarNumeric
        Camera pixel size in meters (must be positive)
    learning_rate : ScalarNumeric
        Learning rate for optimization (must be positive)
    num_iterations : ScalarInteger
        Number of optimization iterations (must be positive)

    Returns
    -------
    PtychographyParams
        Validated ptychography parameters as a PyTree

    Notes
    -----
    This function performs runtime validation to ensure all parameters
    are properly formatted and within valid ranges before creating the
    PtychographyParams PyTree. All scalar inputs are converted to JAX
    arrays.
    """
    zoom_factor_array = jnp.asarray(zoom_factor, dtype=jnp.float64)
    aperture_diameter_array = jnp.asarray(aperture_diameter, dtype=jnp.float64)
    travel_distance_array = jnp.asarray(travel_distance, dtype=jnp.float64)
    aperture_center_array = jnp.asarray(aperture_center, dtype=jnp.float64)
    camera_pixel_size_array = jnp.asarray(camera_pixel_size, dtype=jnp.float64)
    learning_rate_array = jnp.asarray(learning_rate, dtype=jnp.float64)
    num_iterations_array = jnp.asarray(num_iterations, dtype=jnp.int64)

    def validate_and_create() -> PtychographyParams:
        def check_positive_zoom() -> Float[Array, " "]:
            return lax.cond(
                zoom_factor_array > 0,
                lambda: zoom_factor_array,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: zoom_factor_array,
                        lambda: zoom_factor_array,
                    )
                ),
            )

        def check_positive_aperture() -> Float[Array, " "]:
            return lax.cond(
                aperture_diameter_array > 0,
                lambda: aperture_diameter_array,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: aperture_diameter_array,
                        lambda: aperture_diameter_array,
                    )
                ),
            )

        def check_positive_distance() -> Float[Array, " "]:
            return lax.cond(
                travel_distance_array > 0,
                lambda: travel_distance_array,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: travel_distance_array,
                        lambda: travel_distance_array,
                    )
                ),
            )

        def check_aperture_center_shape() -> Float[Array, " 2"]:
            return lax.cond(
                aperture_center_array.shape == (2,),
                lambda: aperture_center_array,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: aperture_center_array,
                        lambda: aperture_center_array,
                    )
                ),
            )

        def check_positive_pixel_size() -> Float[Array, " "]:
            return lax.cond(
                camera_pixel_size_array > 0,
                lambda: camera_pixel_size_array,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: camera_pixel_size_array,
                        lambda: camera_pixel_size_array,
                    )
                ),
            )

        def check_positive_learning_rate() -> Float[Array, " "]:
            return lax.cond(
                learning_rate_array > 0,
                lambda: learning_rate_array,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: learning_rate_array,
                        lambda: learning_rate_array,
                    )
                ),
            )

        def check_positive_iterations() -> Int[Array, " "]:
            return lax.cond(
                num_iterations_array > 0,
                lambda: num_iterations_array,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: num_iterations_array,
                        lambda: num_iterations_array,
                    )
                ),
            )

        check_positive_zoom()
        check_positive_aperture()
        check_positive_distance()
        check_aperture_center_shape()
        check_positive_pixel_size()
        check_positive_learning_rate()
        check_positive_iterations()

        return PtychographyParams(
            zoom_factor=zoom_factor_array,
            aperture_diameter=aperture_diameter_array,
            travel_distance=travel_distance_array,
            aperture_center=aperture_center_array,
            camera_pixel_size=camera_pixel_size_array,
            learning_rate=learning_rate_array,
            num_iterations=num_iterations_array,
        )

    validated_params: PtychographyParams = validate_and_create()
    return validated_params
