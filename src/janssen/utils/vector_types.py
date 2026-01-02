"""Vector wavefront types for full electromagnetic field representation.

Extended Summary
----------------
This module provides PyTree data structures for representing full vector
electromagnetic fields with Ex, Ey, and Ez components. This is essential
for accurate modeling of high-NA focusing, tight focusing of polarized
beams, and any situation where the paraxial approximation breaks down.

The standard OpticalWavefront with (H, W, 2) Jones representation assumes
paraxial propagation where Ez ≈ 0. At high numerical apertures (NA > 0.7),
the longitudinal Ez component becomes significant and cannot be neglected.

Routine Listings
----------------
VectorWavefront3D : NamedTuple
    PyTree structure for full 3-component electric field
make_vector_wavefront_3d : function
    Factory function to create validated VectorWavefront3D instances
jones_to_vector3d : function
    Convert a 2-component Jones field to a 3-component vector field
vector3d_to_jones : function
    Extract the transverse (Ex, Ey) components as a Jones field

Notes
-----
The VectorWavefront3D structure is designed for:
- High-NA focusing simulations (Richards-Wolf integrals)
- Tight focusing of radially/azimuthally polarized beams
- Near-field optics where Ez is significant
- Validation against scalar approximations

The Ez component is typically zero in the input pupil plane and develops
during propagation through high-NA optical systems.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax import lax
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, jaxtyped

from .types import ScalarNumeric


@register_pytree_node_class
class VectorWavefront3D(NamedTuple):
    """PyTree structure for full 3-component vector electric field.

    Attributes
    ----------
    field : Complex[Array, " hh ww 3"]
        Complex amplitude of the full electric field vector.
        Components are ordered as [Ex, Ey, Ez] along the last dimension.
    wavelength : Float[Array, " "]
        Wavelength of the optical wavefront in meters.
    dx : Float[Array, " "]
        Spatial sampling interval (grid spacing) in meters.
    z_position : Float[Array, " "]
        Axial position of the wavefront along the propagation direction.

    Notes
    -----
    Unlike the standard OpticalWavefront which uses Jones vectors (Ex, Ey),
    this structure carries the full 3D electric field vector. The Ez
    component is essential for:

    - High-NA focusing (NA > 0.7)
    - Radially polarized beam focusing (creates strong Ez at focus)
    - Near-field and evanescent wave modeling
    - Accurate energy density calculations

    The field components are defined in a Cartesian coordinate system
    where z is the optical axis (propagation direction).
    """

    field: Complex[Array, " hh ww 3"]
    wavelength: Float[Array, " "]
    dx: Float[Array, " "]
    z_position: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Complex[Array, " hh ww 3"],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
        ],
        None,
    ]:
        """Flatten the VectorWavefront3D into a tuple of its components."""
        return (
            (
                self.field,
                self.wavelength,
                self.dx,
                self.z_position,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Complex[Array, " hh ww 3"],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
        ],
    ) -> "VectorWavefront3D":
        """Unflatten the VectorWavefront3D from a tuple of its components."""
        return cls(*children)

    @property
    def ex(self) -> Complex[Array, " hh ww"]:
        """Return the Ex (x-polarization) component."""
        return self.field[..., 0]

    @property
    def ey(self) -> Complex[Array, " hh ww"]:
        """Return the Ey (y-polarization) component."""
        return self.field[..., 1]

    @property
    def ez(self) -> Complex[Array, " hh ww"]:
        """Return the Ez (longitudinal) component."""
        return self.field[..., 2]

    @property
    def intensity(self) -> Float[Array, " hh ww"]:
        """Return the total intensity |Ex|² + |Ey|² + |Ez|²."""
        return (
            jnp.abs(self.field[..., 0]) ** 2
            + jnp.abs(self.field[..., 1]) ** 2
            + jnp.abs(self.field[..., 2]) ** 2
        )

    @property
    def intensity_transverse(self) -> Float[Array, " hh ww"]:
        """Return the transverse intensity |Ex|² + |Ey|²."""
        return (
            jnp.abs(self.field[..., 0]) ** 2 + jnp.abs(self.field[..., 1]) ** 2
        )

    @property
    def intensity_longitudinal(self) -> Float[Array, " hh ww"]:
        """Return the longitudinal intensity |Ez|²."""
        return jnp.abs(self.field[..., 2]) ** 2


@jaxtyped(typechecker=beartype)
def make_vector_wavefront_3d(
    field: Complex[Array, " hh ww 3"],
    wavelength: ScalarNumeric,
    dx: ScalarNumeric,
    z_position: ScalarNumeric,
) -> VectorWavefront3D:
    """Create a validated VectorWavefront3D instance.

    Factory function that validates inputs and creates a VectorWavefront3D
    PyTree suitable for high-NA vector optics simulations.

    Parameters
    ----------
    field : Complex[Array, " hh ww 3"]
        Complex amplitude of the full electric field vector.
        Shape must be (H, W, 3) where the last dimension contains
        [Ex, Ey, Ez] components.
    wavelength : ScalarNumeric
        Wavelength of the optical wavefront in meters.
    dx : ScalarNumeric
        Spatial sampling interval (grid spacing) in meters.
    z_position : ScalarNumeric
        Axial position of the wavefront along the propagation direction
        in meters.

    Returns
    -------
    VectorWavefront3D
        Validated vector wavefront instance.
    """
    vector_dim: int = 3
    expected_ndim: int = 3

    field = jnp.asarray(field, dtype=jnp.complex128)
    wavelength_arr: Float[Array, " "] = jnp.asarray(
        wavelength, dtype=jnp.float64
    )
    dx_arr: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    z_position_arr: Float[Array, " "] = jnp.asarray(
        z_position, dtype=jnp.float64
    )

    def validate_and_create() -> VectorWavefront3D:
        def check_field_dimensions() -> Complex[Array, " hh ww 3"]:
            is_valid_ndim: Bool[Array, " "] = field.ndim == expected_ndim
            is_valid_vector_dim: Bool[Array, " "] = (
                field.shape[-1] == vector_dim
            )
            is_valid: Bool[Array, " "] = jnp.logical_and(
                is_valid_ndim, is_valid_vector_dim
            )
            return lax.cond(
                is_valid,
                lambda: field,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: field, lambda: field)
                ),
            )

        def check_field_finite(
            f: Complex[Array, " hh ww 3"],
        ) -> Complex[Array, " hh ww 3"]:
            is_finite: Bool[Array, " "] = jnp.all(jnp.isfinite(f))
            return lax.cond(
                is_finite,
                lambda: f,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: f, lambda: f)
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

        def check_z_position() -> Float[Array, " "]:
            return lax.cond(
                jnp.isfinite(z_position_arr),
                lambda: z_position_arr,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: z_position_arr, lambda: z_position_arr
                    )
                ),
            )

        validated_field: Complex[Array, " hh ww 3"] = check_field_finite(
            check_field_dimensions()
        )
        validated_wavelength: Float[Array, " "] = check_wavelength()
        validated_dx: Float[Array, " "] = check_dx()
        validated_z_position: Float[Array, " "] = check_z_position()

        return VectorWavefront3D(
            field=validated_field,
            wavelength=validated_wavelength,
            dx=validated_dx,
            z_position=validated_z_position,
        )

    return validate_and_create()


@jaxtyped(typechecker=beartype)
def jones_to_vector3d(
    jones_field: Complex[Array, " hh ww 2"],
    wavelength: ScalarNumeric,
    dx: ScalarNumeric,
    z_position: ScalarNumeric,
) -> VectorWavefront3D:
    """Convert a 2-component Jones field to a 3-component vector field.

    This conversion assumes Ez = 0, which is valid for paraxial beams
    or fields in the pupil plane before high-NA focusing.

    Parameters
    ----------
    jones_field : Complex[Array, " hh ww 2"]
        Jones vector field with [Ex, Ey] components.
    wavelength : ScalarNumeric
        Wavelength in meters.
    dx : ScalarNumeric
        Pixel spacing in meters.
    z_position : ScalarNumeric
        Axial position in meters.

    Returns
    -------
    VectorWavefront3D
        Vector wavefront with Ez = 0.

    Notes
    -----
    This is useful for preparing input beams for high-NA focusing
    simulations. The Ez component will be generated during the
    Richards-Wolf integration.
    """
    ez: Complex[Array, " hh ww"] = jnp.zeros(
        jones_field.shape[:-1], dtype=jnp.complex128
    )
    field_3d: Complex[Array, " hh ww 3"] = jnp.concatenate(
        [jones_field, ez[..., jnp.newaxis]], axis=-1
    )
    return make_vector_wavefront_3d(
        field=field_3d,
        wavelength=wavelength,
        dx=dx,
        z_position=z_position,
    )


@jaxtyped(typechecker=beartype)
def vector3d_to_jones(
    wavefront: VectorWavefront3D,
) -> Complex[Array, " hh ww 2"]:
    """Extract the transverse (Ex, Ey) components as a Jones field.

    Parameters
    ----------
    wavefront : VectorWavefront3D
        Input vector wavefront.

    Returns
    -------
    Complex[Array, " hh ww 2"]
        Jones vector field [Ex, Ey], discarding Ez.

    Notes
    -----
    This discards the longitudinal component. Use with caution when
    Ez is significant.
    """
    jones_field: Complex[Array, " hh ww 2"] = wavefront.field[..., :2]
    return jones_field
