"""Defined type aliases and PyTrees.

Extended Summary
----------------
Data structures and type definitions for optical microscopy.

Routine Listings
----------------
non_jax_number : TypeAlias
    A type alias for int, float or complex
scalar_bool : TypeAlias
    A type alias for bool or Bool[Array, " "]
scalar_complex : TypeAlias
    A type alias for complex or Complex[Array, " "]
scalar_float : TypeAlias
    A type alias for float or Float[Array, " "]
scalar_integer : TypeAlias
    A type alias for int or Int[Array, " "]
scalar_numeric : TypeAlias
    A type alias for int, float, complex or Num[Array, " "]
LensParams : PyTree
    A named tuple for lens parameters
GridParams : PyTree
    A named tuple for computational grid parameters
OpticalWavefront : PyTree
    A named tuple for representing an optical wavefront
MicroscopeData : PyTree
    A named tuple for storing 3D or 4D microscope image data
SampleFunction : PyTree
    A named tuple for representing a sample function
Diffractogram : PyTree
    A named tuple for storing a single diffraction pattern
OptimizerState : PyTree
    A PyTree for maintaining optimizer state (moments and step count)

Notes
-----
Always use these factory functions instead of directly instantiating the
NamedTuple classes to ensure proper runtime type checking of the contents.
"""

import jax
from beartype import beartype
from beartype.typing import NamedTuple, Tuple, TypeAlias, Union
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, Int, Num, jaxtyped

jax.config.update("jax_enable_x64", True)

non_jax_number: TypeAlias = Union[int, float, complex]
scalar_bool: TypeAlias = Union[bool, Bool[Array, " "]]
scalar_complex: TypeAlias = Union[complex, Complex[Array, " "]]
scalar_float: TypeAlias = Union[float, Float[Array, " "]]
scalar_integer: TypeAlias = Union[int, Int[Array, " "]]
scalar_numeric: TypeAlias = Union[int, float, complex, Num[Array, " "]]


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class LensParams(NamedTuple):
    """PyTree structure for lens parameters.

    Attributes
    ----------
    focal_length : Float[Array, " "]
        Focal length of the lens in meters
    diameter : Float[Array, " "]
        Diameter of the lens in meters
    n : Float[Array, " "]
        Refractive index of the lens material
    center_thickness : Float[Array, " "]
        Thickness at the center of the lens in meters
    r1 : Float[Array, " "]
        Radius of curvature of the first surface in meters
        (positive for convex)
    r2 : Float[Array, " "]
        Radius of curvature of the second surface in meters (
        positive for convex)
    """

    focal_length: Float[Array, " "]
    diameter: Float[Array, " "]
    n: Float[Array, " "]
    center_thickness: Float[Array, " "]
    r1: Float[Array, " "]
    r2: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
        ],
        None,
    ]:
        """Flatten the LensParams into a tuple of its components."""
        return (
            (
                self.focal_length,
                self.diameter,
                self.n,
                self.center_thickness,
                self.r1,
                self.r2,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
        ],
    ) -> "LensParams":
        """Unflatten the LensParams from a tuple of its components."""
        return cls(*children)


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class GridParams(NamedTuple):
    """PyTree structure for computational grid parameters.

    Attributes
    ----------
    xx : Float[Array, " hh ww"]
        Spatial grid in the x-direction
    yy : Float[Array, " hh ww"]
        Spatial grid in the y-direction
    phase_profile : Float[Array, " hh ww"]
        Phase profile of the optical field
    transmission : Float[Array, " hh ww"]
        Transmission profile of the optical field

    Notes
    -----
    This class is registered as a PyTree node, making it
    compatible with JAX transformations like jit, grad, and vmap.
    The auxiliary data in tree_flatten is None as all relevant
    data is stored in JAX arrays.
    """

    xx: Float[Array, " hh ww"]
    yy: Float[Array, " hh ww"]
    phase_profile: Float[Array, " hh ww"]
    transmission: Float[Array, " hh ww"]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, " hh ww"],
            Float[Array, " hh ww"],
            Float[Array, " hh ww"],
            Float[Array, " hh ww"],
        ],
        None,
    ]:
        """Flatten the GridParams into a tuple of its components."""
        return (
            (
                self.xx,
                self.yy,
                self.phase_profile,
                self.transmission,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Float[Array, " hh ww"],
            Float[Array, " hh ww"],
            Float[Array, " hh ww"],
            Float[Array, " hh ww"],
        ],
    ) -> "GridParams":
        """Unflatten the GridParams from a tuple of its components."""
        return cls(*children)


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class OpticalWavefront(NamedTuple):
    """PyTree structure for representing an optical wavefront.

    Attributes
    ----------
    field : Union[Complex[Array, " hh ww"], Complex[Array, " hh ww 2"]]
        Complex amplitude of the optical field. Can be scalar (H, W) or
        polarized with two components (H, W, 2).
    wavelength : Float[Array, " "]
        Wavelength of the optical wavefront in meters.
    dx : Float[Array, " "]
        Spatial sampling interval (grid spacing) in meters.
    z_position : Float[Array, " "]
        Axial position of the wavefront along the propagation direction.
        In meters.
    polarization : Bool[Array, " "]
        Whether the field is polarized (True for 3D field, False for 2D field).
    """

    field: Union[Complex[Array, " hh ww"], Complex[Array, " hh ww 2"]]
    wavelength: Float[Array, " "]
    dx: Float[Array, " "]
    z_position: Float[Array, " "]
    polarization: Bool[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Union[Complex[Array, " hh ww"], Complex[Array, " hh ww 2"]],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Bool[Array, " "],
        ],
        None,
    ]:
        """Flatten the OpticalWavefront into a tuple of its components."""
        return (
            (
                self.field,
                self.wavelength,
                self.dx,
                self.z_position,
                self.polarization,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Union[Complex[Array, " hh ww"], Complex[Array, " hh ww 2"]],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Bool[Array, " "],
        ],
    ) -> "OpticalWavefront":
        """Unflatten the OpticalWavefront from a tuple of its components."""
        return cls(*children)


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class MicroscopeData(NamedTuple):
    """PyTree structure for representing an 3D or 4D microscope image.

    Attributes
    ----------
    image_data : Float[Array, " pp hh ww"] | Float[Array, " xx yy hh ww"]
        3D or 4D image data representing the optical field.
    positions : Num[Array, " pp 2"]
        Positions of the images during collection.
    wavelength : Float[Array, " "]
        Wavelength of the optical wavefront in meters.
    dx : Float[Array, " "]
        Spatial sampling interval (grid spacing) in meters.
    """

    image_data: Union[Float[Array, " pp hh ww"], Float[Array, " xx yy hh ww"]]
    positions: Num[Array, " pp 2"]
    wavelength: Float[Array, " "]
    dx: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Union[Float[Array, " pp hh ww"], Float[Array, " xx yy hh ww"]],
            Num[Array, " pp 2"],
            Float[Array, " "],
            Float[Array, " "],
        ],
        None,
    ]:
        """Flatten the MicroscopeData into a tuple of its components."""
        return (
            (
                self.image_data,
                self.positions,
                self.wavelength,
                self.dx,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Union[Float[Array, " pp hh ww"], Float[Array, " xx yy hh ww"]],
            Num[Array, " pp 2"],
            Float[Array, " "],
            Float[Array, " "],
        ],
    ) -> "MicroscopeData":
        """Unflatten the MicroscopeData from a tuple of its components."""
        return cls(*children)


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class SampleFunction(NamedTuple):
    """PyTree structure for representing a sample function.

    Attributes
    ----------
    sample : Complex[Array, " hh ww"]
        The sample function.
    dx : Float[Array, " "]
        Spatial sampling interval (grid spacing) in meters.
    """

    sample: Complex[Array, " hh ww"]
    dx: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Complex[Array, " hh ww"], Float[Array, " "]], None]:
        """Flatten the SampleFunction into a tuple of its components."""
        return (
            (
                self.sample,
                self.dx,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[Complex[Array, " hh ww"], Float[Array, " "]],
    ) -> "SampleFunction":
        """Unflatten the SampleFunction from a tuple of its components."""
        return cls(*children)


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class Diffractogram(NamedTuple):
    """PyTree structure for representing a single diffractogram.

    Attributes
    ----------
    image : Float[Array, " hh ww"]
        Image data.
    wavelength : Float[Array, " "]
        Wavelength of the optical wavefront in meters.
    dx : Float[Array, " "]
        Spatial sampling interval (grid spacing) in meters.
    """

    image: Float[Array, " hh ww"]
    wavelength: Float[Array, " "]
    dx: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Float[Array, " hh ww"], Float[Array, " "], Float[Array, " "]],
        None,
    ]:
        """Flatten the Diffractogram into a tuple of its components."""
        return (
            (
                self.image,
                self.wavelength,
                self.dx,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Float[Array, " hh ww"], Float[Array, " "], Float[Array, " "]
        ],
    ) -> "Diffractogram":
        """Unflatten the Diffractogram from a tuple of its components."""
        return cls(*children)


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class OptimizerState(NamedTuple):
    """PyTree structure for maintaining optimizer state.

    Attributes
    ----------
    m : Complex[Array, "..."]
        First moment estimate (for Adam-like optimizers)
    v : Float[Array, "..."]
        Second moment estimate (for Adam-like optimizers)
    step : Int[Array, " "]
        Step count
    """

    m: Complex[Array, "..."]
    v: Float[Array, "..."]
    step: Int[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[Complex[Array, "..."], Float[Array, "..."], Int[Array, " "]],
        None,
    ]:
        """Flatten the OptimizerState into a tuple of its components."""
        return (
            (
                self.m,
                self.v,
                self.step,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Complex[Array, "..."], Float[Array, "..."], Int[Array, " "]
        ],
    ) -> "OptimizerState":
        """Unflatten the OptimizerState from a tuple of its components."""
        return cls(*children)
