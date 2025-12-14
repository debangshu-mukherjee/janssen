"""Defined type aliases and PyTrees.

Extended Summary
----------------
Data structures and type definitions for optical microscopy.

Routine Listings
----------------
NonJaxNumber : TypeAlias
    A type alias for int, float or complex
ScalarBool : TypeAlias
    A type alias for bool or Bool[Array, " "]
ScalarComplex : TypeAlias
    A type alias for complex or Complex[Array, " "]
ScalarFloat : TypeAlias
    A type alias for float or Float[Array, " "]
ScalarInteger : TypeAlias
    A type alias for int or Int[Array, " "]
ScalarNumeric : TypeAlias
    A type alias for int, float, complex or Num[Array, " "]
LensParams : PyTree
    A named tuple for lens parameters
GridParams : PyTree
    A named tuple for computational grid parameters
OpticalWavefront : PyTree
    A named tuple for representing an optical wavefront
PropagatingWavefront : PyTree
    A named tuple for representing a propagating optical wavefront
MicroscopeData : PyTree
    A named tuple for storing 3D or 4D microscope image data
SampleFunction : PyTree
    A named tuple for representing a sample function
SlicedMaterialFunction : PyTree
    A named tuple for representing a 3D sliced material with complex
    refractive index.
Diffractogram : PyTree
    A named tuple for storing a single diffraction pattern
OptimizerState : PyTree
    A PyTree for maintaining optimizer state (moments and step count)
PtychographyParams : PyTree
    A PyTree for ptychography reconstruction parameters

Notes
-----
Always use these factory functions instead of directly instantiating the
NamedTuple classes to ensure proper runtime type checking of the
contents.
"""

import jax
from beartype.typing import NamedTuple, Optional, Tuple, TypeAlias, Union
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, Int, Num

jax.config.update("jax_enable_x64", True)

NonJaxNumber: TypeAlias = Union[int, float, complex]
ScalarBool: TypeAlias = Union[bool, Bool[Array, " "]]
ScalarComplex: TypeAlias = Union[complex, Complex[Array, " "]]
ScalarFloat: TypeAlias = Union[float, Float[Array, " "]]
ScalarInteger: TypeAlias = Union[int, Int[Array, " "]]
ScalarNumeric: TypeAlias = Union[int, float, complex, Num[Array, " "]]


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
        Whether the field is polarized (True for 3D field, False for 2D
        field).
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


@register_pytree_node_class
class PropagatingWavefront(NamedTuple):
    """PyTree structure for representing an propagating optical wavefront.

    Attributes
    ----------
    field : Union[Complex[Array, "zz hh ww"], Complex[Array, "zz hh ww 2"]]
        Complex amplitude of the optical field. Can be scalar (Z, H, W) or
        polarized with two components (Z, H, W, 2). Z represents the slices
        along the propagation direction.
    wavelength : Float[Array, " "]
        Wavelength of the optical wavefront in meters.
    dx : Float[Array, " "]
        Spatial sampling interval (grid spacing) in meters.
    z_positions : Float[Array, " zz"]
        Axial positions of the wavefront along the propagation direction.
        In meters.
    polarization : Bool[Array, " "]
        Whether the field is polarized (True for 3D field, False for 2D
        field).
    """

    field: Union[Complex[Array, " zz hh ww"], Complex[Array, " zz hh ww 2"]]
    wavelength: Float[Array, " "]
    dx: Float[Array, " "]
    z_positions: Float[Array, " zz"]
    polarization: Bool[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Union[Complex[Array, " zz hh ww"], Complex[Array, " zz hh ww 2"]],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " zz"],
            Bool[Array, " "],
        ],
        None,
    ]:
        """Flatten the PropagatingWavefront into a tuple of its components."""
        return (
            (
                self.field,
                self.wavelength,
                self.dx,
                self.z_positions,
                self.polarization,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Union[Complex[Array, " zz hh ww"], Complex[Array, " zz hh ww 2"]],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " zz"],
            Bool[Array, " "],
        ],
    ) -> "PropagatingWavefront":
        """Unflatten the PropagatingWavefront from tuple of its components."""
        return cls(*children)


@register_pytree_node_class
class MicroscopeData(NamedTuple):
    """PyTree structure for representing an 3D or 4D microscope image.

    Attributes
    ----------
    image_data :
        Float[Array, " pp hh ww"] | Float[Array, " xx yy hh ww"]
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


@register_pytree_node_class
class SlicedMaterialFunction(NamedTuple):
    """PyTree structure for representing a 3D sliced material.

    Attributes
    ----------
    material : Complex[Array, " hh ww zz"]
        Complex refractive index for each slice. The real part represents
        the refractive index n, and the imaginary part represents the
        extinction coefficient κ (absorption).
    dx : Float[Array, " "]
        Spatial sampling interval (pixel spacing) within each slice in meters.
    tz : Float[Array, " "]
        Interslice distance (spacing between slices) in the z-direction in
        meters.

    Notes
    -----
    This structure represents a 3D material where:
    - material[i, j, k] contains the complex refractive index ñ = n + iκ at
      pixel (i,j) in slice k
    - The real part n determines phase delay: φ = (2π/λ) * (n-1) * t
    - The imaginary part κ determines absorption: A = exp(-4πκt/λ)
    - dx is the pixel spacing in the x-y plane
    - tz is the spacing between slices in the z direction
    """

    material: Complex[Array, " hh ww zz"]
    dx: Float[Array, " "]
    tz: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Complex[Array, " hh ww zz"], Float[Array, " "], Float[Array, " "]
        ],
        None,
    ]:
        """Flatten the SlicedMaterialFunction into tuple of its components."""
        return (
            (
                self.material,
                self.dx,
                self.tz,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Complex[Array, " hh ww zz"], Float[Array, " "], Float[Array, " "]
        ],
    ) -> "SlicedMaterialFunction":
        """Unflatten the SlicedMaterialFunction from tuple of components."""
        return cls(*children)


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


@register_pytree_node_class
class PtychographyParams(NamedTuple):
    """PyTree structure for ptychography reconstruction parameters.

    Attributes
    ----------
    zoom_factor : Float[Array, " "]
        Optical zoom factor for magnification
    aperture_diameter : Float[Array, " "]
        Diameter of the aperture in meters
    travel_distance : Float[Array, " "]
        Light propagation distance in meters
    aperture_center : Float[Array, " 2"]
        Center position of the aperture (x, y) in meters
    camera_pixel_size : Float[Array, " "]
        Camera pixel size in meters (typically fixed)
    learning_rate : Float[Array, " "]
        Learning rate for optimization
    num_iterations : Int[Array, " "]
        Number of optimization iterations

    Notes
    -----
    This class encapsulates all the optical and optimization parameters
    used in ptychographic reconstruction. It is registered as a PyTree
    node to enable JAX transformations and gradient-based optimization
    of these parameters.
    """

    zoom_factor: Float[Array, " "]
    aperture_diameter: Float[Array, " "]
    travel_distance: Float[Array, " "]
    aperture_center: Float[Array, " 2"]
    camera_pixel_size: Float[Array, " "]
    learning_rate: Float[Array, " "]
    num_iterations: Int[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " 2"],
            Float[Array, " "],
            Float[Array, " "],
            Int[Array, " "],
        ],
        None,
    ]:
        """Flatten the PtychographyParams into a tuple of its components."""
        return (
            (
                self.zoom_factor,
                self.aperture_diameter,
                self.travel_distance,
                self.aperture_center,
                self.camera_pixel_size,
                self.learning_rate,
                self.num_iterations,
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
            Float[Array, " 2"],
            Float[Array, " "],
            Float[Array, " "],
            Int[Array, " "],
        ],
    ) -> "PtychographyParams":
        """Unflatten the PtychographyParams from a tuple of its components."""
        return cls(*children)


@register_pytree_node_class
class PtychographyReconstruction(NamedTuple):
    """PyTree structure for ptychography reconstruction results.

    Attributes
    ----------
    sample : SampleFunction
        Final reconstructed sample covering the scanned FOV
    lightwave : OpticalWavefront
        Final reconstructed probe/lightwave
    zoom_factor : Float[Array, " "]
        Final optimized zoom factor
    aperture_diameter : Float[Array, " "]
        Final optimized aperture diameter in meters
    aperture_center : Optional[Float[Array, " 2"]]
        Final optimized aperture center position (x, y)
    travel_distance : Float[Array, " "]
        Final optimized light propagation distance in meters
    intermediate_samples : Complex[Array, " H W S"]
        Intermediate sample reconstructions during optimization
    intermediate_lightwaves : Complex[Array, " H W S"]
        Intermediate probe reconstructions during optimization
    intermediate_zoom_factors : Float[Array, " S"]
        Intermediate zoom factors during optimization
    intermediate_aperture_diameters : Float[Array, " S"]
        Intermediate aperture diameters during optimization
    intermediate_aperture_centers : Float[Array, " 2 S"]
        Intermediate aperture centers during optimization
    intermediate_travel_distances : Float[Array, " S"]
        Intermediate travel distances during optimization

    Notes
    -----
    This class encapsulates all results from ptychographic reconstruction,
    including both final optimized values and intermediate results saved
    during the optimization process. It is registered as a PyTree node
    to enable JAX transformations.
    """

    sample: "SampleFunction"
    lightwave: "OpticalWavefront"
    zoom_factor: Float[Array, " "]
    aperture_diameter: Float[Array, " "]
    aperture_center: Optional[Float[Array, " 2"]]
    travel_distance: Float[Array, " "]
    intermediate_samples: Complex[Array, " H W S"]
    intermediate_lightwaves: Complex[Array, " H W S"]
    intermediate_zoom_factors: Float[Array, " S"]
    intermediate_aperture_diameters: Float[Array, " S"]
    intermediate_aperture_centers: Float[Array, " 2 S"]
    intermediate_travel_distances: Float[Array, " S"]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            "SampleFunction",
            "OpticalWavefront",
            Float[Array, " "],
            Float[Array, " "],
            Optional[Float[Array, " 2"]],
            Float[Array, " "],
            Complex[Array, " H W S"],
            Complex[Array, " H W S"],
            Float[Array, " S"],
            Float[Array, " S"],
            Float[Array, " 2 S"],
            Float[Array, " S"],
        ],
        None,
    ]:
        """Flatten the PtychographyReconstruction into tuple of components."""
        return (
            (
                self.sample,
                self.lightwave,
                self.zoom_factor,
                self.aperture_diameter,
                self.aperture_center,
                self.travel_distance,
                self.intermediate_samples,
                self.intermediate_lightwaves,
                self.intermediate_zoom_factors,
                self.intermediate_aperture_diameters,
                self.intermediate_aperture_centers,
                self.intermediate_travel_distances,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            "SampleFunction",
            "OpticalWavefront",
            Float[Array, " "],
            Float[Array, " "],
            Optional[Float[Array, " 2"]],
            Float[Array, " "],
            Complex[Array, " H W S"],
            Complex[Array, " H W S"],
            Float[Array, " S"],
            Float[Array, " S"],
            Float[Array, " 2 S"],
            Float[Array, " S"],
        ],
    ) -> "PtychographyReconstruction":
        """Unflatten PtychographyReconstruction from tuple of components."""
        return cls(*children)
