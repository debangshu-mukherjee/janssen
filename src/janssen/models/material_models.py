"""3D material models for optical simulations.

Extended Summary
----------------
Pure JAX implementations of common 3D material structures with spatially-
varying complex refractive indices. All functions use JAX operations and
support automatic differentiation.

Routine Listings
----------------
uniform_material : function
    Create uniform 3D material with constant refractive index
spherical_inclusion : function
    Create material with spherical inclusion
layered_material : function
    Create alternating layers of materials
biological_cell : function
    Create biological cell model with nucleus
gradient_index_material : function
    Create gradient-index (GRIN) material with radial profile

Notes
-----
All functions return SlicedMaterialFunction PyTrees and use pure JAX
operations for compatibility with jit, grad, and vmap.

The complex refractive index convention is:
    ñ = n + iκ
where n is the real refractive index and κ is the extinction coefficient.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from janssen.types import (
    ScalarComplex,
    ScalarFloat,
    ScalarInteger,
    SlicedMaterialFunction,
    make_sliced_material_function,
)


@jaxtyped(typechecker=beartype)
def uniform_material(
    shape: Tuple[int, int, int],
    n: ScalarComplex,
    dx: ScalarFloat,
    tz: ScalarFloat,
) -> SlicedMaterialFunction:
    """Create uniform 3D material with constant refractive index.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Shape of material (height, width, num_slices)
    n : ScalarComplex
        Complex refractive index (n_real + i*kappa)
    dx : ScalarFloat
        Pixel spacing in x-y plane in meters
    tz : ScalarFloat
        Slice spacing in z direction in meters

    Returns
    -------
    material : SlicedMaterialFunction
        Uniform material PyTree

    Notes
    -----
    Algorithm:
    - Create array filled with constant refractive index
    - Convert to SlicedMaterialFunction PyTree
    - Return material

    Implementation Details:
    - Uses jnp.ones for efficient array creation
    - Multiplies by complex scalar to set refractive index
    - All operations are JAX-compatible and differentiable

    Examples
    --------
    Create uniform glass material:

    >>> glass = uniform_material(
    ...     shape=(128, 128, 20),
    ...     n=1.52 + 0.0j,
    ...     dx=1e-6,
    ...     tz=5e-6
    ... )
    """
    height: int
    width: int
    num_slices: int
    height, width, num_slices = shape

    n_value: Complex[Array, " "] = jnp.asarray(n, dtype=jnp.complex128)

    material_array: Complex[Array, " h w z"] = (
        jnp.ones((height, width, num_slices), dtype=jnp.complex128) * n_value
    )

    material: SlicedMaterialFunction = make_sliced_material_function(
        material=material_array,
        dx=dx,
        tz=tz,
    )

    return material


@jaxtyped(typechecker=beartype)
def spherical_inclusion(
    shape: Tuple[int, int, int],
    radius: ScalarFloat,
    n_sphere: ScalarComplex,
    n_background: ScalarComplex,
    dx: ScalarFloat,
    tz: ScalarFloat,
    center: Optional[Tuple[ScalarFloat, ScalarFloat, ScalarFloat]] = None,
) -> SlicedMaterialFunction:
    """Create material with spherical inclusion.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Shape of material (height, width, num_slices)
    radius : ScalarFloat
        Radius of sphere in meters
    n_sphere : ScalarComplex
        Complex refractive index of sphere
    n_background : ScalarComplex
        Complex refractive index of background
    dx : ScalarFloat
        Pixel spacing in x-y plane in meters
    tz : ScalarFloat
        Slice spacing in z direction in meters
    center : Tuple[ScalarFloat, ScalarFloat, ScalarFloat], optional
        Center position (y, x, z) in meters. If None, centered in volume.

    Returns
    -------
    material : SlicedMaterialFunction
        Material with spherical inclusion

    Notes
    -----
    Algorithm:
    - Create coordinate grids for x, y, z
    - Compute radial distance from center
    - Use jnp.where to assign refractive index based on distance
    - Convert to SlicedMaterialFunction PyTree

    Implementation Details:
    - Uses meshgrid with indexing='ij' for consistent ordering
    - If center is None, defaults to volume center
    - Radial distance: r = sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2)
    - Refractive index: n = n_sphere if r < radius else n_background
    - All operations use JAX arrays for automatic differentiation

    Examples
    --------
    Create material with absorbing sphere:

    >>> sphere = spherical_inclusion(
    ...     shape=(128, 128, 20),
    ...     radius=20e-6,
    ...     n_sphere=1.5 + 0.01j,
    ...     n_background=1.0 + 0.0j,
    ...     dx=1e-6,
    ...     tz=5e-6
    ... )
    """
    height: int
    width: int
    num_slices: int
    height, width, num_slices = shape

    y_coords: Float[Array, " h"] = (
        jnp.arange(height, dtype=jnp.float64) - height / 2.0
    ) * dx
    x_coords: Float[Array, " w"] = (
        jnp.arange(width, dtype=jnp.float64) - width / 2.0
    ) * dx
    z_coords: Float[Array, " z"] = (
        jnp.arange(num_slices, dtype=jnp.float64) * tz
    )

    def _get_default_center() -> (
        Tuple[Float[Array, " "], Float[Array, " "], Float[Array, " "]]
    ):
        y0: Float[Array, " "] = jnp.asarray(0.0, dtype=jnp.float64)
        x0: Float[Array, " "] = jnp.asarray(0.0, dtype=jnp.float64)
        z0: Float[Array, " "] = jnp.asarray(
            num_slices * tz / 2.0, dtype=jnp.float64
        )
        return (y0, x0, z0)

    def _get_provided_center() -> (
        Tuple[Float[Array, " "], Float[Array, " "], Float[Array, " "]]
    ):
        y0: Float[Array, " "] = jnp.asarray(center[0], dtype=jnp.float64)
        x0: Float[Array, " "] = jnp.asarray(center[1], dtype=jnp.float64)
        z0: Float[Array, " "] = jnp.asarray(center[2], dtype=jnp.float64)
        return (y0, x0, z0)

    if center is None:
        center_y, center_x, center_z = _get_default_center()
    else:
        center_y, center_x, center_z = _get_provided_center()

    yy: Float[Array, " h w z"]
    xx: Float[Array, " h w z"]
    zz: Float[Array, " h w z"]
    yy, xx, zz = jnp.meshgrid(y_coords, x_coords, z_coords, indexing="ij")

    radial_distance: Float[Array, " h w z"] = jnp.sqrt(
        (yy - center_y) ** 2 + (xx - center_x) ** 2 + (zz - center_z) ** 2
    )

    n_sphere_val: Complex[Array, " "] = jnp.asarray(
        n_sphere, dtype=jnp.complex128
    )
    n_background_val: Complex[Array, " "] = jnp.asarray(
        n_background, dtype=jnp.complex128
    )
    radius_val: Float[Array, " "] = jnp.asarray(radius, dtype=jnp.float64)

    material_array: Complex[Array, " h w z"] = jnp.where(
        radial_distance < radius_val,
        n_sphere_val,
        n_background_val,
    )

    material: SlicedMaterialFunction = make_sliced_material_function(
        material=material_array,
        dx=dx,
        tz=tz,
    )

    return material


@jaxtyped(typechecker=beartype)
def layered_material(
    shape: Tuple[int, int, int],
    n_layers: Tuple[ScalarComplex, ScalarComplex],
    dx: ScalarFloat,
    tz: ScalarFloat,
    slices_per_layer: ScalarInteger = 1,
) -> SlicedMaterialFunction:
    """Create alternating layers of materials.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Shape of material (height, width, num_slices)
    n_layers : Tuple[ScalarComplex, ScalarComplex]
        Refractive indices of two alternating layers (n1, n2)
    dx : ScalarFloat
        Pixel spacing in x-y plane in meters
    tz : ScalarFloat
        Slice spacing in z direction in meters
    slices_per_layer : ScalarInteger, optional
        Number of slices per layer, by default 1

    Returns
    -------
    material : SlicedMaterialFunction
        Layered material PyTree

    Notes
    -----
    Algorithm:
    - Create array to hold material
    - Use vmap over z slices to assign refractive index
    - Layer index = z_idx // slices_per_layer
    - Use modulo to alternate between n1 and n2
    - Convert to SlicedMaterialFunction PyTree

    Implementation Details:
    - Uses jax.lax.cond for layer selection (JAX-compatible)
    - Layer pattern: n1, n2, n1, n2, ...
    - Each layer is slices_per_layer thick
    - All operations are differentiable
    - Uses vmap for efficient parallel processing

    Examples
    --------
    Create optical coating with alternating layers:

    >>> coating = layered_material(
    ...     shape=(128, 128, 20),
    ...     n_layers=(1.38 + 0.0j, 2.35 + 0.0j),
    ...     dx=1e-6,
    ...     tz=50e-9,
    ...     slices_per_layer=2
    ... )
    """
    height: int
    width: int
    num_slices: int
    height, width, num_slices = shape

    n1: Complex[Array, " "] = jnp.asarray(n_layers[0], dtype=jnp.complex128)
    n2: Complex[Array, " "] = jnp.asarray(n_layers[1], dtype=jnp.complex128)
    slices_per: int = int(slices_per_layer)

    def _assign_layer_index(z_idx: int) -> Complex[Array, " h w"]:
        """Assign refractive index for a single z-slice."""
        layer_number: int = z_idx // slices_per
        is_even: bool = (layer_number % 2) == 0

        n_value: Complex[Array, " "] = jax.lax.cond(
            is_even,
            lambda: n1,
            lambda: n2,
        )

        layer_slice: Complex[Array, " h w"] = (
            jnp.ones((height, width), dtype=jnp.complex128) * n_value
        )
        return layer_slice

    material_array: Complex[Array, " h w z"] = jax.vmap(
        _assign_layer_index, out_axes=2
    )(jnp.arange(num_slices))

    material: SlicedMaterialFunction = make_sliced_material_function(
        material=material_array,
        dx=dx,
        tz=tz,
    )

    return material


@jaxtyped(typechecker=beartype)
def biological_cell(
    shape: Tuple[int, int, int],
    cell_radius: ScalarFloat,
    nucleus_radius: ScalarFloat,
    n_cytoplasm: ScalarComplex,
    n_nucleus: ScalarComplex,
    n_medium: ScalarComplex,
    dx: ScalarFloat,
    tz: ScalarFloat,
    center: Optional[Tuple[ScalarFloat, ScalarFloat, ScalarFloat]] = None,
) -> SlicedMaterialFunction:
    """Create biological cell model with nucleus.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Shape of material (height, width, num_slices)
    cell_radius : ScalarFloat
        Outer radius of cell in meters
    nucleus_radius : ScalarFloat
        Radius of nucleus in meters
    n_cytoplasm : ScalarComplex
        Complex refractive index of cytoplasm
    n_nucleus : ScalarComplex
        Complex refractive index of nucleus
    n_medium : ScalarComplex
        Complex refractive index of surrounding medium
    dx : ScalarFloat
        Pixel spacing in x-y plane in meters
    tz : ScalarFloat
        Slice spacing in z direction in meters
    center : Tuple[ScalarFloat, ScalarFloat, ScalarFloat], optional
        Center position (y, x, z) in meters. If None, centered in volume.

    Returns
    -------
    material : SlicedMaterialFunction
        Biological cell material PyTree

    Notes
    -----
    Algorithm:
    - Create coordinate grids for x, y, z
    - Compute radial distance from center
    - Use nested jnp.where to assign refractive index:
        - If r < nucleus_radius: n_nucleus
        - Elif r < cell_radius: n_cytoplasm
        - Else: n_medium
    - Convert to SlicedMaterialFunction PyTree

    Implementation Details:
    - Nested jnp.where for three regions
    - Inner to outer: nucleus, cytoplasm, medium
    - All operations use JAX arrays
    - Typical values:
        - n_medium: 1.337 (water)
        - n_cytoplasm: 1.360-1.380
        - n_nucleus: 1.380-1.400
        - κ increases with density

    Examples
    --------
    Create simple cell model:

    >>> cell = biological_cell(
    ...     shape=(128, 128, 20),
    ...     cell_radius=25e-6,
    ...     nucleus_radius=10e-6,
    ...     n_cytoplasm=1.370 + 0.002j,
    ...     n_nucleus=1.390 + 0.005j,
    ...     n_medium=1.337 + 0.0j,
    ...     dx=1e-6,
    ...     tz=5e-6
    ... )
    """
    height: int
    width: int
    num_slices: int
    height, width, num_slices = shape

    y_coords: Float[Array, " h"] = (
        jnp.arange(height, dtype=jnp.float64) - height / 2.0
    ) * dx
    x_coords: Float[Array, " w"] = (
        jnp.arange(width, dtype=jnp.float64) - width / 2.0
    ) * dx
    z_coords: Float[Array, " z"] = (
        jnp.arange(num_slices, dtype=jnp.float64) * tz
    )

    if center is None:
        center_y: Float[Array, " "] = jnp.asarray(0.0, dtype=jnp.float64)
        center_x: Float[Array, " "] = jnp.asarray(0.0, dtype=jnp.float64)
        center_z: Float[Array, " "] = jnp.asarray(
            num_slices * tz / 2.0, dtype=jnp.float64
        )
    else:
        center_y = jnp.asarray(center[0], dtype=jnp.float64)
        center_x = jnp.asarray(center[1], dtype=jnp.float64)
        center_z = jnp.asarray(center[2], dtype=jnp.float64)

    yy: Float[Array, " h w z"]
    xx: Float[Array, " h w z"]
    zz: Float[Array, " h w z"]
    yy, xx, zz = jnp.meshgrid(y_coords, x_coords, z_coords, indexing="ij")

    radial_distance: Float[Array, " h w z"] = jnp.sqrt(
        (yy - center_y) ** 2 + (xx - center_x) ** 2 + (zz - center_z) ** 2
    )

    n_nucleus_val: Complex[Array, " "] = jnp.asarray(
        n_nucleus, dtype=jnp.complex128
    )
    n_cytoplasm_val: Complex[Array, " "] = jnp.asarray(
        n_cytoplasm, dtype=jnp.complex128
    )
    n_medium_val: Complex[Array, " "] = jnp.asarray(
        n_medium, dtype=jnp.complex128
    )
    nucleus_r: Float[Array, " "] = jnp.asarray(
        nucleus_radius, dtype=jnp.float64
    )
    cell_r: Float[Array, " "] = jnp.asarray(cell_radius, dtype=jnp.float64)

    material_array: Complex[Array, " h w z"] = jnp.where(
        radial_distance < nucleus_r,
        n_nucleus_val,
        jnp.where(
            radial_distance < cell_r,
            n_cytoplasm_val,
            n_medium_val,
        ),
    )

    material: SlicedMaterialFunction = make_sliced_material_function(
        material=material_array,
        dx=dx,
        tz=tz,
    )

    return material


@jaxtyped(typechecker=beartype)
def gradient_index_material(
    shape: Tuple[int, int, int],
    n_center: ScalarFloat,
    gradient_constant: ScalarFloat,
    dx: ScalarFloat,
    tz: ScalarFloat,
    n_min: Optional[ScalarFloat] = None,
) -> SlicedMaterialFunction:
    """Create gradient-index (GRIN) material with radial profile.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Shape of material (height, width, num_slices)
    n_center : ScalarFloat
        Refractive index at center
    gradient_constant : ScalarFloat
        GRIN constant A in n(r) = n0(1 - Ar²/2), units: m^-2
    dx : ScalarFloat
        Pixel spacing in x-y plane in meters
    tz : ScalarFloat
        Slice spacing in z direction in meters
    n_min : ScalarFloat, optional
        Minimum refractive index (for clipping), default is 1.0

    Returns
    -------
    material : SlicedMaterialFunction
        GRIN material PyTree

    Notes
    -----
    Algorithm:
    - Create coordinate grids for x, y
    - Compute radial distance in x-y plane: r_xy = sqrt(x² + y²)
    - Apply GRIN formula: n(r) = n_center × (1 - A × r_xy² / 2)
    - Clip to ensure n >= n_min
    - Broadcast to all z slices (z-invariant)
    - Convert to SlicedMaterialFunction PyTree

    Implementation Details:
    - GRIN profile: n(r) = n₀(1 - Ar²/2)
    - Only depends on x-y position (constant along z)
    - Uses jnp.clip to enforce physical bounds
    - Typical A values: 1e8 to 1e12 m^-2
    - All operations are differentiable
    - Real refractive index only (no absorption)

    Examples
    --------
    Create GRIN lens:

    >>> grin = gradient_index_material(
    ...     shape=(128, 128, 20),
    ...     n_center=1.5,
    ...     gradient_constant=1e10,
    ...     dx=1e-6,
    ...     tz=5e-6
    ... )
    """
    height: int
    width: int
    num_slices: int
    height, width, num_slices = shape

    y_coords: Float[Array, " h"] = (
        jnp.arange(height, dtype=jnp.float64) - height / 2.0
    ) * dx
    x_coords: Float[Array, " w"] = (
        jnp.arange(width, dtype=jnp.float64) - width / 2.0
    ) * dx

    yy_2d: Float[Array, " h w"]
    xx_2d: Float[Array, " h w"]
    yy_2d, xx_2d = jnp.meshgrid(y_coords, x_coords, indexing="ij")

    radial_distance_xy: Float[Array, " h w"] = jnp.sqrt(xx_2d**2 + yy_2d**2)

    n0: Float[Array, " "] = jnp.asarray(n_center, dtype=jnp.float64)
    steepness: Float[Array, " "] = jnp.asarray(
        gradient_constant, dtype=jnp.float64
    )

    n_profile: Float[Array, " h w"] = n0 * (
        1.0 - (steepness / 2.0) * radial_distance_xy**2
    )

    n_minimum: Float[Array, " "] = jnp.asarray(
        1.0 if n_min is None else n_min, dtype=jnp.float64
    )
    n_profile_clipped: Float[Array, " h w"] = jnp.clip(
        n_profile, n_minimum, n0
    )

    n_profile_3d: Float[Array, " h w z"] = jnp.broadcast_to(
        n_profile_clipped[:, :, None], (height, width, num_slices)
    )

    material_array: Complex[Array, " h w z"] = n_profile_3d.astype(
        jnp.complex128
    )

    material: SlicedMaterialFunction = make_sliced_material_function(
        material=material_array,
        dx=dx,
        tz=tz,
    )

    return material
