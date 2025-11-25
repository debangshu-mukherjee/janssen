"""Material propagation functions for 3D samples.

Extended Summary
----------------
Multi-slice beam propagation method (BPM) for simulating light propagation
through 3D materials with spatially-varying complex refractive indices.
Implements the split-step beam propagation method that alternates between
material interaction and free-space propagation.

Routine Listings
----------------
multislice_propagation : function
    Propagate optical wavefront through 3D material using multi-slice BPM
optical_path_length : function
    Compute the optical path length through a material.
total_transmit : function
    Compute the total transmission through a material.

Notes
-----
The multi-slice method decomposes 3D propagation into a series of 2D
interactions followed by thin propagation steps. This is accurate when
the slice thickness is small compared to relevant length scales.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Complex, Float, jaxtyped

from janssen.utils import (
    OpticalWavefront,
    ScalarFloat,
    ScalarInteger,
    SlicedMaterialFunction,
    make_optical_wavefront,
)

from .free_space_prop import correct_propagator, optical_zoom

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def multislice_propagation(
    incoming: OpticalWavefront,
    material: SlicedMaterialFunction,
) -> OpticalWavefront:
    """Propagate optical wavefront through 3D material using multi-slice BPM.

    Uses the split-step beam propagation method to simulate light
    propagation through a 3D material with spatially-varying complex
    refractive index. The material is divided into thin slices, and
    propagation alternates between material interaction and free-space
    propagation.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input optical wavefront at the entrance of the material.
        If incoming.dx differs from material.dx, the wavefront will be
        automatically resampled to the smaller dx value for better
        resolution.
    material : SlicedMaterialFunction
        3D material with complex refractive index at each voxel.
        material.material[i, j, k] = n(i,j,k) + i*κ(i,j,k)
        where n is refractive index and κ is extinction coefficient.

    Returns
    -------
    outgoing : OpticalWavefront
        Optical wavefront at the exit of the material, after propagating
        through all slices. The z_position is updated to reflect the
        total propagation distance.

    Notes
    -----
    Implementation Details:

    **Spatial Sampling Handling:**
    If incoming.dx differs from material.dx, the incoming wavefront is
    automatically resampled using optical_zoom to match the smaller dx
    (better resolution). Tolerance for dx matching is 1e-10.

    **Algorithm:**
    1. Resample incoming wavefront to target dx if needed
    2. Calculate wavenumber: k = 2π/λ
    3. Initialize propagating field and z position
    4. For each slice in material:
        a. Extract complex refractive index: n + iκ
        b. Compute phase shift: φ = k × (n - 1) × tz
        c. Compute absorption coefficient: α = 4πκ/λ
        d. Compute absorption: A = exp(-α × tz)
        e. Apply material interaction: field *= A × exp(iφ)
        f. Create wavefront at current z position
        g. Propagate to next slice using correct_propagator
        h. Update z position: z += tz
    5. Return final wavefront with updated field and z position

    **Propagation Method:**
    The propagation method between slices is automatically selected by
    correct_propagator based on the Fresnel number and propagation
    distance. No manual method selection is required.

    **Multi-slice Approximation Validity:**
    - Slice thickness << characteristic propagation distances
    - Paraxial approximation holds (small angles)
    - Refractive index varies slowly within each slice

    **Material Interaction Formulas:**
    - Phase shift: φ = (2π/λ) × (n - 1) × tz
    - Absorption coefficient: α = 4πκ/λ
    - Absorption: A = exp(-α × tz)
    - Combined transmission: T = A × exp(iφ)

    Warnings
    --------
    - For highly absorbing materials (large κ), numerical precision
      may be reduced
    - Very thick slices (tz >> λ) may violate the thin-slice
      approximation

    Examples
    --------
    Propagate through a uniform glass slab:

    >>> import jax.numpy as jnp
    >>> from janssen.utils import make_optical_wavefront,
    >>>                           make_sliced_material_function
    >>> from janssen.prop import multislice_propagation
    >>>
    >>> # Create input wavefront
    >>> field = jnp.ones((128, 128), dtype=jnp.complex128)
    >>> wavefront = make_optical_wavefront(
    ...     field=field, wavelength=550e-9, dx=1e-6, z_position=0.0
    ... )
    >>>
    >>> # Create glass material (n=1.5, no absorption)
    >>> material_array = jnp.ones((128, 128, 20)) * (1.5 + 0.0j)
    >>> material = make_sliced_material_function(
    ...     material=material_array, dx=1e-6, tz=5e-6
    ... )
    >>>
    >>> # Propagate through material
    >>> output = multislice_propagation(wavefront, material)
    >>> print(f"Total propagation: {output.z_position:.2e} m")

    See Also
    --------
    correct_propagator : Automatic selection of propagation method
    optical_zoom : Resampling function
    """
    dx_tolerance: float = 1e-10
    dx_mismatch: Float[Array, " "] = jnp.abs(incoming.dx - material.dx)
    target_dx: Float[Array, " "] = jnp.minimum(incoming.dx, material.dx)

    resampled_incoming: OpticalWavefront = jax.lax.cond(
        dx_mismatch > dx_tolerance,
        lambda: optical_zoom(incoming, incoming.dx / target_dx),
        lambda: incoming,
    )

    height: int
    width: int
    num_slices: int
    height, width, num_slices = material.material.shape

    k: Float[Array, " "] = 2 * jnp.pi / resampled_incoming.wavelength

    current_field: Complex[Array, " H W"] = resampled_incoming.field
    current_z: Float[Array, " "] = resampled_incoming.z_position

    def slice_step(
        carry: tuple[Complex[Array, " H W"], Float[Array, " "]],
        slice_idx: int,
    ) -> tuple[tuple[Complex[Array, " H W"], Float[Array, " "]], None]:
        field: Complex[Array, " H W"]
        z_pos: Float[Array, " "]
        field, z_pos = carry

        n_slice: Complex[Array, " H W"] = material.material[:, :, slice_idx]
        n_real: Float[Array, " H W"] = n_slice.real
        kappa: Float[Array, " H W"] = n_slice.imag

        phase_shift: Float[Array, " H W"] = k * (n_real - 1.0) * material.tz

        alpha: Float[Array, " H W"] = (
            4 * jnp.pi * kappa / resampled_incoming.wavelength
        )
        absorption: Float[Array, " H W"] = jnp.exp(-alpha * material.tz)

        transmission: Complex[Array, " H W"] = absorption * jnp.exp(
            1j * phase_shift
        )
        field_after_material: Complex[Array, " H W"] = field * transmission

        wf: OpticalWavefront = make_optical_wavefront(
            field=field_after_material,
            wavelength=resampled_incoming.wavelength,
            dx=target_dx,
            z_position=z_pos,
        )
        propagated: OpticalWavefront = correct_propagator(
            wf, material.tz, refractive_index=1.0
        )
        field_propagated: Complex[Array, " H W"] = propagated.field

        new_z: Float[Array, " "] = z_pos + material.tz

        return (field_propagated, new_z), None

    initial_carry: tuple[Complex[Array, " H W"], Float[Array, " "]] = (
        current_field,
        current_z,
    )
    final_carry, _ = jax.lax.scan(
        slice_step, initial_carry, jnp.arange(num_slices)
    )

    final_field: Complex[Array, " H W"]
    final_z: Float[Array, " "]
    final_field, final_z = final_carry

    outgoing: OpticalWavefront = make_optical_wavefront(
        field=final_field,
        wavelength=resampled_incoming.wavelength,
        dx=target_dx,
        z_position=final_z,
    )

    return outgoing


@jaxtyped(typechecker=beartype)
def optical_path_length(
    material: SlicedMaterialFunction,
    x_idx: Optional[ScalarInteger] = -1,
    y_idx: Optional[ScalarInteger] = -1,
) -> Float[Array, " H W"]:
    """Compute optical path length through material.

    Calculates the optical path length (OPL) along rays through the
    material. OPL accounts for both the physical distance and the
    refractive index.

    Parameters
    ----------
    material : SlicedMaterialFunction
        3D material with complex refractive index.
    x_idx : ScalarInteger, optional
        X-index for specific ray. If -1 (default), computes for all x.
        When specified, the result is tiled to maintain (H, W) shape.
    y_idx : ScalarInteger, optional
        Y-index for specific ray. If -1 (default), computes for all y.
        When specified, the result is tiled to maintain (H, W) shape.

    Returns
    -------
    opl : Float[Array, " H W"]
        Optical path length in meters as a 2D array (H, W).
        When indices are specified, the result is broadcast/tiled:
        - Both indices: scalar value tiled to (H, W)
        - x_idx only: 1D result tiled along y to (H, W)
        - y_idx only: 1D result tiled along x to (H, W)
        - Neither: full 2D projection (H, W)

        Users can extract specific values with indexing, e.g.,
        opl[0, 0] for scalar, opl[0, :] for x-line, opl[:, 0] for y-line.

    Notes
    -----
    Implementation:
    1. Extract real part of material (refractive index n)
    2. Get material dimensions (height, width)
    3. Use nested jax.lax.cond to select computation:
       - All branches return same shape (H, W) for JIT compatibility
       - Both indices: broadcast scalar to (H, W)
       - Only x_idx: tile 1D result to (H, W)
       - Only y_idx: tile 1D result to (H, W)
       - Neither - compute 2D projection directly.
    4. Extract appropriate slice/element from 2D result
    5. Sum refractive indices along z and multiply by slice thickness

    Uses jax.lax.cond with uniform output shapes, then extracts the
    relevant data. This ensures JIT compilation compatibility while
    supporting dynamic return shapes.

    Formula:
    OPL = Σ_z n(x, y, z) × tz

    This is the phase delay in distance units. To convert to phase:
    φ = (2π/λ) × OPL

    Examples
    --------
    >>> # Compute OPL for entire material
    >>> opl_map = optical_path_length(material)
    >>> # Compute OPL along center ray
    >>> center_opl = optical_path_length(material, 64, 64)
    """
    n_material: Float[Array, " H W Z"] = material.material.real
    height: int
    width: int
    height, width, _ = material.material.shape

    def _both_indices() -> Float[Array, " H W"]:
        """Compute OPL for single ray and tile to full array."""
        n_ray: Float[Array, " Z"] = n_material[y_idx, x_idx, :]
        scalar_opl: Float[Array, " "] = jnp.sum(n_ray) * material.tz
        return jnp.full((height, width), scalar_opl)

    def _x_only() -> Float[Array, " H W"]:
        """Compute OPL for all y at fixed x and tile along height."""
        n_line: Float[Array, " W Z"] = n_material[:, x_idx, :]
        line_opl: Float[Array, " W"] = jnp.sum(n_line, axis=1) * material.tz
        return jnp.tile(line_opl[jnp.newaxis, :], (height, 1))

    def _y_or_neither() -> Float[Array, " H W"]:
        """Branch for y-only or full 2D projection."""

        def y_only() -> Float[Array, " H W"]:
            """Compute OPL for all x at fixed y and tile along width."""
            n_line: Float[Array, " H Z"] = n_material[y_idx, :, :]
            line_opl: Float[Array, " H"] = (
                jnp.sum(n_line, axis=1) * material.tz
            )
            return jnp.tile(line_opl[:, jnp.newaxis], (1, width))

        def neither() -> Float[Array, " H W"]:
            """Compute OPL for entire 2D projection."""
            return jnp.sum(n_material, axis=2) * material.tz

        return jax.lax.cond(y_idx >= 0, y_only, neither)

    def _x_branch() -> Float[Array, " H W"]:
        """Branch when x_idx is specified."""
        return jax.lax.cond(y_idx >= 0, _both_indices, _x_only)

    opl: Float[Array, " H W"] = jax.lax.cond(
        x_idx >= 0, _x_branch, _y_or_neither
    )

    return opl


@jaxtyped(typechecker=beartype)
def total_transmit(
    material: SlicedMaterialFunction,
    wavelength: ScalarFloat,
) -> Float[Array, " H W"]:
    """Compute intensity transmission through material.

    Calculates the total intensity transmission (squared amplitude) for
    light passing through the material, accounting for absorption at
    each slice.

    Parameters
    ----------
    material : SlicedMaterialFunction
        3D material with complex refractive index.
    wavelength : ScalarFloat
        Wavelength of light in meters.

    Returns
    -------
    transmission : Float[Array, " H W"]
        Intensity transmission map (0 to 1). Values < 1 indicate
        absorption.

    Notes
    -----
    Implementation:
    1. Convert wavelength to JAX array
    2. Extract extinction coefficient κ (imaginary part of material)
    3. Compute absorption coefficient: α = 4πκ/λ
    4. Compute amplitude transmission per slice: exp(-α × tz)
    5. Compute total amplitude transmission (product over all slices)
    6. Compute intensity transmission: |amplitude|²

    Formula:
    For each position (x, y):
        T(x,y) = Π_z exp(-2 × α(x,y,z) × tz)
        where α = 4πκ/λ

    The factor of 2 in the exponent comes from intensity = |amplitude|²

    Examples
    --------
    >>> transmission = total_transmit(material, 550e-9)
    >>> absorption_percent = (1 - transmission) * 100
    >>> print(f"Max absorption: {jnp.max(absorption_percent):.1f}%")
    """
    wavelength_array: Float[Array, " "] = jnp.asarray(
        wavelength, dtype=jnp.float64
    )
    kappa: Float[Array, " H W Z"] = material.material.imag
    alpha: Float[Array, " H W Z"] = 4 * jnp.pi * kappa / wavelength_array
    amplitude_transmission: Float[Array, " H W Z"] = jnp.exp(
        -alpha * material.tz
    )
    total_amplitude: Float[Array, " H W"] = jnp.prod(
        amplitude_transmission, axis=2
    )
    intensity_transmission: Float[Array, " H W"] = total_amplitude**2
    return intensity_transmission
