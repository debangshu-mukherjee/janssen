"""Coherence types for partially coherent field representation.

Extended Summary
----------------
This module provides PyTree data structures for representing partially
coherent optical fields. It supports both spatial coherence (extended
sources, van Cittert-Zernike theorem) and temporal coherence (finite
bandwidth, chromatic effects).

The key insight is that any partially coherent field can be decomposed
into orthogonal coherent modes (Mercer's theorem), enabling efficient
simulation by propagating a finite number of modes and summing intensities.

Routine Listings
----------------
CoherentModeSet : NamedTuple
    PyTree for coherent mode decomposition of partially coherent fields
PolychromaticWavefront : NamedTuple
    PyTree structure for polychromatic/broadband field representation
MutualIntensity : NamedTuple
    PyTree structure for full mutual intensity J(r1, r2) representation
MixedStatePtychoData : NamedTuple
    PyTree for mixed-state ptychography reconstruction state
make_coherent_mode_set : function
    Factory function to create validated CoherentModeSet instances
make_polychromatic_wavefront : function
    Factory function to create validated PolychromaticWavefront instances
make_mutual_intensity : function
    Factory function to create validated MutualIntensity instances
make_mixed_state_ptycho_data : function
    Factory function to create validated MixedStatePtychoData instances

Notes
-----
For practical simulations, prefer CoherentModeSet over MutualIntensity:
- CoherentModeSet: O(M × N²) memory for M modes on N×N grid
- MutualIntensity: O(N⁴) memory - use only for small grids or demonstrations

The total intensity from coherent modes is:
    I(r) = Σₙ wₙ |φₙ(r)|²

where wₙ are the modal weights (eigenvalues) and φₙ are the modes.

References
----------
1. Mandel, L. & Wolf, E. "Optical Coherence and Quantum Optics" (1995)
2. Starikov, A. & Wolf, E. "Coherent-mode representation of Gaussian
   Schell-model sources" JOSA A (1982)
3. Thibault, P. & Menzel, A. "Reconstructing state mixtures from
   diffraction measurements" Nature (2013)
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple, Union
from jax import lax
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, jaxtyped

from .common_types import ScalarNumeric


@register_pytree_node_class
class CoherentModeSet(NamedTuple):
    """PyTree for coherent mode decomposition of partially coherent fields.

    A partially coherent field can be represented as a weighted sum of
    orthogonal coherent modes. The total intensity is the incoherent sum:
        I(r) = Σₙ weights[n] × |modes[n](r)|²

    This representation is memory-efficient O(M × N²) compared to the full
    mutual intensity O(N⁴), and naturally parallelizable with vmap.

    Attributes
    ----------
    modes : Union[Complex[Array, " num_modes hh ww"],
                  Complex[Array, " num_modes hh ww 2"]]
        Complex amplitude of coherent modes. Can be scalar (M, H, W) or
        polarized with Jones vectors (M, H, W, 2).
    weights : Float[Array, " num_modes"]
        Modal weights (eigenvalues from Mercer decomposition).
        Must be non-negative and sum to 1 for normalized representation.
    wavelength : Float[Array, " "]
        Wavelength of the optical field in meters.
    dx : Float[Array, " "]
        Spatial sampling interval (grid spacing) in meters.
    z_position : Float[Array, " "]
        Axial position of the modes along the propagation direction in meters.
    polarization : Bool[Array, " "]
        Whether the modes are polarized (True for 4D modes, False for 3D).

    Notes
    -----
    The effective number of modes (participation ratio) quantifies partial
    coherence:
        N_eff = (Σ wₙ)² / Σ(wₙ²)

    N_eff = 1 indicates full coherence (single mode dominates).
    Larger N_eff indicates greater partial coherence.

    For a Gaussian Schell-model source, the eigenvalues and modes have
    analytical forms in terms of Hermite-Gaussian functions.
    """

    modes: Union[
        Complex[Array, " num_modes hh ww"],
        Complex[Array, " num_modes hh ww 2"],
    ]
    weights: Float[Array, " num_modes"]
    wavelength: Float[Array, " "]
    dx: Float[Array, " "]
    z_position: Float[Array, " "]
    polarization: Bool[Array, " "]
    intensity: Float[Array, " hh ww"]
    effective_mode_count: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Union[
                Complex[Array, " num_modes hh ww"],
                Complex[Array, " num_modes hh ww 2"],
            ],
            Float[Array, " num_modes"],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Bool[Array, " "],
            Float[Array, " hh ww"],
            Float[Array, " "],
        ],
        None,
    ]:
        """Flatten the CoherentModeSet into a tuple of its components."""
        return (
            (
                self.modes,
                self.weights,
                self.wavelength,
                self.dx,
                self.z_position,
                self.polarization,
                self.intensity,
                self.effective_mode_count,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Union[
                Complex[Array, " num_modes hh ww"],
                Complex[Array, " num_modes hh ww 2"],
            ],
            Float[Array, " num_modes"],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Bool[Array, " "],
            Float[Array, " hh ww"],
            Float[Array, " "],
        ],
    ) -> "CoherentModeSet":
        """Unflatten the CoherentModeSet from a tuple of its components."""
        return cls(*children)


@register_pytree_node_class
class PolychromaticWavefront(NamedTuple):
    """PyTree structure for polychromatic/broadband field representation.

    Represents a field with finite spectral bandwidth for temporal coherence
    and chromatic simulations. The total intensity is the spectrally-weighted
    incoherent sum:
        I(r) = Σᵢ spectral_weights[i] × |fields[i](r)|²

    Attributes
    ----------
    fields : Union[Complex[Array, " num_wavelengths hh ww"],
                   Complex[Array, " num_wavelengths hh ww 2"]]
        Complex amplitude at each wavelength. Can be scalar (Nλ, H, W) or
        polarized with Jones vectors (Nλ, H, W, 2).
    wavelengths : Float[Array, " num_wavelengths"]
        Wavelength sample points in meters.
    spectral_weights : Float[Array, " num_wavelengths"]
        Normalized spectral weights S(λ). Must be non-negative and sum to 1.
    dx : Float[Array, " "]
        Spatial sampling interval (grid spacing) in meters.
    z_position : Float[Array, " "]
        Axial position along the propagation direction in meters.
    polarization : Bool[Array, " "]
        Whether the fields are polarized (True for 4D fields, False for 3D).

    Notes
    -----
    The coherence length for a Gaussian spectrum is:
        Lc = λ₀² / Δλ

    where λ₀ is the center wavelength and Δλ is the FWHM bandwidth.

    For accurate chromatic simulations, ensure sufficient wavelength sampling:
    - Use at least 5-11 wavelengths spanning ±2σ of the spectrum
    - More samples needed for broader spectra or longer propagation distances
    """

    fields: Union[
        Complex[Array, " num_wavelengths hh ww"],
        Complex[Array, " num_wavelengths hh ww 2"],
    ]
    wavelengths: Float[Array, " num_wavelengths"]
    spectral_weights: Float[Array, " num_wavelengths"]
    dx: Float[Array, " "]
    z_position: Float[Array, " "]
    polarization: Bool[Array, " "]
    intensity: Float[Array, " hh ww"]
    center_wavelength: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Union[
                Complex[Array, " num_wavelengths hh ww"],
                Complex[Array, " num_wavelengths hh ww 2"],
            ],
            Float[Array, " num_wavelengths"],
            Float[Array, " num_wavelengths"],
            Float[Array, " "],
            Float[Array, " "],
            Bool[Array, " "],
            Float[Array, " hh ww"],
            Float[Array, " "],
        ],
        None,
    ]:
        """Flatten PolychromaticWavefront into a tuple of its components."""
        return (
            (
                self.fields,
                self.wavelengths,
                self.spectral_weights,
                self.dx,
                self.z_position,
                self.polarization,
                self.intensity,
                self.center_wavelength,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Union[
                Complex[Array, " num_wavelengths hh ww"],
                Complex[Array, " num_wavelengths hh ww 2"],
            ],
            Float[Array, " num_wavelengths"],
            Float[Array, " num_wavelengths"],
            Float[Array, " "],
            Float[Array, " "],
            Bool[Array, " "],
            Float[Array, " hh ww"],
            Float[Array, " "],
        ],
    ) -> "PolychromaticWavefront":
        """Unflatten PolychromaticWavefront from a tuple of its components."""
        return cls(*children)


@register_pytree_node_class
class MutualIntensity(NamedTuple):
    """PyTree structure for full mutual intensity J(r₁, r₂) representation.

    The mutual intensity describes spatial coherence of a quasi-monochromatic
    field:
        J(r₁, r₂) = ⟨E*(r₁) E(r₂)⟩

    The complex degree of coherence normalizes this:
        μ(r₁, r₂) = J(r₁, r₂) / √(I(r₁) I(r₂))

    where |μ| = 1 indicates full coherence and |μ| = 0 indicates incoherence.

    Attributes
    ----------
    j_matrix : Complex[Array, " hh ww hh ww"]
        Full mutual intensity matrix J(r₁, r₂).
        j_matrix[i, j, k, l] = J(r[i,j], r[k,l])
    wavelength : Float[Array, " "]
        Wavelength of the optical field in meters.
    dx : Float[Array, " "]
        Spatial sampling interval (grid spacing) in meters.
    z_position : Float[Array, " "]
        Axial position along the propagation direction in meters.

    Warnings
    --------
    Memory scales as O(N⁴) for an N×N grid. For a 256×256 grid, this requires
    ~34 GB for complex128. Use only for small grids (≤64×64) or theoretical
    demonstrations. For practical simulations, use CoherentModeSet instead.

    Notes
    -----
    The mutual intensity can be decomposed into coherent modes via eigenvalue
    decomposition:
        J(r₁, r₂) = Σₙ λₙ φₙ*(r₁) φₙ(r₂)

    where λₙ are eigenvalues and φₙ are orthonormal eigenfunctions.
    """

    j_matrix: Complex[Array, " hh ww hh ww"]
    wavelength: Float[Array, " "]
    dx: Float[Array, " "]
    z_position: Float[Array, " "]
    intensity: Float[Array, " hh ww"]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Complex[Array, " hh ww hh ww"],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " hh ww"],
        ],
        None,
    ]:
        """Flatten the MutualIntensity into a tuple of its components."""
        return (
            (
                self.j_matrix,
                self.wavelength,
                self.dx,
                self.z_position,
                self.intensity,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Complex[Array, " hh ww hh ww"],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " "],
            Float[Array, " hh ww"],
        ],
    ) -> "MutualIntensity":
        """Unflatten the MutualIntensity from a tuple of its components."""
        return cls(*children)


@jaxtyped(typechecker=beartype)
def make_coherent_mode_set(
    modes: Union[
        Complex[Array, " num_modes hh ww"],
        Complex[Array, " num_modes hh ww 2"],
    ],
    weights: Float[Array, " num_modes"],
    wavelength: ScalarNumeric,
    dx: ScalarNumeric,
    z_position: ScalarNumeric = 0.0,
    polarization: Union[bool, Bool[Array, " "]] = False,
    normalize_weights: bool = True,
) -> CoherentModeSet:
    """Create a validated CoherentModeSet instance.

    Factory function that validates inputs and creates a CoherentModeSet
    PyTree suitable for partially coherent field simulations.

    Parameters
    ----------
    modes : Union[Complex[Array, " num_modes hh ww"],
                  Complex[Array, " num_modes hh ww 2"]]
        Complex amplitude of coherent modes. Shape (M, H, W) for scalar
        or (M, H, W, 2) for polarized fields.
    weights : Float[Array, " num_modes"]
        Modal weights (eigenvalues). Must be non-negative.
    wavelength : ScalarNumeric
        Wavelength of the optical field in meters. Must be positive.
    dx : ScalarNumeric
        Spatial sampling interval in meters. Must be positive.
    z_position : ScalarNumeric, optional
        Axial position in meters. Default is 0.0.
    polarization : Union[bool, Bool[Array, " "]], optional
        Whether modes are polarized. Accepts Python bool or JAX bool array.
        Default is False.
    normalize_weights : bool, optional
        If True, normalize weights to sum to 1. Default is True.

    Returns
    -------
    CoherentModeSet
        Validated coherent mode set instance.

    Raises
    ------
    ValueError
        If modes and weights have inconsistent shapes, or if validation fails.
    """
    non_polar_dim: int = 3
    polar_dim: int = 4
    modes = jnp.asarray(modes, dtype=jnp.complex128)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    wavelength_arr: Float[Array, " "] = jnp.asarray(
        wavelength, dtype=jnp.float64
    )
    dx_arr: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    z_position_arr: Float[Array, " "] = jnp.asarray(
        z_position, dtype=jnp.float64
    )
    polarization_arr: Bool[Array, " "] = jnp.asarray(
        polarization, dtype=jnp.bool_
    )

    polarization_arr = jnp.where(
        modes.ndim == polar_dim,
        jnp.asarray(modes.shape[-1] == 2, dtype=jnp.bool_),
        polarization_arr,
    )

    def validate_and_create() -> CoherentModeSet:
        def check_modes_shape() -> Complex[Array, "..."]:
            def check_polarized() -> Complex[Array, " num_modes hh ww 2"]:
                return lax.cond(
                    jnp.logical_and(
                        modes.ndim == polar_dim,
                        modes.shape[-1] == 2,
                    ),
                    lambda: modes,
                    lambda: lax.stop_gradient(
                        lax.cond(False, lambda: modes, lambda: modes)
                    ),
                )

            def check_scalar() -> Complex[Array, " num_modes hh ww"]:
                return lax.cond(
                    modes.ndim == non_polar_dim,
                    lambda: modes,
                    lambda: lax.stop_gradient(
                        lax.cond(False, lambda: modes, lambda: modes)
                    ),
                )

            return lax.cond(
                polarization_arr,
                check_polarized,
                check_scalar,
            )

        def check_weights_shape(
            m: Complex[Array, "..."],
        ) -> Float[Array, " num_modes"]:
            num_modes: int = m.shape[0]
            is_valid: Bool[Array, " "] = weights.shape[0] == num_modes
            return lax.cond(
                is_valid,
                lambda: weights,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: weights, lambda: weights)
                ),
            )

        def check_weights_nonnegative(
            w: Float[Array, " num_modes"],
        ) -> Float[Array, " num_modes"]:
            w_clipped: Float[Array, " num_modes"] = jnp.maximum(w, 0.0)
            return w_clipped

        def normalize_weights_fn(
            w: Float[Array, " num_modes"],
        ) -> Float[Array, " num_modes"]:
            w_sum: Float[Array, " "] = jnp.sum(w)
            return lax.cond(
                jnp.logical_and(normalize_weights, w_sum > 1e-12),
                lambda: w / w_sum,
                lambda: w,
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

        validated_modes: Complex[Array, "..."] = check_modes_shape()
        validated_weights: Float[Array, " num_modes"] = normalize_weights_fn(
            check_weights_nonnegative(check_weights_shape(validated_modes))
        )
        validated_wavelength: Float[Array, " "] = check_wavelength()
        validated_dx: Float[Array, " "] = check_dx()

        def _compute_intensity_int() -> Float[Array, " hh ww"]:
            """Compute total intensity from incoherent mode sum."""
            abs_squared: Float[Array, "..."] = jnp.abs(validated_modes) ** 2
            is_polarized: bool = validated_modes.ndim == 4
            mode_intensities: Float[Array, " num_modes hh ww"] = (
                jnp.sum(abs_squared, axis=-1) if is_polarized else abs_squared
            )
            total: Float[Array, " hh ww"] = jnp.sum(
                validated_weights[:, jnp.newaxis, jnp.newaxis]
                * mode_intensities,
                axis=0,
            )
            return total

        def _compute_effective_mode_count_int() -> Float[Array, " "]:
            """Compute effective number of modes (participation ratio)."""
            weights_sum: Float[Array, " "] = jnp.sum(validated_weights)
            weights_sq_sum: Float[Array, " "] = jnp.sum(validated_weights**2)
            n_eff: Float[Array, " "] = weights_sum**2 / (
                weights_sq_sum + 1e-12
            )
            return n_eff

        intensity: Float[Array, " hh ww"] = _compute_intensity_int()
        effective_mode_count: Float[Array, " "] = (
            _compute_effective_mode_count_int()
        )

        return CoherentModeSet(
            modes=validated_modes,
            weights=validated_weights,
            wavelength=validated_wavelength,
            dx=validated_dx,
            z_position=z_position_arr,
            polarization=polarization_arr,
            intensity=intensity,
            effective_mode_count=effective_mode_count,
        )

    return validate_and_create()


@jaxtyped(typechecker=beartype)
def make_polychromatic_wavefront(
    fields: Union[
        Complex[Array, " num_wavelengths hh ww"],
        Complex[Array, " num_wavelengths hh ww 2"],
    ],
    wavelengths: Float[Array, " num_wavelengths"],
    spectral_weights: Float[Array, " num_wavelengths"],
    dx: ScalarNumeric,
    z_position: ScalarNumeric = 0.0,
    polarization: Union[bool, Bool[Array, " "]] = False,
    normalize_weights: bool = True,
) -> PolychromaticWavefront:
    """Create a validated PolychromaticWavefront instance.

    Factory function that validates inputs and creates a PolychromaticWavefront
    PyTree suitable for chromatic/temporal coherence simulations.

    Parameters
    ----------
    fields : Union[Complex[Array, " num_wavelengths hh ww"],
                   Complex[Array, " num_wavelengths hh ww 2"]]
        Complex amplitude at each wavelength. Shape (Nλ, H, W) for scalar
        or (Nλ, H, W, 2) for polarized fields.
    wavelengths : Float[Array, " num_wavelengths"]
        Wavelength sample points in meters. Must be positive.
    spectral_weights : Float[Array, " num_wavelengths"]
        Spectral weights S(λ). Must be non-negative.
    dx : ScalarNumeric
        Spatial sampling interval in meters. Must be positive.
    z_position : ScalarNumeric, optional
        Axial position in meters. Default is 0.0.
    polarization : Union[bool, Bool[Array, " "]], optional
        Whether fields are polarized. Accepts Python bool or JAX bool array.
        Default is False.
    normalize_weights : bool, optional
        If True, normalize spectral_weights to sum to 1. Default is True.

    Returns
    -------
    PolychromaticWavefront
        Validated polychromatic wavefront instance.
    """
    non_polar_dim: int = 3
    polar_dim: int = 4
    fields = jnp.asarray(fields, dtype=jnp.complex128)
    wavelengths = jnp.asarray(wavelengths, dtype=jnp.float64)
    spectral_weights = jnp.asarray(spectral_weights, dtype=jnp.float64)
    dx_arr: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    z_position_arr: Float[Array, " "] = jnp.asarray(
        z_position, dtype=jnp.float64
    )
    polarization_arr: Bool[Array, " "] = jnp.asarray(
        polarization, dtype=jnp.bool_
    )

    polarization_arr = jnp.where(
        fields.ndim == polar_dim,
        jnp.asarray(fields.shape[-1] == 2, dtype=jnp.bool_),
        polarization_arr,
    )

    def validate_and_create() -> PolychromaticWavefront:
        def check_fields_shape() -> Complex[Array, "..."]:
            def check_polarized() -> Complex[Array, " n hh ww 2"]:
                return lax.cond(
                    jnp.logical_and(
                        fields.ndim == polar_dim,
                        fields.shape[-1] == 2,
                    ),
                    lambda: fields,
                    lambda: lax.stop_gradient(
                        lax.cond(False, lambda: fields, lambda: fields)
                    ),
                )

            def check_scalar() -> Complex[Array, " num_wavelengths hh ww"]:
                return lax.cond(
                    fields.ndim == non_polar_dim,
                    lambda: fields,
                    lambda: lax.stop_gradient(
                        lax.cond(False, lambda: fields, lambda: fields)
                    ),
                )

            return lax.cond(
                polarization_arr,
                check_polarized,
                check_scalar,
            )

        def check_wavelengths_shape(
            f: Complex[Array, "..."],
        ) -> Float[Array, " num_wavelengths"]:
            num_wl: int = f.shape[0]
            is_valid: Bool[Array, " "] = wavelengths.shape[0] == num_wl
            return lax.cond(
                is_valid,
                lambda: wavelengths,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: wavelengths, lambda: wavelengths)
                ),
            )

        def check_spectral_weights_shape(
            f: Complex[Array, "..."],
        ) -> Float[Array, " num_wavelengths"]:
            num_wl: int = f.shape[0]
            is_valid: Bool[Array, " "] = spectral_weights.shape[0] == num_wl
            return lax.cond(
                is_valid,
                lambda: spectral_weights,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: spectral_weights,
                        lambda: spectral_weights,
                    )
                ),
            )

        def check_weights_nonnegative(
            w: Float[Array, " num_wavelengths"],
        ) -> Float[Array, " num_wavelengths"]:
            return jnp.maximum(w, 0.0)

        def normalize_weights_fn(
            w: Float[Array, " num_wavelengths"],
        ) -> Float[Array, " num_wavelengths"]:
            w_sum: Float[Array, " "] = jnp.sum(w)
            return lax.cond(
                jnp.logical_and(normalize_weights, w_sum > 1e-12),
                lambda: w / w_sum,
                lambda: w,
            )

        def check_wavelengths_positive(
            wl: Float[Array, " num_wavelengths"],
        ) -> Float[Array, " num_wavelengths"]:
            is_valid: Bool[Array, " "] = jnp.all(wl > 0)
            return lax.cond(
                is_valid,
                lambda: wl,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: wl, lambda: wl)
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

        validated_fields: Complex[Array, "..."] = check_fields_shape()
        validated_wavelengths: Float[Array, " num_wavelengths"] = (
            check_wavelengths_positive(
                check_wavelengths_shape(validated_fields)
            )
        )
        validated_spectral_weights: Float[Array, " num_wavelengths"] = (
            normalize_weights_fn(
                check_weights_nonnegative(
                    check_spectral_weights_shape(validated_fields)
                )
            )
        )
        validated_dx: Float[Array, " "] = check_dx()

        def _compute_intensity_int() -> Float[Array, " hh ww"]:
            """Compute total intensity from spectral sum."""
            abs_squared: Float[Array, "..."] = jnp.abs(validated_fields) ** 2
            is_polarized: bool = validated_fields.ndim == 4
            field_intensities: Float[Array, " num_wavelengths hh ww"] = (
                jnp.sum(abs_squared, axis=-1) if is_polarized else abs_squared
            )
            total: Float[Array, " hh ww"] = jnp.sum(
                validated_spectral_weights[:, jnp.newaxis, jnp.newaxis]
                * field_intensities,
                axis=0,
            )
            return total

        def _compute_center_wavelength_int() -> Float[Array, " "]:
            """Compute weighted center wavelength."""
            center: Float[Array, " "] = jnp.sum(
                validated_spectral_weights * validated_wavelengths
            )
            return center

        intensity: Float[Array, " hh ww"] = _compute_intensity_int()
        center_wavelength: Float[Array, " "] = _compute_center_wavelength_int()

        return PolychromaticWavefront(
            fields=validated_fields,
            wavelengths=validated_wavelengths,
            spectral_weights=validated_spectral_weights,
            dx=validated_dx,
            z_position=z_position_arr,
            polarization=polarization_arr,
            intensity=intensity,
            center_wavelength=center_wavelength,
        )

    return validate_and_create()


@jaxtyped(typechecker=beartype)
def make_mutual_intensity(
    j_matrix: Complex[Array, " hh ww hh ww"],
    wavelength: ScalarNumeric,
    dx: ScalarNumeric,
    z_position: ScalarNumeric = 0.0,
) -> MutualIntensity:
    """Create a validated MutualIntensity instance.

    Factory function that validates inputs and creates a MutualIntensity
    PyTree. Use with caution due to O(N⁴) memory scaling.

    Parameters
    ----------
    j_matrix : Complex[Array, " hh ww hh ww"]
        Full mutual intensity matrix J(r₁, r₂).
    wavelength : ScalarNumeric
        Wavelength in meters. Must be positive.
    dx : ScalarNumeric
        Spatial sampling interval in meters. Must be positive.
    z_position : ScalarNumeric, optional
        Axial position in meters. Default is 0.0.

    Returns
    -------
    MutualIntensity
        Validated mutual intensity instance.

    Warnings
    --------
    Memory scales as O(N⁴). For 64×64 grid: ~134 MB.
    For 128×128 grid: ~2 GB. For 256×256 grid: ~34 GB.
    Consider using CoherentModeSet for larger grids.
    """
    j_matrix = jnp.asarray(j_matrix, dtype=jnp.complex128)
    wavelength_arr: Float[Array, " "] = jnp.asarray(
        wavelength, dtype=jnp.float64
    )
    dx_arr: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    z_position_arr: Float[Array, " "] = jnp.asarray(
        z_position, dtype=jnp.float64
    )

    def validate_and_create() -> MutualIntensity:
        def check_j_matrix_shape() -> Complex[Array, " hh ww hh ww"]:
            is_valid_ndim: Bool[Array, " "] = j_matrix.ndim == 4
            is_valid_shape: Bool[Array, " "] = jnp.logical_and(
                j_matrix.shape[0] == j_matrix.shape[2],
                j_matrix.shape[1] == j_matrix.shape[3],
            )
            is_valid: Bool[Array, " "] = jnp.logical_and(
                is_valid_ndim, is_valid_shape
            )
            return lax.cond(
                is_valid,
                lambda: j_matrix,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: j_matrix, lambda: j_matrix)
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

        validated_j_matrix: Complex[Array, " hh ww hh ww"] = (
            check_j_matrix_shape()
        )
        validated_wavelength: Float[Array, " "] = check_wavelength()
        validated_dx: Float[Array, " "] = check_dx()

        def _compute_intensity_int() -> Float[Array, " hh ww"]:
            """Compute intensity I(r) = J(r, r) from diagonal."""
            diagonal: Float[Array, " hh ww"] = jnp.real(
                jnp.diagonal(
                    jnp.diagonal(validated_j_matrix, axis1=0, axis2=2),
                    axis1=0,
                    axis2=1,
                )
            )
            return diagonal

        intensity: Float[Array, " hh ww"] = _compute_intensity_int()

        return MutualIntensity(
            j_matrix=validated_j_matrix,
            wavelength=validated_wavelength,
            dx=validated_dx,
            z_position=z_position_arr,
            intensity=intensity,
        )

    return validate_and_create()


@register_pytree_node_class
class MixedStatePtychoData(NamedTuple):
    """PyTree structure for mixed-state ptychography reconstruction.

    Extends standard ptychography data to support partially coherent
    illumination via coherent mode decomposition.

    Attributes
    ----------
    diffraction_patterns : Float[Array, " N H W"]
        Measured diffraction intensities at each scan position.
    probe_modes : CoherentModeSet
        Coherent mode decomposition of the partially coherent probe.
    sample : Complex[Array, " Hs Ws"]
        Object transmission function estimate.
    positions : Float[Array, " N 2"]
        Scan positions in pixels.
    wavelength : Float[Array, " "]
        Wavelength in meters.
    dx : Float[Array, " "]
        Pixel spacing in meters.

    Notes
    -----
    The forward model is:
        I_i = Sigma_n w_n |FFT(probe_n * shift(object, r_i))|^2

    Gradients flow through probe_modes.modes, probe_modes.weights,
    and sample, enabling joint optimization of all parameters.
    """

    diffraction_patterns: Float[Array, " N H W"]
    probe_modes: CoherentModeSet
    sample: Complex[Array, " Hs Ws"]
    positions: Float[Array, " N 2"]
    wavelength: Float[Array, " "]
    dx: Float[Array, " "]

    def tree_flatten(
        self,
    ) -> Tuple[
        Tuple[
            Float[Array, " N H W"],
            CoherentModeSet,
            Complex[Array, " Hs Ws"],
            Float[Array, " N 2"],
            Float[Array, " "],
            Float[Array, " "],
        ],
        None,
    ]:
        """Flatten for JAX pytree compatibility."""
        children = (
            self.diffraction_patterns,
            self.probe_modes,
            self.sample,
            self.positions,
            self.wavelength,
            self.dx,
        )
        return (children, None)

    @classmethod
    def tree_unflatten(
        cls,
        _aux_data: None,
        children: Tuple[
            Float[Array, " N H W"],
            CoherentModeSet,
            Complex[Array, " Hs Ws"],
            Float[Array, " N 2"],
            Float[Array, " "],
            Float[Array, " "],
        ],
    ) -> "MixedStatePtychoData":
        """Unflatten from JAX pytree representation."""
        return cls(*children)


@jaxtyped(typechecker=beartype)
def make_mixed_state_ptycho_data(
    diffraction_patterns: Float[Array, " N H W"],
    probe_modes: CoherentModeSet,
    sample: Complex[Array, " Hs Ws"],
    positions: Float[Array, " N 2"],
    wavelength: ScalarNumeric,
    dx: ScalarNumeric,
) -> MixedStatePtychoData:
    """Create validated MixedStatePtychoData.

    Factory function that validates inputs and creates a MixedStatePtychoData
    PyTree suitable for mixed-state ptychography reconstruction.

    Parameters
    ----------
    diffraction_patterns : Float[Array, " N H W"]
        Measured diffraction patterns.
    probe_modes : CoherentModeSet
        Partially coherent probe as coherent modes.
    sample : Complex[Array, " Hs Ws"]
        Initial object estimate.
    positions : Float[Array, " N 2"]
        Scan positions (x, y) in pixels.
    wavelength : ScalarNumeric
        Wavelength in meters. Must be positive.
    dx : ScalarNumeric
        Pixel size in meters. Must be positive.

    Returns
    -------
    data : MixedStatePtychoData
        Validated data structure.
    """
    diffraction_patterns_arr: Float[Array, " N H W"] = jnp.asarray(
        diffraction_patterns, dtype=jnp.float64
    )
    sample_arr: Complex[Array, " Hs Ws"] = jnp.asarray(
        sample, dtype=jnp.complex128
    )
    positions_arr: Float[Array, " N 2"] = jnp.asarray(
        positions, dtype=jnp.float64
    )
    wavelength_arr: Float[Array, " "] = jnp.asarray(
        wavelength, dtype=jnp.float64
    )
    dx_arr: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)

    expected_dp_ndim: int = 3
    expected_pos_cols: int = 2

    def validate_and_create() -> MixedStatePtychoData:
        def check_diffraction_patterns() -> Float[Array, " N H W"]:
            dp_arr = diffraction_patterns_arr
            is_valid_ndim: Bool[Array, " "] = dp_arr.ndim == expected_dp_ndim
            is_non_negative: Bool[Array, " "] = jnp.all(dp_arr >= 0)
            is_valid: Bool[Array, " "] = jnp.logical_and(
                is_valid_ndim, is_non_negative
            )
            return lax.cond(
                is_valid,
                lambda: dp_arr,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: dp_arr, lambda: dp_arr)
                ),
            )

        def check_positions(
            dp: Float[Array, " N H W"],
        ) -> Float[Array, " N 2"]:
            num_positions: int = dp.shape[0]
            is_valid_shape: Bool[Array, " "] = jnp.logical_and(
                positions_arr.shape[0] == num_positions,
                positions_arr.shape[1] == expected_pos_cols,
            )
            return lax.cond(
                is_valid_shape,
                lambda: positions_arr,
                lambda: lax.stop_gradient(
                    lax.cond(
                        False, lambda: positions_arr, lambda: positions_arr
                    )
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

        validated_dp: Float[Array, " N H W"] = check_diffraction_patterns()
        validated_positions: Float[Array, " N 2"] = check_positions(
            validated_dp
        )
        validated_wavelength: Float[Array, " "] = check_wavelength()
        validated_dx: Float[Array, " "] = check_dx()

        return MixedStatePtychoData(
            diffraction_patterns=validated_dp,
            probe_modes=probe_modes,
            sample=sample_arr,
            positions=validated_positions,
            wavelength=validated_wavelength,
            dx=validated_dx,
        )

    return validate_and_create()
