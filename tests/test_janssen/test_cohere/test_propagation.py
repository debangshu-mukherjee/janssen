"""Tests for coherent mode propagation in janssen.cohere.propagation."""

import chex
import jax.numpy as jnp
from absl.testing import parameterized

from janssen.cohere.modes import (
    hermite_gaussian_modes,
    modes_from_wavefront,
    thermal_mode_weights,
)
from janssen.cohere.propagation import (
    apply_element_to_modes,
    intensity_from_modes,
    intensity_from_polychromatic,
    propagate_and_focus_modes,
    propagate_coherent_modes,
    propagate_polychromatic,
)
from janssen.types import make_optical_wavefront, make_polychromatic_wavefront


def _hg_modes(wavelength, dx, grid_size, beam_waist, max_order):
    """Helper to create HG modes with thermal weights."""
    num_modes = (max_order + 1) * (max_order + 2) // 2
    weights = thermal_mode_weights(num_modes)
    return hermite_gaussian_modes(
        wavelength, dx, grid_size, beam_waist, max_order, weights
    )


class TestPropagateCoherentModes(chex.TestCase):
    """Test propagate_coherent_modes function."""

    @chex.variants(without_jit=True)
    def test_output_type(self) -> None:
        """Test that output is a CoherentModeSet."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=1
        )
        distance = 1e-3
        var_fn = self.variant(propagate_coherent_modes)
        propagated = var_fn(mode_set, distance)
        assert hasattr(propagated, "modes")
        assert hasattr(propagated, "weights")

    @chex.variants(without_jit=True)
    def test_mode_shapes_preserved(self) -> None:
        """Test that mode shapes are preserved after propagation."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 48)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=1
        )
        distance = 1e-3
        var_fn = self.variant(propagate_coherent_modes)
        propagated = var_fn(mode_set, distance)
        chex.assert_shape(propagated.modes, mode_set.modes.shape)

    @chex.variants(without_jit=True)
    def test_weights_preserved(self) -> None:
        """Test that weights are preserved after propagation."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=1
        )
        distance = 1e-3
        var_fn = self.variant(propagate_coherent_modes)
        propagated = var_fn(mode_set, distance)
        chex.assert_trees_all_close(
            propagated.weights, mode_set.weights, atol=1e-10
        )

    @chex.variants(without_jit=True)
    def test_z_position_updated(self) -> None:
        """Test that z_position is updated after propagation."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=1
        )
        distance = 5e-3
        var_fn = self.variant(propagate_coherent_modes)
        propagated = var_fn(mode_set, distance)
        expected_z = mode_set.z_position + distance
        chex.assert_trees_all_close(
            propagated.z_position, expected_z, rtol=1e-10
        )

    @chex.variants(without_jit=True)
    def test_angular_spectrum_method(self) -> None:
        """Test propagation with angular_spectrum method."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=0
        )
        distance = 1e-3
        var_fn = self.variant(propagate_coherent_modes)
        propagated = var_fn(mode_set, distance, method="angular_spectrum")
        assert jnp.all(jnp.isfinite(propagated.modes))

    @chex.variants(without_jit=True)
    def test_fresnel_method(self) -> None:
        """Test propagation with fresnel method."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=0
        )
        distance = 10e-3
        var_fn = self.variant(propagate_coherent_modes)
        propagated = var_fn(mode_set, distance, method="fresnel")
        assert jnp.all(jnp.isfinite(propagated.modes))

    def test_invalid_method_raises(self) -> None:
        """Test that invalid method raises ValueError."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=0
        )
        with self.assertRaises(ValueError):
            propagate_coherent_modes(mode_set, 1e-3, method="invalid")

    @chex.variants(without_jit=True)
    def test_refractive_index_effect(self) -> None:
        """Test that refractive index affects z_position update."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=0
        )
        distance = 1e-3
        refractive_index = 1.5
        var_fn = self.variant(propagate_coherent_modes)
        propagated = var_fn(
            mode_set, distance, refractive_index=refractive_index
        )
        expected_z = mode_set.z_position + distance * refractive_index
        chex.assert_trees_all_close(
            propagated.z_position, expected_z, rtol=1e-10
        )


class TestPropagatePolychromatic(chex.TestCase):
    """Test propagate_polychromatic function."""

    @chex.variants(without_jit=True)
    def test_output_type(self) -> None:
        """Test that output is a PolychromaticWavefront."""
        fields = jnp.ones((5, 32, 32), dtype=jnp.complex128)
        wavelengths = jnp.linspace(600e-9, 700e-9, 5)
        spectral_weights = jnp.ones(5) / 5
        wavefront = make_polychromatic_wavefront(
            fields=fields,
            wavelengths=wavelengths,
            spectral_weights=spectral_weights,
            dx=1e-6,
            z_position=0.0,
        )
        distance = 1e-3
        var_fn = self.variant(propagate_polychromatic)
        propagated = var_fn(wavefront, distance)
        assert hasattr(propagated, "fields")
        assert hasattr(propagated, "wavelengths")

    @chex.variants(without_jit=True)
    def test_field_shapes_preserved(self) -> None:
        """Test that field shapes are preserved after propagation."""
        fields = jnp.ones((7, 32, 48), dtype=jnp.complex128)
        wavelengths = jnp.linspace(500e-9, 600e-9, 7)
        spectral_weights = jnp.ones(7) / 7
        wavefront = make_polychromatic_wavefront(
            fields=fields,
            wavelengths=wavelengths,
            spectral_weights=spectral_weights,
            dx=1e-6,
            z_position=0.0,
        )
        distance = 1e-3
        var_fn = self.variant(propagate_polychromatic)
        propagated = var_fn(wavefront, distance)
        chex.assert_shape(propagated.fields, fields.shape)

    @chex.variants(without_jit=True)
    def test_spectral_weights_preserved(self) -> None:
        """Test that spectral weights are preserved after propagation."""
        fields = jnp.ones((5, 32, 32), dtype=jnp.complex128)
        wavelengths = jnp.linspace(600e-9, 700e-9, 5)
        spectral_weights = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
        wavefront = make_polychromatic_wavefront(
            fields=fields,
            wavelengths=wavelengths,
            spectral_weights=spectral_weights,
            dx=1e-6,
            z_position=0.0,
        )
        distance = 1e-3
        var_fn = self.variant(propagate_polychromatic)
        propagated = var_fn(wavefront, distance)
        chex.assert_trees_all_close(
            propagated.spectral_weights, spectral_weights, atol=1e-10
        )

    @chex.variants(without_jit=True)
    def test_z_position_updated(self) -> None:
        """Test that z_position is updated after propagation."""
        fields = jnp.ones((5, 32, 32), dtype=jnp.complex128)
        wavelengths = jnp.linspace(600e-9, 700e-9, 5)
        spectral_weights = jnp.ones(5) / 5
        z_initial = 0.01
        wavefront = make_polychromatic_wavefront(
            fields=fields,
            wavelengths=wavelengths,
            spectral_weights=spectral_weights,
            dx=1e-6,
            z_position=z_initial,
        )
        distance = 5e-3
        var_fn = self.variant(propagate_polychromatic)
        propagated = var_fn(wavefront, distance)
        expected_z = z_initial + distance
        chex.assert_trees_all_close(
            propagated.z_position, expected_z, rtol=1e-10
        )


class TestApplyElementToModes(chex.TestCase):
    """Test apply_element_to_modes function."""

    @chex.variants(without_jit=True)
    def test_output_type(self) -> None:
        """Test that output is a CoherentModeSet."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=1
        )

        def identity_element(wf):
            return wf

        var_fn = self.variant(apply_element_to_modes)
        transformed = var_fn(mode_set, identity_element)
        assert hasattr(transformed, "modes")
        assert hasattr(transformed, "weights")

    @chex.variants(without_jit=True)
    def test_identity_element(self) -> None:
        """Test that identity element preserves modes."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=1
        )

        def identity_element(wf):
            return wf

        var_fn = self.variant(apply_element_to_modes)
        transformed = var_fn(mode_set, identity_element)
        chex.assert_trees_all_close(
            transformed.modes, mode_set.modes, atol=1e-10
        )

    @chex.variants(without_jit=True)
    def test_amplitude_scaling(self) -> None:
        """Test that amplitude element scales modes correctly."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=0
        )
        scale_factor = 0.5

        def amplitude_element(wf):
            return make_optical_wavefront(
                field=wf.field * scale_factor,
                wavelength=wf.wavelength,
                dx=wf.dx,
                z_position=wf.z_position,
                polarization=wf.polarization,
            )

        var_fn = self.variant(apply_element_to_modes)
        transformed = var_fn(mode_set, amplitude_element)
        expected_modes = mode_set.modes * scale_factor
        chex.assert_trees_all_close(
            transformed.modes, expected_modes, atol=1e-10
        )

    @chex.variants(without_jit=True)
    def test_weights_preserved(self) -> None:
        """Test that weights are preserved after element application."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=1
        )

        def phase_element(wf):
            phase_mask = jnp.exp(1j * 0.5)
            return make_optical_wavefront(
                field=wf.field * phase_mask,
                wavelength=wf.wavelength,
                dx=wf.dx,
                z_position=wf.z_position,
                polarization=wf.polarization,
            )

        var_fn = self.variant(apply_element_to_modes)
        transformed = var_fn(mode_set, phase_element)
        chex.assert_trees_all_close(
            transformed.weights, mode_set.weights, atol=1e-10
        )


class TestIntensityFromModes(chex.TestCase):
    """Test intensity_from_modes function."""

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 48)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=1
        )
        var_fn = self.variant(intensity_from_modes)
        intensity = var_fn(mode_set)
        chex.assert_shape(intensity, grid_size)

    @chex.variants(without_jit=True)
    def test_non_negative(self) -> None:
        """Test that intensity is non-negative."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=2
        )
        var_fn = self.variant(intensity_from_modes)
        intensity = var_fn(mode_set)
        assert jnp.all(intensity >= 0)

    @chex.variants(without_jit=True)
    def test_single_mode_squared_amplitude(self) -> None:
        """Test that single mode gives squared amplitude."""
        field = jnp.ones((32, 32), dtype=jnp.complex128) * 2.0
        wavelength = 633e-9
        dx = 1e-6
        mode_set = modes_from_wavefront(field, wavelength, dx)
        var_fn = self.variant(intensity_from_modes)
        intensity = var_fn(mode_set)
        expected = jnp.abs(field) ** 2
        chex.assert_trees_all_close(intensity, expected, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_weighted_sum(self) -> None:
        """Test that intensity is weighted sum of mode intensities."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=1
        )
        var_fn = self.variant(intensity_from_modes)
        intensity = var_fn(mode_set)
        expected = jnp.zeros(grid_size)
        for i in range(mode_set.modes.shape[0]):
            expected += mode_set.weights[i] * jnp.abs(mode_set.modes[i]) ** 2
        chex.assert_trees_all_close(intensity, expected, atol=1e-10)


class TestIntensityFromPolychromatic(chex.TestCase):
    """Test intensity_from_polychromatic function."""

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        fields = jnp.ones((5, 32, 48), dtype=jnp.complex128)
        wavelengths = jnp.linspace(500e-9, 600e-9, 5)
        spectral_weights = jnp.ones(5) / 5
        wavefront = make_polychromatic_wavefront(
            fields=fields,
            wavelengths=wavelengths,
            spectral_weights=spectral_weights,
            dx=1e-6,
            z_position=0.0,
        )
        var_fn = self.variant(intensity_from_polychromatic)
        intensity = var_fn(wavefront)
        chex.assert_shape(intensity, (32, 48))

    @chex.variants(without_jit=True)
    def test_non_negative(self) -> None:
        """Test that intensity is non-negative."""
        fields = jnp.exp(1j * jnp.linspace(0, jnp.pi, 5 * 32 * 32)).reshape(
            5, 32, 32
        )
        wavelengths = jnp.linspace(500e-9, 600e-9, 5)
        spectral_weights = jnp.ones(5) / 5
        wavefront = make_polychromatic_wavefront(
            fields=fields,
            wavelengths=wavelengths,
            spectral_weights=spectral_weights,
            dx=1e-6,
            z_position=0.0,
        )
        var_fn = self.variant(intensity_from_polychromatic)
        intensity = var_fn(wavefront)
        assert jnp.all(intensity >= 0)

    @chex.variants(without_jit=True)
    def test_weighted_sum(self) -> None:
        """Test that intensity is weighted sum of field intensities."""
        fields = jnp.ones((3, 32, 32), dtype=jnp.complex128)
        fields = fields.at[0].set(fields[0] * 1.0)
        fields = fields.at[1].set(fields[1] * 2.0)
        fields = fields.at[2].set(fields[2] * 3.0)
        wavelengths = jnp.linspace(500e-9, 600e-9, 3)
        spectral_weights = jnp.array([0.2, 0.5, 0.3])
        wavefront = make_polychromatic_wavefront(
            fields=fields,
            wavelengths=wavelengths,
            spectral_weights=spectral_weights,
            dx=1e-6,
            z_position=0.0,
        )
        var_fn = self.variant(intensity_from_polychromatic)
        intensity = var_fn(wavefront)
        expected = (
            0.2 * jnp.abs(fields[0]) ** 2
            + 0.5 * jnp.abs(fields[1]) ** 2
            + 0.3 * jnp.abs(fields[2]) ** 2
        )
        chex.assert_trees_all_close(intensity, expected, atol=1e-10)


class TestPropagateAndFocusModes(chex.TestCase):
    """Test propagate_and_focus_modes function."""

    @chex.variants(without_jit=True)
    def test_output_type(self) -> None:
        """Test that output is a CoherentModeSet."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=0
        )
        focal_length = 50e-3
        propagation_distance = 50e-3
        var_fn = self.variant(propagate_and_focus_modes)
        focused = var_fn(mode_set, focal_length, propagation_distance)
        assert hasattr(focused, "modes")
        assert hasattr(focused, "weights")

    @chex.variants(without_jit=True)
    def test_mode_shapes_preserved(self) -> None:
        """Test that mode shapes are preserved after focusing."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 48)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=1
        )
        focal_length = 50e-3
        propagation_distance = 50e-3
        var_fn = self.variant(propagate_and_focus_modes)
        focused = var_fn(mode_set, focal_length, propagation_distance)
        chex.assert_shape(focused.modes, mode_set.modes.shape)

    @chex.variants(without_jit=True)
    def test_weights_preserved(self) -> None:
        """Test that weights are preserved after focusing."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=1
        )
        focal_length = 50e-3
        propagation_distance = 50e-3
        var_fn = self.variant(propagate_and_focus_modes)
        focused = var_fn(mode_set, focal_length, propagation_distance)
        chex.assert_trees_all_close(
            focused.weights, mode_set.weights, atol=1e-10
        )

    @chex.variants(without_jit=True)
    def test_finite_values(self) -> None:
        """Test that all output values are finite."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=0
        )
        focal_length = 50e-3
        propagation_distance = 50e-3
        var_fn = self.variant(propagate_and_focus_modes)
        focused = var_fn(mode_set, focal_length, propagation_distance)
        assert jnp.all(jnp.isfinite(focused.modes))

    @chex.variants(without_jit=True)
    def test_z_position_updated(self) -> None:
        """Test that z_position is updated after focusing."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        mode_set = _hg_modes(
            wavelength, dx, grid_size, beam_waist=20e-6, max_order=0
        )
        focal_length = 50e-3
        propagation_distance = 30e-3
        var_fn = self.variant(propagate_and_focus_modes)
        focused = var_fn(mode_set, focal_length, propagation_distance)
        expected_z = mode_set.z_position + propagation_distance
        chex.assert_trees_all_close(focused.z_position, expected_z, rtol=1e-10)
