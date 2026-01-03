"""Tests for source models in janssen.coherence.sources."""

import chex
import jax.numpy as jnp
from absl.testing import parameterized

from janssen.coherence.sources import (
    laser_with_mode_noise,
    led_source,
    multimode_fiber_output,
    synchrotron_source,
    thermal_source,
)


class TestLedSource(chex.TestCase):
    """Test led_source function."""

    @chex.variants(without_jit=True)
    def test_output_types(self) -> None:
        """Test that outputs have correct types."""
        center_wl = 530e-9
        bandwidth = 30e-9
        coherence_width = 50e-6
        dx = 1e-6
        grid_size = (32, 32)
        var_fn = self.variant(led_source)
        mode_set, wavelengths, spectral_weights = var_fn(
            center_wl, bandwidth, coherence_width, dx, grid_size
        )
        assert hasattr(mode_set, "modes")
        assert hasattr(mode_set, "weights")
        chex.assert_rank(wavelengths, 1)
        chex.assert_rank(spectral_weights, 1)

    @chex.variants(without_jit=True)
    def test_mode_shapes(self) -> None:
        """Test that modes have correct shape."""
        center_wl = 530e-9
        bandwidth = 30e-9
        coherence_width = 50e-6
        dx = 1e-6
        grid_size = (32, 48)
        num_spatial_modes = 8
        var_fn = self.variant(led_source)
        mode_set, _, _ = var_fn(
            center_wl,
            bandwidth,
            coherence_width,
            dx,
            grid_size,
            num_spatial_modes=num_spatial_modes,
        )
        chex.assert_shape(mode_set.modes, (num_spatial_modes, 32, 48))

    @chex.variants(without_jit=True)
    def test_spectral_samples(self) -> None:
        """Test that correct number of spectral samples returned."""
        center_wl = 530e-9
        bandwidth = 30e-9
        coherence_width = 50e-6
        dx = 1e-6
        grid_size = (32, 32)
        num_spectral = 15
        var_fn = self.variant(led_source)
        _, wavelengths, spectral_weights = var_fn(
            center_wl,
            bandwidth,
            coherence_width,
            dx,
            grid_size,
            num_spectral_samples=num_spectral,
        )
        chex.assert_shape(wavelengths, (num_spectral,))
        chex.assert_shape(spectral_weights, (num_spectral,))

    @chex.variants(without_jit=True)
    def test_spectral_weights_normalized(self) -> None:
        """Test that spectral weights sum to 1."""
        center_wl = 530e-9
        bandwidth = 30e-9
        coherence_width = 50e-6
        dx = 1e-6
        grid_size = (32, 32)
        var_fn = self.variant(led_source)
        _, _, spectral_weights = var_fn(
            center_wl, bandwidth, coherence_width, dx, grid_size
        )
        chex.assert_trees_all_close(jnp.sum(spectral_weights), 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_mode_weights_normalized(self) -> None:
        """Test that mode weights sum to 1."""
        center_wl = 530e-9
        bandwidth = 30e-9
        coherence_width = 50e-6
        dx = 1e-6
        grid_size = (32, 32)
        var_fn = self.variant(led_source)
        mode_set, _, _ = var_fn(
            center_wl, bandwidth, coherence_width, dx, grid_size
        )
        chex.assert_trees_all_close(jnp.sum(mode_set.weights), 1.0, atol=1e-10)


class TestThermalSource(chex.TestCase):
    """Test thermal_source function."""

    @chex.variants(without_jit=True)
    def test_output_types(self) -> None:
        """Test that outputs have correct types."""
        temperature = 3000.0
        source_diameter = 5e-3
        propagation_distance = 0.5
        dx = 1e-6
        grid_size = (32, 32)
        wl_range = (500e-9, 1500e-9)
        var_fn = self.variant(thermal_source)
        mode_set, wavelengths, spectral_weights = var_fn(
            temperature,
            source_diameter,
            propagation_distance,
            dx,
            grid_size,
            wl_range,
        )
        assert hasattr(mode_set, "modes")
        assert hasattr(mode_set, "weights")
        chex.assert_rank(wavelengths, 1)
        chex.assert_rank(spectral_weights, 1)

    @chex.variants(without_jit=True)
    def test_mode_shapes(self) -> None:
        """Test that modes have correct shape."""
        temperature = 3000.0
        source_diameter = 5e-3
        propagation_distance = 0.5
        dx = 1e-6
        grid_size = (32, 48)
        wl_range = (500e-9, 1500e-9)
        num_modes = 15
        var_fn = self.variant(thermal_source)
        mode_set, _, _ = var_fn(
            temperature,
            source_diameter,
            propagation_distance,
            dx,
            grid_size,
            wl_range,
            num_modes=num_modes,
        )
        chex.assert_shape(mode_set.modes, (num_modes, 32, 48))

    @chex.variants(without_jit=True)
    def test_wien_wavelength_used(self) -> None:
        """Test that Wien peak wavelength is used when not specified."""
        temperature = 5800.0
        source_diameter = 1e-3
        propagation_distance = 0.1
        dx = 1e-6
        grid_size = (32, 32)
        wl_range = (400e-9, 700e-9)
        var_fn = self.variant(thermal_source)
        mode_set, _, _ = var_fn(
            temperature,
            source_diameter,
            propagation_distance,
            dx,
            grid_size,
            wl_range,
        )
        wien_peak = 2.898e-3 / temperature
        chex.assert_trees_all_close(mode_set.wavelength, wien_peak, rtol=0.01)

    @chex.variants(without_jit=True)
    def test_custom_center_wavelength(self) -> None:
        """Test that custom center wavelength is used."""
        temperature = 3000.0
        source_diameter = 5e-3
        propagation_distance = 0.5
        dx = 1e-6
        grid_size = (32, 32)
        wl_range = (500e-9, 1500e-9)
        center_wl = 800e-9
        var_fn = self.variant(thermal_source)
        mode_set, _, _ = var_fn(
            temperature,
            source_diameter,
            propagation_distance,
            dx,
            grid_size,
            wl_range,
            center_wavelength=center_wl,
        )
        chex.assert_trees_all_close(mode_set.wavelength, center_wl, rtol=1e-10)

    @chex.variants(without_jit=True)
    def test_spectral_weights_normalized(self) -> None:
        """Test that spectral weights sum to 1."""
        temperature = 3000.0
        source_diameter = 5e-3
        propagation_distance = 0.5
        dx = 1e-6
        grid_size = (32, 32)
        wl_range = (500e-9, 1500e-9)
        var_fn = self.variant(thermal_source)
        _, _, spectral_weights = var_fn(
            temperature,
            source_diameter,
            propagation_distance,
            dx,
            grid_size,
            wl_range,
        )
        chex.assert_trees_all_close(jnp.sum(spectral_weights), 1.0, atol=1e-10)


class TestSynchrotronSource(chex.TestCase):
    """Test synchrotron_source function."""

    @chex.variants(without_jit=True)
    def test_output_type(self) -> None:
        """Test that output is a CoherentModeSet."""
        center_wl = 1e-10
        horizontal_coh = 10e-6
        vertical_coh = 50e-6
        dx = 1e-6
        grid_size = (32, 32)
        var_fn = self.variant(synchrotron_source)
        mode_set = var_fn(
            center_wl, horizontal_coh, vertical_coh, dx, grid_size
        )
        assert hasattr(mode_set, "modes")
        assert hasattr(mode_set, "weights")

    @chex.variants(without_jit=True)
    def test_mode_count(self) -> None:
        """Test that correct number of modes generated."""
        center_wl = 1e-10
        horizontal_coh = 10e-6
        vertical_coh = 50e-6
        dx = 1e-6
        grid_size = (32, 32)
        num_modes_h = 4
        num_modes_v = 3
        expected_modes = num_modes_h * num_modes_v
        var_fn = self.variant(synchrotron_source)
        mode_set = var_fn(
            center_wl,
            horizontal_coh,
            vertical_coh,
            dx,
            grid_size,
            num_modes_h=num_modes_h,
            num_modes_v=num_modes_v,
        )
        chex.assert_shape(mode_set.modes, (expected_modes, 32, 32))

    @chex.variants(without_jit=True)
    def test_weights_normalized(self) -> None:
        """Test that weights sum to 1."""
        center_wl = 1e-10
        horizontal_coh = 10e-6
        vertical_coh = 50e-6
        dx = 1e-6
        grid_size = (32, 32)
        var_fn = self.variant(synchrotron_source)
        mode_set = var_fn(
            center_wl, horizontal_coh, vertical_coh, dx, grid_size
        )
        chex.assert_trees_all_close(jnp.sum(mode_set.weights), 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_modes_normalized(self) -> None:
        """Test that each mode has unit energy."""
        center_wl = 1e-10
        horizontal_coh = 10e-6
        vertical_coh = 50e-6
        dx = 1e-6
        grid_size = (32, 32)
        var_fn = self.variant(synchrotron_source)
        mode_set = var_fn(
            center_wl, horizontal_coh, vertical_coh, dx, grid_size
        )
        for i in range(mode_set.modes.shape[0]):
            mode_energy = jnp.sum(jnp.abs(mode_set.modes[i]) ** 2)
            chex.assert_trees_all_close(mode_energy, 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_wavelength_stored(self) -> None:
        """Test that wavelength is correctly stored."""
        center_wl = 1e-10
        horizontal_coh = 10e-6
        vertical_coh = 50e-6
        dx = 1e-6
        grid_size = (32, 32)
        var_fn = self.variant(synchrotron_source)
        mode_set = var_fn(
            center_wl, horizontal_coh, vertical_coh, dx, grid_size
        )
        chex.assert_trees_all_close(mode_set.wavelength, center_wl, rtol=1e-10)


class TestLaserWithModeNoise(chex.TestCase):
    """Test laser_with_mode_noise function."""

    @chex.variants(without_jit=True)
    def test_output_type(self) -> None:
        """Test that output is a CoherentModeSet."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_waist = 20e-6
        mode_purity = 0.9
        var_fn = self.variant(laser_with_mode_noise)
        mode_set = var_fn(wavelength, dx, grid_size, beam_waist, mode_purity)
        assert hasattr(mode_set, "modes")
        assert hasattr(mode_set, "weights")

    @chex.variants(without_jit=True)
    def test_mode_shapes(self) -> None:
        """Test that modes have correct shape."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 48)
        beam_waist = 20e-6
        mode_purity = 0.9
        num_modes = 7
        var_fn = self.variant(laser_with_mode_noise)
        mode_set = var_fn(
            wavelength, dx, grid_size, beam_waist, mode_purity, num_modes
        )
        chex.assert_shape(mode_set.modes, (num_modes, 32, 48))

    @chex.variants(without_jit=True)
    def test_weights_normalized(self) -> None:
        """Test that weights sum to 1."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_waist = 20e-6
        mode_purity = 0.8
        var_fn = self.variant(laser_with_mode_noise)
        mode_set = var_fn(wavelength, dx, grid_size, beam_waist, mode_purity)
        chex.assert_trees_all_close(jnp.sum(mode_set.weights), 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_high_purity_dominant_tem00(self) -> None:
        """Test that high purity gives dominant TEM00 mode."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_waist = 20e-6
        mode_purity = 0.99
        var_fn = self.variant(laser_with_mode_noise)
        mode_set = var_fn(wavelength, dx, grid_size, beam_waist, mode_purity)
        first_mode_weight = mode_set.weights[0]
        assert first_mode_weight > 0.95

    @chex.variants(without_jit=True)
    def test_low_purity_distributed(self) -> None:
        """Test that low purity distributes power across modes."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_waist = 20e-6
        mode_purity = 0.1
        var_fn = self.variant(laser_with_mode_noise)
        mode_set = var_fn(wavelength, dx, grid_size, beam_waist, mode_purity)
        first_mode_weight = mode_set.weights[0]
        assert first_mode_weight < 0.2

    @chex.variants(without_jit=True)
    def test_purity_clipped(self) -> None:
        """Test that purity is clipped to [0, 1]."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_waist = 20e-6
        var_fn = self.variant(laser_with_mode_noise)
        mode_set_high = var_fn(
            wavelength, dx, grid_size, beam_waist, mode_purity=1.5
        )
        mode_set_low = var_fn(
            wavelength, dx, grid_size, beam_waist, mode_purity=-0.5
        )
        chex.assert_trees_all_close(
            mode_set_high.weights[0], 1.0, atol=1e-10
        )
        chex.assert_trees_all_close(
            mode_set_low.weights[0], 0.0, atol=1e-10
        )


class TestMultimodeFiberOutput(chex.TestCase):
    """Test multimode_fiber_output function."""

    @chex.variants(without_jit=True)
    def test_output_type(self) -> None:
        """Test that output is a CoherentModeSet."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        fiber_core_radius = 25e-6
        var_fn = self.variant(multimode_fiber_output)
        mode_set = var_fn(wavelength, dx, grid_size, fiber_core_radius)
        assert hasattr(mode_set, "modes")
        assert hasattr(mode_set, "weights")

    @chex.variants(without_jit=True)
    def test_mode_shapes(self) -> None:
        """Test that modes have correct shape."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 48)
        fiber_core_radius = 25e-6
        num_modes = 8
        var_fn = self.variant(multimode_fiber_output)
        mode_set = var_fn(
            wavelength, dx, grid_size, fiber_core_radius, num_modes=num_modes
        )
        chex.assert_shape(mode_set.modes, (num_modes, 32, 48))

    @chex.variants(without_jit=True)
    def test_uniform_distribution(self) -> None:
        """Test that uniform distribution gives equal weights."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        fiber_core_radius = 25e-6
        num_modes = 5
        var_fn = self.variant(multimode_fiber_output)
        mode_set = var_fn(
            wavelength,
            dx,
            grid_size,
            fiber_core_radius,
            num_modes=num_modes,
            mode_distribution="uniform",
        )
        expected_weight = 1.0 / num_modes
        chex.assert_trees_all_close(
            mode_set.weights, expected_weight, atol=1e-10
        )

    @chex.variants(without_jit=True)
    def test_thermal_distribution(self) -> None:
        """Test that thermal distribution gives decreasing weights."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        fiber_core_radius = 25e-6
        num_modes = 5
        var_fn = self.variant(multimode_fiber_output)
        mode_set = var_fn(
            wavelength,
            dx,
            grid_size,
            fiber_core_radius,
            num_modes=num_modes,
            mode_distribution="thermal",
        )
        weights = mode_set.weights
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]

    @chex.variants(without_jit=True)
    def test_weights_normalized(self) -> None:
        """Test that weights sum to 1."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        fiber_core_radius = 25e-6
        var_fn = self.variant(multimode_fiber_output)
        mode_set = var_fn(wavelength, dx, grid_size, fiber_core_radius)
        chex.assert_trees_all_close(jnp.sum(mode_set.weights), 1.0, atol=1e-10)

    def test_invalid_distribution_raises(self) -> None:
        """Test that invalid distribution raises ValueError."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        fiber_core_radius = 25e-6
        with self.assertRaises(ValueError):
            multimode_fiber_output(
                wavelength,
                dx,
                grid_size,
                fiber_core_radius,
                mode_distribution="invalid",
            )
