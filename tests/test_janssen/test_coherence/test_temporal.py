"""Tests for temporal coherence functions in janssen.coherence.temporal."""

import chex
import jax.numpy as jnp
from absl.testing import parameterized

from janssen.coherence.temporal import (
    bandwidth_from_coherence_length,
    blackbody_spectrum,
    coherence_length,
    coherence_time,
    gaussian_spectrum,
    lorentzian_spectrum,
    rectangular_spectrum,
    spectral_phase_from_dispersion,
)


class TestGaussianSpectrum(chex.TestCase):
    """Test gaussian_spectrum function."""

    @chex.variants(without_jit=True)
    def test_output_shapes(self) -> None:
        """Test that output arrays have correct shapes."""
        center_wl = 633e-9
        bandwidth = 10e-9
        num_wl = 11
        var_fn = self.variant(gaussian_spectrum)
        wavelengths, weights = var_fn(center_wl, bandwidth, num_wl)
        chex.assert_shape(wavelengths, (num_wl,))
        chex.assert_shape(weights, (num_wl,))

    @chex.variants(without_jit=True)
    def test_weights_normalized(self) -> None:
        """Test that spectral weights sum to 1."""
        center_wl = 550e-9
        bandwidth = 20e-9
        num_wl = 21
        var_fn = self.variant(gaussian_spectrum)
        _, weights = var_fn(center_wl, bandwidth, num_wl)
        chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_peak_at_center(self) -> None:
        """Test that peak weight is at center wavelength."""
        center_wl = 633e-9
        bandwidth = 10e-9
        num_wl = 21
        var_fn = self.variant(gaussian_spectrum)
        wavelengths, weights = var_fn(center_wl, bandwidth, num_wl)
        peak_idx = jnp.argmax(weights)
        center_idx = num_wl // 2
        chex.assert_equal(peak_idx, center_idx)

    @chex.variants(without_jit=True)
    def test_wavelength_range_default(self) -> None:
        """Test default wavelength range is 3 sigma on each side."""
        center_wl = 500e-9
        bandwidth = 10e-9
        num_wl = 11
        var_fn = self.variant(gaussian_spectrum)
        wavelengths, _ = var_fn(center_wl, bandwidth, num_wl)
        fwhm_to_sigma = 2.0 * jnp.sqrt(2.0 * jnp.log(2.0))
        sigma = bandwidth / fwhm_to_sigma
        expected_min = center_wl - 3.0 * sigma
        expected_max = center_wl + 3.0 * sigma
        chex.assert_trees_all_close(wavelengths[0], expected_min, rtol=1e-10)
        chex.assert_trees_all_close(wavelengths[-1], expected_max, rtol=1e-10)

    @chex.variants(without_jit=True)
    def test_custom_wavelength_range(self) -> None:
        """Test custom wavelength range is respected."""
        center_wl = 633e-9
        bandwidth = 10e-9
        num_wl = 11
        wl_range = (620e-9, 650e-9)
        var_fn = self.variant(gaussian_spectrum)
        wavelengths, _ = var_fn(center_wl, bandwidth, num_wl, wl_range)
        chex.assert_trees_all_close(wavelengths[0], wl_range[0], rtol=1e-10)
        chex.assert_trees_all_close(wavelengths[-1], wl_range[1], rtol=1e-10)

    @chex.variants(without_jit=True)
    def test_weights_positive(self) -> None:
        """Test that all weights are positive."""
        center_wl = 550e-9
        bandwidth = 30e-9
        num_wl = 31
        var_fn = self.variant(gaussian_spectrum)
        _, weights = var_fn(center_wl, bandwidth, num_wl)
        chex.assert_trees_all_close(
            jnp.all(weights > 0), True, atol=0
        )


class TestLorentzianSpectrum(chex.TestCase):
    """Test lorentzian_spectrum function."""

    @chex.variants(without_jit=True)
    def test_output_shapes(self) -> None:
        """Test that output arrays have correct shapes."""
        center_wl = 633e-9
        bandwidth = 10e-9
        num_wl = 11
        var_fn = self.variant(lorentzian_spectrum)
        wavelengths, weights = var_fn(center_wl, bandwidth, num_wl)
        chex.assert_shape(wavelengths, (num_wl,))
        chex.assert_shape(weights, (num_wl,))

    @chex.variants(without_jit=True)
    def test_weights_normalized(self) -> None:
        """Test that spectral weights sum to 1."""
        center_wl = 550e-9
        bandwidth = 20e-9
        num_wl = 51
        var_fn = self.variant(lorentzian_spectrum)
        _, weights = var_fn(center_wl, bandwidth, num_wl)
        chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_peak_at_center(self) -> None:
        """Test that peak weight is at center wavelength."""
        center_wl = 633e-9
        bandwidth = 10e-9
        num_wl = 21
        var_fn = self.variant(lorentzian_spectrum)
        wavelengths, weights = var_fn(center_wl, bandwidth, num_wl)
        peak_idx = jnp.argmax(weights)
        center_idx = num_wl // 2
        chex.assert_equal(peak_idx, center_idx)

    @chex.variants(without_jit=True)
    def test_broader_tails_than_gaussian(self) -> None:
        """Test that Lorentzian has broader tails than Gaussian."""
        center_wl = 550e-9
        bandwidth = 10e-9
        num_wl = 101
        wl_range = (center_wl - 50e-9, center_wl + 50e-9)
        var_gauss = self.variant(gaussian_spectrum)
        var_lorentz = self.variant(lorentzian_spectrum)
        _, weights_gauss = var_gauss(center_wl, bandwidth, num_wl, wl_range)
        _, weights_lorentz = var_lorentz(center_wl, bandwidth, num_wl, wl_range)
        tail_gauss = weights_gauss[0]
        tail_lorentz = weights_lorentz[0]
        assert tail_lorentz > tail_gauss

    @chex.variants(without_jit=True)
    def test_custom_wavelength_range(self) -> None:
        """Test custom wavelength range is respected."""
        center_wl = 633e-9
        bandwidth = 10e-9
        num_wl = 11
        wl_range = (600e-9, 670e-9)
        var_fn = self.variant(lorentzian_spectrum)
        wavelengths, _ = var_fn(center_wl, bandwidth, num_wl, wl_range)
        chex.assert_trees_all_close(wavelengths[0], wl_range[0], rtol=1e-10)
        chex.assert_trees_all_close(wavelengths[-1], wl_range[1], rtol=1e-10)


class TestRectangularSpectrum(chex.TestCase):
    """Test rectangular_spectrum function."""

    @chex.variants(without_jit=True)
    def test_output_shapes(self) -> None:
        """Test that output arrays have correct shapes."""
        center_wl = 633e-9
        bandwidth = 50e-9
        num_wl = 11
        var_fn = self.variant(rectangular_spectrum)
        wavelengths, weights = var_fn(center_wl, bandwidth, num_wl)
        chex.assert_shape(wavelengths, (num_wl,))
        chex.assert_shape(weights, (num_wl,))

    @chex.variants(without_jit=True)
    def test_weights_normalized(self) -> None:
        """Test that spectral weights sum to 1."""
        center_wl = 550e-9
        bandwidth = 100e-9
        num_wl = 21
        var_fn = self.variant(rectangular_spectrum)
        _, weights = var_fn(center_wl, bandwidth, num_wl)
        chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_uniform_weights(self) -> None:
        """Test that all weights are equal."""
        center_wl = 550e-9
        bandwidth = 100e-9
        num_wl = 21
        var_fn = self.variant(rectangular_spectrum)
        _, weights = var_fn(center_wl, bandwidth, num_wl)
        expected_weight = 1.0 / num_wl
        chex.assert_trees_all_close(weights, expected_weight, rtol=1e-10)

    @chex.variants(without_jit=True)
    def test_wavelength_range(self) -> None:
        """Test that wavelength range spans the bandwidth."""
        center_wl = 633e-9
        bandwidth = 50e-9
        num_wl = 11
        var_fn = self.variant(rectangular_spectrum)
        wavelengths, _ = var_fn(center_wl, bandwidth, num_wl)
        expected_min = center_wl - bandwidth / 2
        expected_max = center_wl + bandwidth / 2
        chex.assert_trees_all_close(wavelengths[0], expected_min, rtol=1e-10)
        chex.assert_trees_all_close(wavelengths[-1], expected_max, rtol=1e-10)


class TestBlackbodySpectrum(chex.TestCase):
    """Test blackbody_spectrum function."""

    @chex.variants(without_jit=True)
    def test_output_shapes(self) -> None:
        """Test that output arrays have correct shapes."""
        temperature = 5800.0
        wl_range = (400e-9, 700e-9)
        num_wl = 31
        var_fn = self.variant(blackbody_spectrum)
        wavelengths, weights = var_fn(temperature, wl_range, num_wl)
        chex.assert_shape(wavelengths, (num_wl,))
        chex.assert_shape(weights, (num_wl,))

    @chex.variants(without_jit=True)
    def test_weights_normalized(self) -> None:
        """Test that spectral weights sum to 1."""
        temperature = 5800.0
        wl_range = (400e-9, 700e-9)
        num_wl = 51
        var_fn = self.variant(blackbody_spectrum)
        _, weights = var_fn(temperature, wl_range, num_wl)
        chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_weights_positive(self) -> None:
        """Test that all weights are positive."""
        temperature = 3000.0
        wl_range = (500e-9, 2000e-9)
        num_wl = 51
        var_fn = self.variant(blackbody_spectrum)
        _, weights = var_fn(temperature, wl_range, num_wl)
        assert jnp.all(weights > 0)

    @chex.variants(without_jit=True)
    def test_wien_peak_visible_sun(self) -> None:
        """Test that Sun-like blackbody peaks in visible range."""
        temperature = 5800.0
        wl_range = (300e-9, 900e-9)
        num_wl = 61
        var_fn = self.variant(blackbody_spectrum)
        wavelengths, weights = var_fn(temperature, wl_range, num_wl)
        peak_idx = jnp.argmax(weights)
        peak_wl = wavelengths[peak_idx]
        wien_peak = 2.898e-3 / temperature
        chex.assert_trees_all_close(peak_wl, wien_peak, rtol=0.1)

    @chex.variants(without_jit=True)
    def test_wavelength_range_respected(self) -> None:
        """Test that specified wavelength range is used."""
        temperature = 5000.0
        wl_range = (450e-9, 650e-9)
        num_wl = 21
        var_fn = self.variant(blackbody_spectrum)
        wavelengths, _ = var_fn(temperature, wl_range, num_wl)
        chex.assert_trees_all_close(wavelengths[0], wl_range[0], rtol=1e-10)
        chex.assert_trees_all_close(wavelengths[-1], wl_range[1], rtol=1e-10)


class TestCoherenceLength(chex.TestCase):
    """Test coherence_length function."""

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output is a scalar."""
        center_wl = 633e-9
        bandwidth = 10e-9
        var_fn = self.variant(coherence_length)
        l_c = var_fn(center_wl, bandwidth)
        chex.assert_shape(l_c, ())

    @chex.variants(without_jit=True)
    def test_narrow_bandwidth_long_coherence(self) -> None:
        """Test that narrow bandwidth gives long coherence length."""
        center_wl = 633e-9
        narrow_bw = 1e-12
        wide_bw = 50e-9
        var_fn = self.variant(coherence_length)
        l_c_narrow = var_fn(center_wl, narrow_bw)
        l_c_wide = var_fn(center_wl, wide_bw)
        assert l_c_narrow > l_c_wide

    @chex.variants(without_jit=True)
    def test_inverse_bandwidth_scaling(self) -> None:
        """Test that coherence length scales inversely with bandwidth."""
        center_wl = 633e-9
        bandwidth1 = 10e-9
        bandwidth2 = 20e-9
        var_fn = self.variant(coherence_length)
        l_c_1 = var_fn(center_wl, bandwidth1)
        l_c_2 = var_fn(center_wl, bandwidth2)
        chex.assert_trees_all_close(l_c_1 / l_c_2, 2.0, rtol=1e-10)

    @chex.variants(without_jit=True)
    def test_wavelength_squared_scaling(self) -> None:
        """Test that coherence length scales with wavelength squared."""
        center_wl_1 = 500e-9
        center_wl_2 = 1000e-9
        bandwidth = 10e-9
        var_fn = self.variant(coherence_length)
        l_c_1 = var_fn(center_wl_1, bandwidth)
        l_c_2 = var_fn(center_wl_2, bandwidth)
        chex.assert_trees_all_close(l_c_2 / l_c_1, 4.0, rtol=1e-10)

    @chex.variants(without_jit=True)
    def test_led_coherence_length(self) -> None:
        """Test typical LED coherence length (few micrometers)."""
        center_wl = 550e-9
        bandwidth = 50e-9
        var_fn = self.variant(coherence_length)
        l_c = var_fn(center_wl, bandwidth)
        assert 1e-6 < l_c < 20e-6


class TestCoherenceTime(chex.TestCase):
    """Test coherence_time function."""

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output is a scalar."""
        center_wl = 633e-9
        bandwidth = 10e-9
        var_fn = self.variant(coherence_time)
        tau_c = var_fn(center_wl, bandwidth)
        chex.assert_shape(tau_c, ())

    @chex.variants(without_jit=True)
    def test_relation_to_coherence_length(self) -> None:
        """Test that tau_c = L_c / c."""
        center_wl = 633e-9
        bandwidth = 10e-9
        c_light = 299792458.0
        var_length = self.variant(coherence_length)
        var_time = self.variant(coherence_time)
        l_c = var_length(center_wl, bandwidth)
        tau_c = var_time(center_wl, bandwidth)
        chex.assert_trees_all_close(tau_c, l_c / c_light, rtol=1e-10)

    @chex.variants(without_jit=True)
    def test_positive_value(self) -> None:
        """Test that coherence time is positive."""
        center_wl = 550e-9
        bandwidth = 30e-9
        var_fn = self.variant(coherence_time)
        tau_c = var_fn(center_wl, bandwidth)
        assert tau_c > 0


class TestBandwidthFromCoherenceLength(chex.TestCase):
    """Test bandwidth_from_coherence_length function."""

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output is a scalar."""
        center_wl = 633e-9
        coh_length = 10e-6
        var_fn = self.variant(bandwidth_from_coherence_length)
        bw = var_fn(center_wl, coh_length)
        chex.assert_shape(bw, ())

    @chex.variants(without_jit=True)
    def test_inverse_of_coherence_length(self) -> None:
        """Test that this function inverts coherence_length."""
        center_wl = 633e-9
        original_bandwidth = 15e-9
        var_coh_len = self.variant(coherence_length)
        var_bw = self.variant(bandwidth_from_coherence_length)
        l_c = var_coh_len(center_wl, original_bandwidth)
        recovered_bandwidth = var_bw(center_wl, l_c)
        chex.assert_trees_all_close(
            recovered_bandwidth, original_bandwidth, rtol=1e-10
        )

    @chex.variants(without_jit=True)
    def test_positive_value(self) -> None:
        """Test that bandwidth is positive."""
        center_wl = 550e-9
        coh_length = 5e-6
        var_fn = self.variant(bandwidth_from_coherence_length)
        bw = var_fn(center_wl, coh_length)
        assert bw > 0


class TestSpectralPhaseFromDispersion(chex.TestCase):
    """Test spectral_phase_from_dispersion function."""

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output has same shape as input wavelengths."""
        wavelengths = jnp.linspace(780e-9, 820e-9, 21)
        center_wl = 800e-9
        var_fn = self.variant(spectral_phase_from_dispersion)
        phase = var_fn(wavelengths, center_wl, gdd=100e-30)
        chex.assert_shape(phase, wavelengths.shape)

    @chex.variants(without_jit=True)
    def test_zero_dispersion_zero_phase(self) -> None:
        """Test that zero GDD and TOD gives zero phase."""
        wavelengths = jnp.linspace(780e-9, 820e-9, 21)
        center_wl = 800e-9
        var_fn = self.variant(spectral_phase_from_dispersion)
        phase = var_fn(wavelengths, center_wl, gdd=0.0, tod=0.0)
        chex.assert_trees_all_close(phase, 0.0, atol=1e-20)

    @chex.variants(without_jit=True)
    def test_phase_zero_at_center(self) -> None:
        """Test that phase is zero at center wavelength."""
        num_wl = 21
        center_wl = 800e-9
        wavelengths = jnp.linspace(780e-9, 820e-9, num_wl)
        var_fn = self.variant(spectral_phase_from_dispersion)
        phase = var_fn(wavelengths, center_wl, gdd=100e-30, tod=50e-45)
        center_idx = num_wl // 2
        chex.assert_trees_all_close(phase[center_idx], 0.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_gdd_gives_quadratic_phase(self) -> None:
        """Test that GDD produces quadratic spectral phase."""
        wavelengths = jnp.linspace(790e-9, 810e-9, 21)
        center_wl = 800e-9
        gdd = 100e-30
        var_fn = self.variant(spectral_phase_from_dispersion)
        phase = var_fn(wavelengths, center_wl, gdd=gdd, tod=0.0)
        c_light = 299792458.0
        omega = 2.0 * jnp.pi * c_light / wavelengths
        omega0 = 2.0 * jnp.pi * c_light / center_wl
        delta_omega = omega - omega0
        expected_phase = 0.5 * gdd * delta_omega**2
        chex.assert_trees_all_close(phase, expected_phase, rtol=1e-10)

    @chex.variants(without_jit=True)
    def test_finite_values(self) -> None:
        """Test that phase values are finite."""
        wavelengths = jnp.linspace(700e-9, 900e-9, 51)
        center_wl = 800e-9
        var_fn = self.variant(spectral_phase_from_dispersion)
        phase = var_fn(wavelengths, center_wl, gdd=500e-30, tod=100e-45)
        assert jnp.all(jnp.isfinite(phase))
