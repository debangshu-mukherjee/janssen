"""Tests for internal _impl functions with JIT compilation.

These tests verify that the pure JAX internal implementations work correctly
under JIT compilation. The _impl functions use static_argnums for grid
dimensions and can be called directly in pure JAX workflows.

Note: These tests use with_jit=True to verify JIT compatibility.
"""

import chex
import jax.numpy as jnp

from janssen.coherence.spatial import (
    _complex_degree_of_coherence_impl,
    _gaussian_coherence_kernel_impl,
    _jinc_coherence_kernel_impl,
    _rectangular_coherence_kernel_impl,
)
from janssen.coherence.temporal import (
    _blackbody_spectrum_impl,
    _gaussian_spectrum_impl,
    _lorentzian_spectrum_impl,
    _rectangular_spectrum_impl,
)
from janssen.coherence.modes import (
    _eigenmode_decomposition_impl,
    _gaussian_schell_model_modes_impl,
    _hermite_gaussian_modes_impl,
    _mutual_intensity_from_modes_impl,
)
from janssen.coherence.sources import (
    _multimode_fiber_output_impl,
    _synchrotron_source_impl,
)


class TestSpatialImplJIT(chex.TestCase):
    """Test spatial coherence _impl functions with JIT."""

    @chex.variants(with_jit=True)
    def test_gaussian_coherence_kernel_impl(self) -> None:
        """Test _gaussian_coherence_kernel_impl under JIT."""
        hh, ww = 32, 32
        dx = jnp.asarray(1e-6, dtype=jnp.float64)
        coherence_width = jnp.asarray(10e-6, dtype=jnp.float64)
        var_fn = self.variant(
            lambda dx, cw: _gaussian_coherence_kernel_impl(hh, ww, dx, cw)
        )
        kernel = var_fn(dx, coherence_width)
        chex.assert_shape(kernel, (hh, ww))
        chex.assert_trees_all_close(kernel[0, 0], 1.0, atol=1e-10)

    @chex.variants(with_jit=True)
    def test_jinc_coherence_kernel_impl(self) -> None:
        """Test _jinc_coherence_kernel_impl under JIT."""
        hh, ww = 32, 32
        dx = jnp.asarray(1e-6, dtype=jnp.float64)
        source_diameter = jnp.asarray(1e-3, dtype=jnp.float64)
        wavelength = jnp.asarray(633e-9, dtype=jnp.float64)
        propagation_distance = jnp.asarray(0.1, dtype=jnp.float64)
        var_fn = self.variant(
            lambda dx, d, wl, z: _jinc_coherence_kernel_impl(
                hh, ww, dx, d, wl, z
            )
        )
        kernel = var_fn(dx, source_diameter, wavelength, propagation_distance)
        chex.assert_shape(kernel, (hh, ww))

    @chex.variants(with_jit=True)
    def test_rectangular_coherence_kernel_impl(self) -> None:
        """Test _rectangular_coherence_kernel_impl under JIT."""
        hh, ww = 32, 32
        dx = jnp.asarray(1e-6, dtype=jnp.float64)
        source_width_x = jnp.asarray(1e-3, dtype=jnp.float64)
        source_width_y = jnp.asarray(1e-3, dtype=jnp.float64)
        wavelength = jnp.asarray(633e-9, dtype=jnp.float64)
        propagation_distance = jnp.asarray(0.1, dtype=jnp.float64)
        var_fn = self.variant(
            lambda dx, wx, wy, wl, z: _rectangular_coherence_kernel_impl(
                hh, ww, dx, wx, wy, wl, z
            )
        )
        kernel = var_fn(
            dx, source_width_x, source_width_y, wavelength, propagation_distance
        )
        chex.assert_shape(kernel, (hh, ww))

    @chex.variants(with_jit=True)
    def test_complex_degree_of_coherence_impl(self) -> None:
        """Test _complex_degree_of_coherence_impl under JIT."""
        hh, ww = 4, 4
        field = jnp.ones((hh, ww), dtype=jnp.complex128)
        j_matrix = jnp.einsum("ij,kl->ijkl", jnp.conj(field), field)
        var_fn = self.variant(_complex_degree_of_coherence_impl)
        mu = var_fn(j_matrix)
        chex.assert_shape(mu, (hh, ww, hh, ww))


class TestTemporalImplJIT(chex.TestCase):
    """Test temporal coherence _impl functions with JIT."""

    @chex.variants(with_jit=True)
    def test_gaussian_spectrum_impl(self) -> None:
        """Test _gaussian_spectrum_impl under JIT."""
        num_wavelengths = 11
        center_wl = jnp.asarray(633e-9, dtype=jnp.float64)
        bandwidth = jnp.asarray(10e-9, dtype=jnp.float64)
        lam_min = jnp.asarray(600e-9, dtype=jnp.float64)
        lam_max = jnp.asarray(666e-9, dtype=jnp.float64)
        var_fn = self.variant(
            lambda c, b, mi, ma: _gaussian_spectrum_impl(
                c, b, num_wavelengths, mi, ma
            )
        )
        wavelengths, weights = var_fn(center_wl, bandwidth, lam_min, lam_max)
        chex.assert_shape(wavelengths, (num_wavelengths,))
        chex.assert_shape(weights, (num_wavelengths,))
        chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-10)

    @chex.variants(with_jit=True)
    def test_lorentzian_spectrum_impl(self) -> None:
        """Test _lorentzian_spectrum_impl under JIT."""
        num_wavelengths = 11
        center_wl = jnp.asarray(633e-9, dtype=jnp.float64)
        bandwidth = jnp.asarray(10e-9, dtype=jnp.float64)
        lam_min = jnp.asarray(600e-9, dtype=jnp.float64)
        lam_max = jnp.asarray(666e-9, dtype=jnp.float64)
        var_fn = self.variant(
            lambda c, b, mi, ma: _lorentzian_spectrum_impl(
                c, b, num_wavelengths, mi, ma
            )
        )
        wavelengths, weights = var_fn(center_wl, bandwidth, lam_min, lam_max)
        chex.assert_shape(wavelengths, (num_wavelengths,))
        chex.assert_shape(weights, (num_wavelengths,))
        chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-10)

    @chex.variants(with_jit=True)
    def test_rectangular_spectrum_impl(self) -> None:
        """Test _rectangular_spectrum_impl under JIT."""
        num_wavelengths = 11
        center_wl = jnp.asarray(633e-9, dtype=jnp.float64)
        bandwidth = jnp.asarray(10e-9, dtype=jnp.float64)
        var_fn = self.variant(
            lambda c, b: _rectangular_spectrum_impl(c, b, num_wavelengths)
        )
        wavelengths, weights = var_fn(center_wl, bandwidth)
        chex.assert_shape(wavelengths, (num_wavelengths,))
        chex.assert_shape(weights, (num_wavelengths,))
        chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-10)

    @chex.variants(with_jit=True)
    def test_blackbody_spectrum_impl(self) -> None:
        """Test _blackbody_spectrum_impl under JIT."""
        num_wavelengths = 11
        temperature = jnp.asarray(5800.0, dtype=jnp.float64)
        lam_min = jnp.asarray(400e-9, dtype=jnp.float64)
        lam_max = jnp.asarray(700e-9, dtype=jnp.float64)
        var_fn = self.variant(
            lambda t, mi, ma: _blackbody_spectrum_impl(
                t, mi, num_wavelengths, ma
            )
        )
        wavelengths, weights = var_fn(temperature, lam_min, lam_max)
        chex.assert_shape(wavelengths, (num_wavelengths,))
        chex.assert_shape(weights, (num_wavelengths,))
        chex.assert_trees_all_close(jnp.sum(weights), 1.0, atol=1e-10)


class TestModesImplJIT(chex.TestCase):
    """Test modes _impl functions with JIT."""

    @chex.variants(with_jit=True)
    def test_hermite_gaussian_modes_impl(self) -> None:
        """Test _hermite_gaussian_modes_impl under JIT."""
        hh, ww = 32, 32
        mode_indices = ((0, 0), (1, 0), (0, 1))
        dx = jnp.asarray(1e-6, dtype=jnp.float64)
        beam_waist = jnp.asarray(10e-6, dtype=jnp.float64)
        var_fn = self.variant(
            lambda dx, w: _hermite_gaussian_modes_impl(
                dx, w, hh, ww, mode_indices
            )
        )
        modes = var_fn(dx, beam_waist)
        chex.assert_shape(modes, (len(mode_indices), hh, ww))

    @chex.variants(with_jit=True)
    def test_gaussian_schell_model_modes_impl(self) -> None:
        """Test _gaussian_schell_model_modes_impl under JIT."""
        hh, ww = 32, 32
        n_modes = 5
        dx = jnp.asarray(1e-6, dtype=jnp.float64)
        beam_width = jnp.asarray(50e-6, dtype=jnp.float64)
        coherence_width = jnp.asarray(20e-6, dtype=jnp.float64)
        var_fn = self.variant(
            lambda dx, bw, cw: _gaussian_schell_model_modes_impl(
                dx, bw, cw, hh, ww, n_modes
            )
        )
        modes, eigenvalues = var_fn(dx, beam_width, coherence_width)
        chex.assert_shape(modes, (n_modes, hh, ww))
        chex.assert_shape(eigenvalues, (n_modes,))

    @chex.variants(with_jit=True)
    def test_eigenmode_decomposition_impl(self) -> None:
        """Test _eigenmode_decomposition_impl under JIT."""
        hh, ww = 8, 8
        n_modes = 3
        field = jnp.ones((hh, ww), dtype=jnp.complex128)
        j_matrix = jnp.einsum("ij,kl->ijkl", jnp.conj(field), field)
        var_fn = self.variant(
            lambda j: _eigenmode_decomposition_impl(j, hh, ww, n_modes)
        )
        modes, eigenvalues = var_fn(j_matrix)
        chex.assert_shape(modes, (n_modes, hh, ww))
        chex.assert_shape(eigenvalues, (n_modes,))

    @chex.variants(with_jit=True)
    def test_mutual_intensity_from_modes_impl(self) -> None:
        """Test _mutual_intensity_from_modes_impl under JIT."""
        hh, ww = 8, 8
        n_modes = 3
        modes = jnp.ones((n_modes, hh, ww), dtype=jnp.complex128)
        weights = jnp.ones(n_modes, dtype=jnp.float64) / n_modes
        var_fn = self.variant(
            lambda m, w: _mutual_intensity_from_modes_impl(m, w, n_modes)
        )
        j_matrix = var_fn(modes, weights)
        chex.assert_shape(j_matrix, (hh, ww, hh, ww))


class TestSourcesImplJIT(chex.TestCase):
    """Test sources _impl functions with JIT."""

    @chex.variants(with_jit=True)
    def test_synchrotron_source_impl(self) -> None:
        """Test _synchrotron_source_impl under JIT."""
        hh, ww = 32, 32
        mode_indices = ((0, 0), (1, 0), (0, 1), (1, 1))
        dx = jnp.asarray(1e-6, dtype=jnp.float64)
        horizontal_coh = jnp.asarray(10e-6, dtype=jnp.float64)
        vertical_coh = jnp.asarray(50e-6, dtype=jnp.float64)
        var_fn = self.variant(
            lambda dx, h, v: _synchrotron_source_impl(
                dx, h, v, hh, ww, mode_indices
            )
        )
        modes, weights = var_fn(dx, horizontal_coh, vertical_coh)
        chex.assert_shape(modes, (len(mode_indices), hh, ww))
        chex.assert_shape(weights, (len(mode_indices),))

    @chex.variants(with_jit=True)
    def test_multimode_fiber_output_impl(self) -> None:
        """Test _multimode_fiber_output_impl under JIT."""
        hh, ww = 32, 32
        n_modes = 5
        mode_indices = ((0, 1), (1, 1), (0, 2), (1, 2), (2, 1))
        dx = jnp.asarray(1e-6, dtype=jnp.float64)
        fiber_core_radius = jnp.asarray(25e-6, dtype=jnp.float64)
        var_fn = self.variant(
            lambda dx, r: _multimode_fiber_output_impl(
                dx, r, hh, ww, n_modes, mode_indices
            )
        )
        modes = var_fn(dx, fiber_core_radius)
        chex.assert_shape(modes, (n_modes, hh, ww))
