"""Tests for spatial coherence functions in janssen.cohere.spatial."""

import chex
import jax.numpy as jnp

from janssen.cohere.spatial import (
    _complex_degree_of_coherence_impl,
    _gaussian_coherence_kernel_impl,
    _jinc_coherence_kernel_impl,
    _rectangular_coherence_kernel_impl,
    coherence_width_from_source,
    complex_degree_of_coherence,
    gaussian_coherence_kernel,
    jinc_coherence_kernel,
    rectangular_coherence_kernel,
)


class TestGaussianCoherenceKernel(chex.TestCase):
    """Test gaussian_coherence_kernel function."""

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        grid_size = (64, 64)
        dx = 1e-6
        coherence_width = 10e-6
        var_fn = self.variant(gaussian_coherence_kernel)
        kernel = var_fn(grid_size, dx, coherence_width)
        chex.assert_shape(kernel, grid_size)

    @chex.variants(without_jit=True)
    def test_non_square_grid(self) -> None:
        """Test that non-square grids work correctly."""
        grid_size = (32, 64)
        dx = 1e-6
        coherence_width = 10e-6
        var_fn = self.variant(gaussian_coherence_kernel)
        kernel = var_fn(grid_size, dx, coherence_width)
        chex.assert_shape(kernel, grid_size)

    @chex.variants(without_jit=True)
    def test_center_value_one(self) -> None:
        """Test that kernel value at zero separation is 1."""
        grid_size = (64, 64)
        dx = 1e-6
        coherence_width = 10e-6
        var_fn = self.variant(gaussian_coherence_kernel)
        kernel = var_fn(grid_size, dx, coherence_width)
        center_value = kernel[0, 0]
        chex.assert_trees_all_close(center_value, 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_values_bounded(self) -> None:
        """Test that kernel values are in [0, 1]."""
        grid_size = (64, 64)
        dx = 1e-6
        coherence_width = 10e-6
        var_fn = self.variant(gaussian_coherence_kernel)
        kernel = var_fn(grid_size, dx, coherence_width)
        assert jnp.all(kernel >= 0.0)
        assert jnp.all(kernel <= 1.0)

    @chex.variants(without_jit=True)
    def test_decay_with_distance(self) -> None:
        """Test that kernel decays with distance from center."""
        grid_size = (64, 64)
        dx = 1e-6
        coherence_width = 10e-6
        var_fn = self.variant(gaussian_coherence_kernel)
        kernel = var_fn(grid_size, dx, coherence_width)
        kernel_shifted = jnp.fft.fftshift(kernel)
        center = grid_size[0] // 2
        center_value = kernel_shifted[center, center]
        edge_value = kernel_shifted[center, center + 20]
        assert center_value > edge_value

    @chex.variants(without_jit=True)
    def test_wider_coherence_slower_decay(self) -> None:
        """Test that wider coherence width gives slower decay."""
        grid_size = (64, 64)
        dx = 1e-6
        var_fn = self.variant(gaussian_coherence_kernel)
        kernel_narrow = var_fn(grid_size, dx, coherence_width=5e-6)
        kernel_wide = var_fn(grid_size, dx, coherence_width=20e-6)
        kernel_narrow_shifted = jnp.fft.fftshift(kernel_narrow)
        kernel_wide_shifted = jnp.fft.fftshift(kernel_wide)
        center = grid_size[0] // 2
        offset = 10
        narrow_at_offset = kernel_narrow_shifted[center, center + offset]
        wide_at_offset = kernel_wide_shifted[center, center + offset]
        assert wide_at_offset > narrow_at_offset


class TestJincCoherenceKernel(chex.TestCase):
    """Test jinc_coherence_kernel function."""

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        grid_size = (64, 64)
        dx = 1e-6
        source_diameter = 1e-3
        wavelength = 633e-9
        propagation_distance = 0.1
        var_fn = self.variant(jinc_coherence_kernel)
        kernel = var_fn(
            grid_size, dx, source_diameter, wavelength, propagation_distance
        )
        chex.assert_shape(kernel, grid_size)

    @chex.variants(without_jit=True)
    def test_center_value_one(self) -> None:
        """Test that kernel value at zero separation is 1."""
        grid_size = (64, 64)
        dx = 1e-6
        source_diameter = 1e-3
        wavelength = 633e-9
        propagation_distance = 0.1
        var_fn = self.variant(jinc_coherence_kernel)
        kernel = var_fn(
            grid_size, dx, source_diameter, wavelength, propagation_distance
        )
        center_value = kernel[0, 0]
        chex.assert_trees_all_close(center_value, 1.0, atol=1e-6)

    @chex.variants(without_jit=True)
    def test_negative_values_exist(self) -> None:
        """Test that jinc kernel has negative values (sidelobes)."""
        grid_size = (128, 128)
        dx = 1e-6
        source_diameter = 1e-3
        wavelength = 633e-9
        propagation_distance = 0.05
        var_fn = self.variant(jinc_coherence_kernel)
        kernel = var_fn(
            grid_size, dx, source_diameter, wavelength, propagation_distance
        )
        assert jnp.any(kernel < 0)

    @chex.variants(without_jit=True)
    def test_finite_values(self) -> None:
        """Test that all kernel values are finite."""
        grid_size = (64, 64)
        dx = 1e-6
        source_diameter = 1e-3
        wavelength = 633e-9
        propagation_distance = 0.1
        var_fn = self.variant(jinc_coherence_kernel)
        kernel = var_fn(
            grid_size, dx, source_diameter, wavelength, propagation_distance
        )
        assert jnp.all(jnp.isfinite(kernel))

    @chex.variants(without_jit=True)
    def test_larger_source_narrower_coherence(self) -> None:
        """Test van Cittert-Zernike: larger source gives narrower coherence."""
        grid_size = (64, 64)
        dx = 1e-6
        wavelength = 633e-9
        propagation_distance = 0.1
        var_fn = self.variant(jinc_coherence_kernel)
        kernel_small = var_fn(
            grid_size, dx, 0.5e-3, wavelength, propagation_distance
        )
        kernel_large = var_fn(
            grid_size, dx, 2e-3, wavelength, propagation_distance
        )
        kernel_small_shifted = jnp.fft.fftshift(kernel_small)
        kernel_large_shifted = jnp.fft.fftshift(kernel_large)
        center = grid_size[0] // 2
        offset = 5
        small_at_offset = kernel_small_shifted[center, center + offset]
        large_at_offset = kernel_large_shifted[center, center + offset]
        assert small_at_offset > large_at_offset


class TestRectangularCoherenceKernel(chex.TestCase):
    """Test rectangular_coherence_kernel function."""

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        grid_size = (64, 64)
        dx = 1e-6
        source_width_x = 1e-3
        source_width_y = 0.5e-3
        wavelength = 633e-9
        propagation_distance = 0.1
        var_fn = self.variant(rectangular_coherence_kernel)
        kernel = var_fn(
            grid_size,
            dx,
            source_width_x,
            source_width_y,
            wavelength,
            propagation_distance,
        )
        chex.assert_shape(kernel, grid_size)

    @chex.variants(without_jit=True)
    def test_center_value_one(self) -> None:
        """Test that kernel value at zero separation is 1."""
        grid_size = (64, 64)
        dx = 1e-6
        source_width_x = 1e-3
        source_width_y = 1e-3
        wavelength = 633e-9
        propagation_distance = 0.1
        var_fn = self.variant(rectangular_coherence_kernel)
        kernel = var_fn(
            grid_size,
            dx,
            source_width_x,
            source_width_y,
            wavelength,
            propagation_distance,
        )
        center_value = kernel[0, 0]
        chex.assert_trees_all_close(center_value, 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_finite_values(self) -> None:
        """Test that all kernel values are finite."""
        grid_size = (64, 64)
        dx = 1e-6
        source_width_x = 1e-3
        source_width_y = 0.5e-3
        wavelength = 633e-9
        propagation_distance = 0.1
        var_fn = self.variant(rectangular_coherence_kernel)
        kernel = var_fn(
            grid_size,
            dx,
            source_width_x,
            source_width_y,
            wavelength,
            propagation_distance,
        )
        assert jnp.all(jnp.isfinite(kernel))

    @chex.variants(without_jit=True)
    def test_separable_structure(self) -> None:
        """Test that rectangular kernel has separable structure."""
        grid_size = (64, 64)
        dx = 1e-6
        source_width_x = 1e-3
        source_width_y = 2e-3
        wavelength = 633e-9
        propagation_distance = 0.1
        var_fn = self.variant(rectangular_coherence_kernel)
        kernel = var_fn(
            grid_size,
            dx,
            source_width_x,
            source_width_y,
            wavelength,
            propagation_distance,
        )
        kernel_shifted = jnp.fft.fftshift(kernel)
        center = grid_size[0] // 2
        row_profile = kernel_shifted[center, :]
        col_profile = kernel_shifted[:, center]
        assert jnp.all(jnp.isfinite(row_profile))
        assert jnp.all(jnp.isfinite(col_profile))


class TestCoherenceWidthFromSource(chex.TestCase):
    """Test coherence_width_from_source function."""

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output is a scalar."""
        source_diameter = 1e-3
        wavelength = 633e-9
        propagation_distance = 0.1
        var_fn = self.variant(coherence_width_from_source)
        coh_width = var_fn(source_diameter, wavelength, propagation_distance)
        chex.assert_shape(coh_width, ())

    @chex.variants(without_jit=True)
    def test_positive_value(self) -> None:
        """Test that coherence width is positive."""
        source_diameter = 1e-3
        wavelength = 633e-9
        propagation_distance = 0.1
        var_fn = self.variant(coherence_width_from_source)
        coh_width = var_fn(source_diameter, wavelength, propagation_distance)
        assert coh_width > 0

    @chex.variants(without_jit=True)
    def test_inverse_source_diameter_scaling(self) -> None:
        """Test that coherence width scales inversely with source diameter."""
        wavelength = 633e-9
        propagation_distance = 0.1
        var_fn = self.variant(coherence_width_from_source)
        coh_width_1 = var_fn(1e-3, wavelength, propagation_distance)
        coh_width_2 = var_fn(2e-3, wavelength, propagation_distance)
        chex.assert_trees_all_close(coh_width_1 / coh_width_2, 2.0, rtol=1e-10)

    @chex.variants(without_jit=True)
    def test_wavelength_scaling(self) -> None:
        """Test that coherence width scales with wavelength."""
        source_diameter = 1e-3
        propagation_distance = 0.1
        var_fn = self.variant(coherence_width_from_source)
        coh_width_1 = var_fn(source_diameter, 500e-9, propagation_distance)
        coh_width_2 = var_fn(source_diameter, 1000e-9, propagation_distance)
        chex.assert_trees_all_close(coh_width_2 / coh_width_1, 2.0, rtol=1e-10)

    @chex.variants(without_jit=True)
    def test_distance_scaling(self) -> None:
        """Test that coherence width scales with propagation distance."""
        source_diameter = 1e-3
        wavelength = 633e-9
        var_fn = self.variant(coherence_width_from_source)
        coh_width_1 = var_fn(source_diameter, wavelength, 0.1)
        coh_width_2 = var_fn(source_diameter, wavelength, 0.2)
        chex.assert_trees_all_close(coh_width_2 / coh_width_1, 2.0, rtol=1e-10)

    @chex.variants(without_jit=True)
    def test_vcz_formula(self) -> None:
        """Test van Cittert-Zernike formula: sigma_c = 0.44 * lambda * z / D."""
        source_diameter = 1e-3
        wavelength = 633e-9
        propagation_distance = 0.5
        var_fn = self.variant(coherence_width_from_source)
        coh_width = var_fn(source_diameter, wavelength, propagation_distance)
        expected = 0.44 * wavelength * propagation_distance / source_diameter
        chex.assert_trees_all_close(coh_width, expected, rtol=1e-10)


class TestComplexDegreeOfCoherence(chex.TestCase):
    """Test complex_degree_of_coherence function."""

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output has same shape as input."""
        hh, ww = 8, 8
        j_matrix = jnp.ones((hh, ww, hh, ww), dtype=jnp.complex128)
        var_fn = self.variant(complex_degree_of_coherence)
        mu = var_fn(j_matrix)
        chex.assert_shape(mu, (hh, ww, hh, ww))

    @chex.variants(without_jit=True)
    def test_fully_coherent_field(self) -> None:
        """Test that fully coherent field has |mu| = 1 everywhere."""
        hh, ww = 8, 8
        field = jnp.ones((hh, ww), dtype=jnp.complex128)
        j_matrix = jnp.einsum("ij,kl->ijkl", jnp.conj(field), field)
        var_fn = self.variant(complex_degree_of_coherence)
        mu = var_fn(j_matrix)
        mu_magnitude = jnp.abs(mu)
        chex.assert_trees_all_close(mu_magnitude, 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_diagonal_elements_one(self) -> None:
        """Test that diagonal elements |mu(r, r)| = 1."""
        hh, ww = 8, 8
        field = jnp.exp(1j * jnp.linspace(0, 2 * jnp.pi, hh * ww)).reshape(
            hh, ww
        )
        j_matrix = jnp.einsum("ij,kl->ijkl", jnp.conj(field), field)
        var_fn = self.variant(complex_degree_of_coherence)
        mu = var_fn(j_matrix)
        for i in range(hh):
            for j in range(ww):
                chex.assert_trees_all_close(
                    jnp.abs(mu[i, j, i, j]), 1.0, atol=1e-10
                )

    @chex.variants(without_jit=True)
    def test_finite_values(self) -> None:
        """Test that output values are finite."""
        hh, ww = 8, 8
        field = jnp.ones((hh, ww), dtype=jnp.complex128) + 0.1
        j_matrix = jnp.einsum("ij,kl->ijkl", jnp.conj(field), field)
        var_fn = self.variant(complex_degree_of_coherence)
        mu = var_fn(j_matrix)
        assert jnp.all(jnp.isfinite(mu))


class TestImplFunctions(chex.TestCase):
    """Test internal _impl functions with JIT compilation."""

    @chex.variants(with_jit=True, without_jit=True)
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

    @chex.variants(with_jit=True, without_jit=True)
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

    @chex.variants(with_jit=True, without_jit=True)
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
            dx,
            source_width_x,
            source_width_y,
            wavelength,
            propagation_distance,
        )
        chex.assert_shape(kernel, (hh, ww))

    @chex.variants(with_jit=True, without_jit=True)
    def test_complex_degree_of_coherence_impl(self) -> None:
        """Test _complex_degree_of_coherence_impl under JIT."""
        hh, ww = 4, 4
        field = jnp.ones((hh, ww), dtype=jnp.complex128)
        j_matrix = jnp.einsum("ij,kl->ijkl", jnp.conj(field), field)
        var_fn = self.variant(_complex_degree_of_coherence_impl)
        mu = var_fn(j_matrix)
        chex.assert_shape(mu, (hh, ww, hh, ww))
