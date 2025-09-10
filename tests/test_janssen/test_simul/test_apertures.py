"""Tests for aperture functions in janssen.simul.apertures module."""

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float

from janssen.simul.apertures import (
    _arrayed_grids,
    annular_aperture,
    circular_aperture,
    gaussian_apodizer,
    gaussian_apodizer_elliptical,
    rectangular_aperture,
    supergaussian_apodizer,
    supergaussian_apodizer_elliptical,
    variable_transmission_aperture,
)
from janssen.utils import OpticalWavefront, make_optical_wavefront


class TestArrayedGrids(chex.TestCase):
    """Test the _arrayed_grids helper function."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_grid_dimensions(self) -> None:
        """Test that grids have correct dimensions."""
        hh, ww = 32, 64
        dx = 1e-6
        x0 = jnp.zeros((hh, ww))
        y0 = jnp.zeros((hh, ww))
        var_arrayed_grids = self.variant(_arrayed_grids)
        xx, yy = var_arrayed_grids(x0, y0, dx)
        chex.assert_shape(xx, (hh, ww))
        chex.assert_shape(yy, (hh, ww))

    @chex.variants(with_jit=True, without_jit=True)
    def test_grid_centering(self) -> None:
        """Test that grids are properly centered."""
        hh, ww = 64, 64
        dx = 1e-6
        x0 = jnp.zeros((hh, ww))
        y0 = jnp.zeros((hh, ww))
        var_arrayed_grids = self.variant(_arrayed_grids)
        xx, yy = var_arrayed_grids(x0, y0, dx)
        center_x = xx[hh // 2, ww // 2]
        center_y = yy[hh // 2, ww // 2]
        chex.assert_trees_all_close(center_x, 0.0, atol=dx / 2)
        chex.assert_trees_all_close(center_y, 0.0, atol=dx / 2)

    @chex.variants(with_jit=True, without_jit=True)
    def test_grid_spacing(self) -> None:
        """Test that grid spacing is correct."""
        hh, ww = 32, 32
        dx = 2e-6
        x0 = jnp.zeros((hh, ww))
        y0 = jnp.zeros((hh, ww))
        var_arrayed_grids = self.variant(_arrayed_grids)
        xx, yy = var_arrayed_grids(x0, y0, dx)
        x_spacing = xx[0, 1] - xx[0, 0]
        y_spacing = yy[1, 0] - yy[0, 0]
        chex.assert_trees_all_close(x_spacing, dx, rtol=1e-10)
        chex.assert_trees_all_close(y_spacing, dx, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_scalar_dx(self) -> None:
        """Test that scalar dx works correctly."""
        hh, ww = 32, 32
        dx = 2e-6
        x0 = jnp.zeros((hh, ww))
        y0 = jnp.zeros((hh, ww))
        var_arrayed_grids = self.variant(_arrayed_grids)
        xx, yy = var_arrayed_grids(x0, y0, dx)
        x_spacing = xx[0, 1] - xx[0, 0]
        y_spacing = yy[1, 0] - yy[0, 0]
        chex.assert_trees_all_close(x_spacing, dx, rtol=1e-10)
        chex.assert_trees_all_close(y_spacing, dx, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_array_dx(self) -> None:
        """Test that 2-element array dx works correctly."""
        hh, ww = 32, 32
        dx_val = 2e-6
        dy_val = 3e-6
        dx = jnp.array([dx_val, dy_val])
        x0 = jnp.zeros((hh, ww))
        y0 = jnp.zeros((hh, ww))
        var_arrayed_grids = self.variant(_arrayed_grids)
        xx, yy = var_arrayed_grids(x0, y0, dx)
        x_spacing = xx[0, 1] - xx[0, 0]
        y_spacing = yy[1, 0] - yy[0, 0]
        chex.assert_trees_all_close(x_spacing, dx_val, rtol=1e-10)
        chex.assert_trees_all_close(y_spacing, dy_val, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_matches_meshgrid(self) -> None:
        """Test that _arrayed_grids produces same output as meshgrid."""
        hh, ww = 16, 24
        dx = 1.5e-6
        x0 = jnp.zeros((hh, ww))
        y0 = jnp.zeros((hh, ww))
        var_arrayed_grids = self.variant(_arrayed_grids)
        xx_arrayed, yy_arrayed = var_arrayed_grids(x0, y0, dx)
        x = jnp.arange(-ww // 2, ww // 2) * dx
        y = jnp.arange(-hh // 2, hh // 2) * dx
        xx_mesh, yy_mesh = jnp.meshgrid(x, y)
        chex.assert_trees_all_close(xx_arrayed, xx_mesh, rtol=1e-12)
        chex.assert_trees_all_close(yy_arrayed, yy_mesh, rtol=1e-12)

    @chex.variants(with_jit=True, without_jit=True)
    def test_different_dx_dy_meshgrid(self) -> None:
        """Test that _arrayed_grids with [dx, dy] matches expected meshgrid."""
        hh, ww = 20, 30
        dx_val = 2e-6
        dy_val = 3e-6
        x0 = jnp.zeros((hh, ww))
        y0 = jnp.zeros((hh, ww))
        var_arrayed_grids = self.variant(_arrayed_grids)
        xx_arrayed, yy_arrayed = var_arrayed_grids(
            x0, y0, jnp.array([dx_val, dy_val])
        )
        x = jnp.arange(-ww // 2, ww // 2) * dx_val
        y = jnp.arange(-hh // 2, hh // 2) * dy_val
        xx_mesh, yy_mesh = jnp.meshgrid(x, y)
        chex.assert_trees_all_close(xx_arrayed, xx_mesh, rtol=1e-12)
        chex.assert_trees_all_close(yy_arrayed, yy_mesh, rtol=1e-12)


class TestCircularAperture(chex.TestCase, parameterized.TestCase):
    """Test circular aperture function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.nx = 128
        self.ny = 128
        self.dx = 1e-6
        self.wavelength = 500e-9
        field = jnp.ones((self.ny, self.nx), dtype=complex)
        self.test_wavefront = make_optical_wavefront(
            field=field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_circular_aperture(self) -> None:
        """Test basic circular aperture application."""
        var_circular_aperture = self.variant(circular_aperture)
        diameter = 50e-6
        result = var_circular_aperture(self.test_wavefront, diameter)
        chex.assert_shape(result.field, self.test_wavefront.field.shape)
        chex.assert_trees_all_close(
            result.wavelength, self.test_wavefront.wavelength
        )
        chex.assert_trees_all_close(result.dx, self.test_wavefront.dx)
        chex.assert_trees_all_close(
            result.z_position, self.test_wavefront.z_position
        )
        center_val = result.field[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(jnp.abs(center_val), 1.0, rtol=1e-10)
        corner_val = result.field[0, 0]
        chex.assert_trees_all_close(jnp.abs(corner_val), 0.0, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("small_aperture", 10e-6, 0.5),
        ("medium_aperture", 50e-6, 0.8),
        ("large_aperture", 100e-6, 1.0),
        ("with_attenuation", 40e-6, 0.3),
    )
    def test_circular_aperture_sizes(
        self, diameter: float, transmittivity: float
    ) -> None:
        """Test circular aperture with various sizes and transmittivities."""
        var_circular_aperture = self.variant(circular_aperture)
        result = var_circular_aperture(
            self.test_wavefront, diameter, transmittivity=transmittivity
        )
        center_val = result.field[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(
            jnp.abs(center_val), transmittivity, rtol=1e-10
        )
        r_test = diameter
        idx_test = int(r_test / self.dx)
        if idx_test < self.nx // 2:
            outside_val = result.field[self.ny // 2, self.nx // 2 + idx_test]
            chex.assert_trees_all_close(jnp.abs(outside_val), 0.0, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_circular_aperture_offset(self) -> None:
        """Test circular aperture with offset center."""
        var_circular_aperture = self.variant(circular_aperture)
        diameter = 30e-6
        center = jnp.array([10e-6, -5e-6])
        result = var_circular_aperture(
            self.test_wavefront, diameter, center=center
        )
        offset_x_idx = self.nx // 2 + int(center[0] / self.dx)
        offset_y_idx = self.ny // 2 + int(center[1] / self.dx)
        if 0 <= offset_x_idx < self.nx and 0 <= offset_y_idx < self.ny:
            offset_center = result.field[offset_y_idx, offset_x_idx]
            chex.assert_trees_all_close(
                jnp.abs(offset_center), 1.0, rtol=1e-10
            )

    @chex.variants(with_jit=True, without_jit=True)
    def test_transmittivity_clipping(self) -> None:
        """Test that transmittivity is clipped to [0, 1]."""
        diameter = 50e-6
        var_circular_aperture = self.variant(circular_aperture)
        result_high = var_circular_aperture(
            self.test_wavefront, diameter, transmittivity=2.0
        )
        center_val = result_high.field[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(jnp.abs(center_val), 1.0, rtol=1e-10)
        result_neg = var_circular_aperture(
            self.test_wavefront, diameter, transmittivity=-0.5
        )
        center_val = result_neg.field[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(jnp.abs(center_val), 0.0, atol=1e-10)


class TestRectangularAperture(chex.TestCase, parameterized.TestCase):
    """Test rectangular aperture function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.nx = 128
        self.ny = 128
        self.dx = 1e-6
        self.wavelength = 500e-9
        field = jnp.ones((self.ny, self.nx), dtype=complex)
        self.test_wavefront = make_optical_wavefront(
            field=field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_rectangular_aperture(self) -> None:
        """Test basic rectangular aperture."""
        var_rectangular = self.variant(rectangular_aperture)
        width = 40e-6
        height = 60e-6
        result = var_rectangular(self.test_wavefront, width, height)
        chex.assert_shape(result.field, self.test_wavefront.field.shape)
        center_val = result.field[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(jnp.abs(center_val), 1.0, rtol=1e-10)
        corner_val = result.field[0, 0]
        chex.assert_trees_all_close(jnp.abs(corner_val), 0.0, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("square", 40e-6, 40e-6),
        ("wide_rect", 80e-6, 20e-6),
        ("tall_rect", 20e-6, 80e-6),
        ("with_attenuation", 50e-6, 30e-6),
    )
    def test_rectangular_aperture_shapes(
        self, width: float, height: float
    ) -> None:
        """Test rectangular aperture with various shapes."""
        var_rectangular = self.variant(rectangular_aperture)
        test_width = 50e-6
        transmittivity = 0.7 if width == test_width else 1.0
        result = var_rectangular(
            self.test_wavefront, width, height, transmittivity=transmittivity
        )
        center_val = result.field[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(
            jnp.abs(center_val), transmittivity, rtol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_rectangular_aperture_offset(self) -> None:
        """Test rectangular aperture with offset center."""
        var_rectangular = self.variant(rectangular_aperture)
        width = 30e-6
        height = 40e-6
        center = jnp.array([15e-6, -10e-6])
        result = var_rectangular(
            self.test_wavefront, width, height, center=center
        )
        offset_x_idx = self.nx // 2 + int(center[0] / self.dx)
        offset_y_idx = self.ny // 2 + int(center[1] / self.dx)
        if 0 <= offset_x_idx < self.nx and 0 <= offset_y_idx < self.ny:
            offset_center = result.field[offset_y_idx, offset_x_idx]
            chex.assert_trees_all_close(
                jnp.abs(offset_center), 1.0, rtol=1e-10
            )


class TestAnnularAperture(chex.TestCase, parameterized.TestCase):
    """Test annular aperture function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.nx = 128
        self.ny = 128
        self.dx = 1e-6
        self.wavelength = 500e-9

        field = jnp.ones((self.ny, self.nx), dtype=complex)
        self.test_wavefront = make_optical_wavefront(
            field=field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_annular_aperture(self) -> None:
        """Test basic annular aperture."""
        var_annular = self.variant(annular_aperture)
        inner_diameter = 20e-6
        outer_diameter = 60e-6
        result = var_annular(
            self.test_wavefront, inner_diameter, outer_diameter
        )
        center_val = result.field[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(jnp.abs(center_val), 0.0, atol=1e-10)
        ring_radius = (inner_diameter + outer_diameter) / 4
        ring_idx = int(ring_radius / self.dx)
        if ring_idx < self.nx // 2:
            ring_val = result.field[self.ny // 2, self.nx // 2 + ring_idx]
            chex.assert_trees_all_close(jnp.abs(ring_val), 1.0, rtol=1e-10)
        corner_val = result.field[0, 0]
        chex.assert_trees_all_close(jnp.abs(corner_val), 0.0, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("thin_ring", 40e-6, 50e-6, 1.0),
        ("thick_ring", 20e-6, 80e-6, 1.0),
        ("with_attenuation", 30e-6, 60e-6, 0.5),
    )
    def test_annular_aperture_sizes(
        self, inner_d: float, outer_d: float, transmittivity: float
    ) -> None:
        """Test annular aperture with various ring thicknesses."""
        var_annular = self.variant(annular_aperture)
        result = var_annular(
            self.test_wavefront,
            inner_d,
            outer_d,
            transmittivity=transmittivity,
        )
        ring_radius = (inner_d + outer_d) / 4
        ring_idx = int(ring_radius / self.dx)
        if ring_idx < self.nx // 2:
            ring_val = result.field[self.ny // 2, self.nx // 2 + ring_idx]
            chex.assert_trees_all_close(
                jnp.abs(ring_val), transmittivity, rtol=1e-10
            )


class TestVariableTransmissionAperture(chex.TestCase, parameterized.TestCase):
    """Test variable transmission aperture function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.nx = 64
        self.ny = 64
        self.dx = 1e-6
        self.wavelength = 500e-9
        field = jnp.ones((self.ny, self.nx), dtype=complex)
        self.test_wavefront = make_optical_wavefront(
            field=field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_scalar_transmission(self) -> None:
        """Test with scalar transmission value."""
        var_transmission = self.variant(variable_transmission_aperture)
        transmission = 0.5
        result = var_transmission(self.test_wavefront, transmission)
        chex.assert_trees_all_close(
            jnp.abs(result.field),
            jnp.ones_like(result.field) * 0.5,
            rtol=1e-10,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_array_transmission(self) -> None:
        """Test with array transmission map."""
        var_transmission = self.variant(variable_transmission_aperture)
        x = jnp.linspace(0, 1, self.nx)
        y = jnp.linspace(0, 1, self.ny)
        xx, yy = jnp.meshgrid(x, y)
        transmission_map = xx
        result = var_transmission(self.test_wavefront, transmission_map)
        left_val = jnp.abs(result.field[self.ny // 2, 0])
        right_val = jnp.abs(result.field[self.ny // 2, -1])
        chex.assert_scalar_positive(float(right_val) - float(left_val))

    @chex.variants(with_jit=True, without_jit=True)
    def test_transmission_clipping(self) -> None:
        """Test that transmission values are clipped to [0, 1]."""
        var_transmission = self.variant(variable_transmission_aperture)
        transmission_map = jnp.ones((self.ny, self.nx)) * 2.0
        transmission_map = transmission_map.at[0, 0].set(-1.0)
        result = var_transmission(self.test_wavefront, transmission_map)
        max_val = jnp.max(jnp.abs(result.field))
        min_val = jnp.min(jnp.abs(result.field))
        chex.assert_trees_all_close(max_val, 1.0, rtol=1e-10)
        chex.assert_trees_all_close(min_val, 0.0, atol=1e-10)


class TestGaussianApodizer(chex.TestCase, parameterized.TestCase):
    """Test Gaussian apodizer function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.nx = 128
        self.ny = 128
        self.dx = 1e-6
        self.wavelength = 500e-9

        field = jnp.ones((self.ny, self.nx), dtype=complex)
        self.test_wavefront = make_optical_wavefront(
            field=field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_gaussian_apodizer(self) -> None:
        """Test basic Gaussian apodizer."""
        var_gaussian = self.variant(gaussian_apodizer)
        sigma = 20e-6
        result = var_gaussian(self.test_wavefront, sigma)
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        chex.assert_trees_all_close(center_val, 1.0, rtol=1e-10)
        edge_val = jnp.abs(result.field[self.ny // 2, self.nx - 1])
        chex.assert_scalar_positive(float(center_val) - float(edge_val))
        sigma_idx = int(sigma / self.dx)
        if sigma_idx < self.nx // 2:
            sigma_val = jnp.abs(
                result.field[self.ny // 2, self.nx // 2 + sigma_idx]
            )
            expected = jnp.exp(-0.5)  # Value at 1 sigma
            chex.assert_trees_all_close(sigma_val, expected, rtol=0.1)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("narrow", 10e-6, 1.0),
        ("medium", 30e-6, 1.0),
        ("wide", 50e-6, 1.0),
        ("with_attenuation", 25e-6, 0.8),
    )
    def test_gaussian_widths(
        self, sigma: float, peak_transmittivity: float
    ) -> None:
        """Test Gaussian apodizer with various widths."""
        var_gaussian = self.variant(gaussian_apodizer)
        result = var_gaussian(
            self.test_wavefront, sigma, peak_transmittivity=peak_transmittivity
        )
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        chex.assert_trees_all_close(
            center_val, peak_transmittivity, rtol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_gaussian_offset(self) -> None:
        """Test Gaussian apodizer with offset center."""
        var_gaussian = self.variant(gaussian_apodizer)
        sigma = 25e-6
        center = jnp.array([10e-6, -5e-6])
        result = var_gaussian(self.test_wavefront, sigma, center=center)
        left_val = jnp.abs(result.field[self.ny // 2, 10])
        right_val = jnp.abs(result.field[self.ny // 2, self.nx - 10])
        chex.assert_trees_all_equal(jnp.allclose(left_val, right_val), False)


class TestSuperGaussianApodizer(chex.TestCase, parameterized.TestCase):
    """Test super-Gaussian apodizer function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.nx = 128
        self.ny = 128
        self.dx = 1e-6
        self.wavelength = 500e-9
        field = jnp.ones((self.ny, self.nx), dtype=complex)
        self.test_wavefront = make_optical_wavefront(
            field=field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("gaussian_m1", 20e-6, 1, 1.0),
        ("super_gaussian_m2", 20e-6, 2, 1.0),
        ("super_gaussian_m4", 20e-6, 4, 1.0),
        ("super_gaussian_m8", 20e-6, 8, 1.0),
    )
    def test_supergaussian_orders(
        self, sigma: float, m: int, peak_transmittivity: float
    ) -> None:
        """Test super-Gaussian with various orders."""
        var_supergaussian = self.variant(supergaussian_apodizer)
        result = var_supergaussian(
            self.test_wavefront,
            sigma,
            m,
            peak_transmittivity=peak_transmittivity,
        )
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        chex.assert_trees_all_close(
            center_val, peak_transmittivity, rtol=1e-10
        )
        if m > 1:
            near_center_idx = 2
            near_val = jnp.abs(
                result.field[
                    self.ny // 2 + near_center_idx,
                    self.nx // 2 + near_center_idx,
                ]
            )
            if m >= 4:
                chex.assert_trees_all_close(
                    near_val, peak_transmittivity, rtol=0.2
                )

    @chex.variants(with_jit=True, without_jit=True)
    def test_supergaussian_vs_gaussian(self) -> None:
        """Test that super-Gaussian with m=1 has expected behavior."""
        var_supergaussian = self.variant(supergaussian_apodizer)
        sigma = 25e-6
        super_result = var_supergaussian(self.test_wavefront, sigma, m=1)
        center_val = jnp.abs(super_result.field[self.ny // 2, self.nx // 2])
        chex.assert_trees_all_close(center_val, 1.0, rtol=1e-10)
        edge_val = jnp.abs(super_result.field[self.ny // 2, self.nx - 1])
        chex.assert_scalar_positive(float(center_val) - float(edge_val))


class TestEllipticalApodizers(chex.TestCase, parameterized.TestCase):
    """Test elliptical Gaussian and super-Gaussian apodizers."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.nx = 128
        self.ny = 128
        self.dx = 1e-6
        self.wavelength = 500e-9
        field = jnp.ones((self.ny, self.nx), dtype=complex)
        self.test_wavefront = make_optical_wavefront(
            field=field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_elliptical_gaussian_basic(self) -> None:
        """Test basic elliptical Gaussian apodizer."""
        var_elliptical = self.variant(gaussian_apodizer_elliptical)
        sigma_x = 30e-6
        sigma_y = 20e-6
        result = var_elliptical(self.test_wavefront, sigma_x, sigma_y)
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        chex.assert_trees_all_close(center_val, 1.0, rtol=1e-10)
        x_val = jnp.abs(result.field[self.ny // 2, self.nx // 2 + 20])
        y_val = jnp.abs(result.field[self.ny // 2 + 20, self.nx // 2])
        chex.assert_scalar_positive(float(x_val) - float(y_val))

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("no_rotation", 30e-6, 20e-6, 0.0),
        ("rotation_45deg", 30e-6, 20e-6, jnp.pi / 4),
        ("rotation_90deg", 30e-6, 20e-6, jnp.pi / 2),
        ("rotation_neg45deg", 30e-6, 20e-6, -jnp.pi / 4),
    )
    def test_elliptical_gaussian_rotation(
        self, sigma_x: float, sigma_y: float, theta: float
    ) -> None:
        """Test elliptical Gaussian with rotation."""
        var_elliptical = self.variant(gaussian_apodizer_elliptical)
        result = var_elliptical(
            self.test_wavefront, sigma_x, sigma_y, theta=theta
        )
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        chex.assert_trees_all_close(center_val, 1.0, rtol=1e-10)
        theta_difference = 0.01
        if jnp.abs(theta - jnp.pi / 2) < theta_difference:
            x_val = jnp.abs(result.field[self.ny // 2, self.nx // 2 + 20])
            y_val = jnp.abs(result.field[self.ny // 2 + 20, self.nx // 2])
            chex.assert_scalar_positive(float(y_val) - float(x_val))

    @chex.variants(with_jit=True, without_jit=True)
    def test_elliptical_supergaussian(self) -> None:
        """Test elliptical super-Gaussian apodizer."""
        var_elliptical_super = self.variant(supergaussian_apodizer_elliptical)
        sigma_x = 30e-6
        sigma_y = 20e-6
        m = 4
        result = var_elliptical_super(self.test_wavefront, sigma_x, sigma_y, m)
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        chex.assert_trees_all_close(center_val, 1.0, rtol=1e-10)
        near_center_val = jnp.abs(
            result.field[self.ny // 2 + 2, self.nx // 2 + 2]
        )
        chex.assert_trees_all_close(near_center_val, 1.0, rtol=0.2)

    @chex.variants(with_jit=True, without_jit=True)
    def test_circular_from_elliptical(self) -> None:
        """Test that equal sigmas give circular profile."""
        var_circular = self.variant(gaussian_apodizer)
        var_elliptical = self.variant(gaussian_apodizer_elliptical)
        sigma = 25e-6
        circular_result = var_circular(self.test_wavefront, sigma)
        elliptical_result = var_elliptical(
            self.test_wavefront, sigma, sigma, theta=0.0
        )
        chex.assert_trees_all_close(
            circular_result.field, elliptical_result.field, rtol=1e-5
        )


class TestJAXTransformations(chex.TestCase):
    """Test JAX transformations on aperture functions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.nx = 64
        self.ny = 64
        self.dx = 1e-6
        self.wavelength = 500e-9
        field = jnp.ones((self.ny, self.nx), dtype=complex)
        self.test_wavefront = make_optical_wavefront(
            field=field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_jit_compilation(self) -> None:
        """Test JIT compilation of aperture functions."""
        var_circular_aperture = self.variant(circular_aperture)

        @jax.jit
        def apply_circular(wf: OpticalWavefront) -> OpticalWavefront:
            return var_circular_aperture(wf, 50e-6)

        result_jit = apply_circular(self.test_wavefront)
        result_normal = var_circular_aperture(self.test_wavefront, 50e-6)
        chex.assert_trees_all_close(result_jit.field, result_normal.field)

    @chex.variants(with_jit=True, without_jit=True)
    def test_gradient_computation(self) -> None:
        """Test gradient computation through apertures."""
        var_circular_aperture = self.variant(circular_aperture)

        def loss_fn(diameter: Float[Array, " "]) -> Float[Array, " "]:
            apertured = var_circular_aperture(self.test_wavefront, diameter)
            return jnp.sum(jnp.abs(apertured.field) ** 2)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(jnp.array(50e-6))
        chex.assert_shape(grad, ())
        chex.assert_tree_all_finite(grad)

    @chex.variants(with_jit=True, without_jit=True)
    def test_vmap_apertures(self) -> None:
        """Test vmapping over aperture parameters."""
        var_circular_aperture = self.variant(circular_aperture)
        diameters = jnp.array([20e-6, 40e-6, 60e-6])

        def apply_aperture(d: Float[Array, " "]) -> Complex[Array, "64 64"]:
            result = var_circular_aperture(self.test_wavefront, d)
            return result.field

        vmapped_apply = jax.vmap(apply_aperture)
        batch_results = vmapped_apply(diameters)
        chex.assert_shape(batch_results, (3, self.ny, self.nx))
        total_1 = jnp.sum(jnp.abs(batch_results[0]) ** 2)
        total_2 = jnp.sum(jnp.abs(batch_results[1]) ** 2)
        total_3 = jnp.sum(jnp.abs(batch_results[2]) ** 2)
        chex.assert_scalar_positive(float(total_2) - float(total_1))
        chex.assert_scalar_positive(float(total_3) - float(total_2))

    @chex.variants(with_jit=True, without_jit=True)
    def test_composed_apertures(self) -> None:
        """Test composing multiple aperture functions."""
        var_circular_aperture = self.variant(circular_aperture)
        var_gaussian_apodizer = self.variant(gaussian_apodizer)
        var_variable_transmission = self.variant(
            variable_transmission_aperture
        )

        def complex_aperture(wf: OpticalWavefront) -> OpticalWavefront:
            wf = var_circular_aperture(wf, 80e-6)
            wf = var_gaussian_apodizer(wf, 30e-6)
            return var_variable_transmission(wf, 0.9)

        result = complex_aperture(self.test_wavefront)
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        chex.assert_trees_all_close(center_val, 0.9, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_grad_through_composition(self) -> None:
        """Test gradient computation through composed apertures."""
        var_circular_aperture = self.variant(circular_aperture)
        var_gaussian_apodizer = self.variant(gaussian_apodizer)

        def loss_fn(
            params: Tuple[Float[Array, " "], Float[Array, " "]],
        ) -> Float[Array, " "]:
            diameter, sigma = params
            wf = var_circular_aperture(self.test_wavefront, diameter)
            wf = var_gaussian_apodizer(wf, sigma)
            return jnp.sum(jnp.abs(wf.field) ** 2)

        grad_fn = jax.grad(loss_fn)
        params = (jnp.array(50e-6), jnp.array(25e-6))
        grads = grad_fn(params)
        chex.assert_shape(grads[0], ())
        chex.assert_shape(grads[1], ())
        chex.assert_tree_all_finite(grads)


class TestEdgeCases(chex.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.nx = 32
        self.ny = 32
        self.dx = 1e-6
        self.wavelength = 500e-9
        field = jnp.ones((self.ny, self.nx), dtype=complex)
        self.test_wavefront = make_optical_wavefront(
            field=field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_diameter_aperture(self) -> None:
        """Test aperture with zero diameter."""
        var_circular_aperture = self.variant(circular_aperture)
        result = var_circular_aperture(self.test_wavefront, 0.0)
        total_transmission = jnp.sum(jnp.abs(result.field))
        chex.assert_trees_all_close(total_transmission, 0.0, atol=1.0)

    @chex.variants(with_jit=True, without_jit=True)
    def test_very_large_aperture(self) -> None:
        """Test aperture much larger than field."""
        var_circular_aperture = self.variant(circular_aperture)
        result = var_circular_aperture(self.test_wavefront, 1.0)  # 1 meter
        chex.assert_trees_all_close(
            jnp.abs(result.field), jnp.ones_like(result.field), rtol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_annular_inverted_diameters(self) -> None:
        """Test annular aperture with inner > outer."""
        var_annular_aperture = self.variant(annular_aperture)
        result = var_annular_aperture(self.test_wavefront, 60e-6, 20e-6)
        chex.assert_trees_all_close(
            jnp.abs(result.field), jnp.zeros_like(result.field), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_sigma_gaussian(self) -> None:
        """Test Gaussian with zero or very small sigma."""
        var_gaussian_apodizer = self.variant(gaussian_apodizer)
        result = var_gaussian_apodizer(self.test_wavefront, 1e-9)
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        near_center_val = jnp.abs(result.field[self.ny // 2 + 1, self.nx // 2])
        chex.assert_trees_all_close(center_val, 1.0, rtol=1e-5)
        chex.assert_scalar_positive(float(center_val) - float(near_center_val))

    @chex.variants(with_jit=True, without_jit=True)
    def test_complex_field_preservation(self) -> None:
        """Test that phase information is preserved."""
        var_circular_aperture = self.variant(circular_aperture)
        phase = jnp.linspace(0, 2 * jnp.pi, self.nx * self.ny).reshape(
            self.ny, self.nx
        )
        complex_field = jnp.exp(1j * phase)
        wf = make_optical_wavefront(
            field=complex_field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )
        result = var_circular_aperture(wf, 30e-6)
        center_idx = self.ny // 2, self.nx // 2
        original_phase = jnp.angle(complex_field[center_idx])
        result_phase = jnp.angle(result.field[center_idx])

        chex.assert_trees_all_close(result_phase, original_phase, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
