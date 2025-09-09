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

        # Create test wavefront
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
        var_circular = self.variant(circular_aperture)
        diameter = 50e-6

        result = var_circular(self.test_wavefront, diameter)

        # Check shape preservation
        chex.assert_shape(result.field, self.test_wavefront.field.shape)

        # Check metadata preservation
        chex.assert_trees_all_close(
            result.wavelength, self.test_wavefront.wavelength
        )
        chex.assert_trees_all_close(result.dx, self.test_wavefront.dx)
        chex.assert_trees_all_close(
            result.z_position, self.test_wavefront.z_position
        )

        # Check that center is transmitted
        center_val = result.field[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(jnp.abs(center_val), 1.0, rtol=1e-10)

        # Check that corners are blocked
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
        var_circular = self.variant(circular_aperture)

        result = var_circular(
            self.test_wavefront, diameter, transmittivity=transmittivity
        )

        # Center should have the specified transmittivity
        center_val = result.field[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(
            jnp.abs(center_val), transmittivity, rtol=1e-10
        )

        # Outside the aperture should be zero
        # Find a point definitely outside
        r_test = diameter  # Double the radius
        idx_test = int(r_test / self.dx)
        if idx_test < self.nx // 2:
            outside_val = result.field[self.ny // 2, self.nx // 2 + idx_test]
            chex.assert_trees_all_close(jnp.abs(outside_val), 0.0, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_circular_aperture_offset(self) -> None:
        """Test circular aperture with offset center."""
        var_circular = self.variant(circular_aperture)
        diameter = 30e-6
        center = jnp.array([10e-6, -5e-6])

        result = var_circular(self.test_wavefront, diameter, center=center)

        # Check that the aperture is offset
        # Original center should be less transmitted
        original_center = result.field[self.ny // 2, self.nx // 2]

        # Offset center should be fully transmitted
        offset_x_idx = self.nx // 2 + int(center[0] / self.dx)
        offset_y_idx = self.ny // 2 + int(center[1] / self.dx)
        if 0 <= offset_x_idx < self.nx and 0 <= offset_y_idx < self.ny:
            offset_center = result.field[offset_y_idx, offset_x_idx]
            chex.assert_scalar_positive(
                float(jnp.abs(offset_center)) - float(jnp.abs(original_center))
            )

    def test_transmittivity_clipping(self) -> None:
        """Test that transmittivity is clipped to [0, 1]."""
        diameter = 50e-6

        # Test over-unity transmittivity
        result_high = circular_aperture(
            self.test_wavefront, diameter, transmittivity=2.0
        )
        center_val = result_high.field[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(jnp.abs(center_val), 1.0, rtol=1e-10)

        # Test negative transmittivity
        result_neg = circular_aperture(
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

        # Check shape preservation
        chex.assert_shape(result.field, self.test_wavefront.field.shape)

        # Center should be transmitted
        center_val = result.field[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(jnp.abs(center_val), 1.0, rtol=1e-10)

        # Corners should be blocked
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
        transmittivity = 0.7 if width == 50e-6 else 1.0

        result = var_rectangular(
            self.test_wavefront, width, height, transmittivity=transmittivity
        )

        # Center should have correct transmittivity
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

        # Check asymmetry due to offset
        left_val = result.field[self.ny // 2, 10]
        right_val = result.field[self.ny // 2, self.nx - 10]
        chex.assert_trees_all_equal(
            jnp.allclose(jnp.abs(left_val), jnp.abs(right_val)), False
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

        # Center should be blocked (inside inner diameter)
        center_val = result.field[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(jnp.abs(center_val), 0.0, atol=1e-10)

        # Ring region should be transmitted
        ring_radius = (inner_diameter + outer_diameter) / 4  # Middle of ring
        ring_idx = int(ring_radius / self.dx)
        if ring_idx < self.nx // 2:
            ring_val = result.field[self.ny // 2, self.nx // 2 + ring_idx]
            chex.assert_trees_all_close(jnp.abs(ring_val), 1.0, rtol=1e-10)

        # Outside should be blocked
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

        # Check ring region
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

        # All points should have same attenuation
        chex.assert_trees_all_close(
            jnp.abs(result.field),
            jnp.ones_like(result.field) * 0.5,
            rtol=1e-10,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_array_transmission(self) -> None:
        """Test with array transmission map."""
        var_transmission = self.variant(variable_transmission_aperture)

        # Create a gradient transmission map
        x = jnp.linspace(0, 1, self.nx)
        y = jnp.linspace(0, 1, self.ny)
        xx, yy = jnp.meshgrid(x, y)
        transmission_map = xx  # Gradient along x

        result = var_transmission(self.test_wavefront, transmission_map)

        # Check that transmission varies along x
        left_val = jnp.abs(result.field[self.ny // 2, 0])
        right_val = jnp.abs(result.field[self.ny // 2, -1])
        chex.assert_scalar_positive(float(right_val) - float(left_val))

    @chex.variants(with_jit=True, without_jit=True)
    def test_transmission_clipping(self) -> None:
        """Test that transmission values are clipped to [0, 1]."""
        var_transmission = self.variant(variable_transmission_aperture)

        # Create transmission map with out-of-range values
        transmission_map = jnp.ones((self.ny, self.nx)) * 2.0
        transmission_map = transmission_map.at[0, 0].set(-1.0)

        result = var_transmission(self.test_wavefront, transmission_map)

        # Check clipping
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

        # Center should have maximum transmission
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        chex.assert_trees_all_close(center_val, 1.0, rtol=1e-10)

        # Transmission should decrease away from center
        edge_val = jnp.abs(result.field[self.ny // 2, self.nx - 1])
        chex.assert_scalar_positive(float(center_val) - float(edge_val))

        # Should be smooth (Gaussian profile)
        # Check 1-sigma point
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

        # Center should have peak transmittivity
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

        # Check asymmetry due to offset
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

        # Center should have peak transmittivity
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        chex.assert_trees_all_close(
            center_val, peak_transmittivity, rtol=1e-10
        )

        # Higher m should give flatter top
        if m > 1:
            # Check that near-center is also close to peak
            near_center_idx = 2
            near_val = jnp.abs(
                result.field[
                    self.ny // 2 + near_center_idx,
                    self.nx // 2 + near_center_idx,
                ]
            )
            # For higher m, near-center should be closer to peak
            if m >= 4:
                chex.assert_trees_all_close(
                    near_val, peak_transmittivity, rtol=0.2
                )

    @chex.variants(with_jit=True, without_jit=True)
    def test_supergaussian_vs_gaussian(self) -> None:
        """Test that m=1 gives same result as Gaussian apodizer."""
        var_gaussian = self.variant(gaussian_apodizer)
        var_supergaussian = self.variant(supergaussian_apodizer)
        sigma = 25e-6

        gauss_result = var_gaussian(self.test_wavefront, sigma)
        super_result = var_supergaussian(self.test_wavefront, sigma, m=1)

        # Results should be very close
        chex.assert_trees_all_close(
            gauss_result.field, super_result.field, rtol=1e-5
        )


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

        # Center should have maximum transmission
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        chex.assert_trees_all_close(center_val, 1.0, rtol=1e-10)

        # Check elliptical shape - should be wider in x than y
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

        # Center should always have maximum transmission
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        chex.assert_trees_all_close(center_val, 1.0, rtol=1e-10)

        # For 90 degree rotation, axes should be swapped
        if jnp.abs(theta - jnp.pi / 2) < 0.01:
            x_val = jnp.abs(result.field[self.ny // 2, self.nx // 2 + 20])
            y_val = jnp.abs(result.field[self.ny // 2 + 20, self.nx // 2])
            # Now y should be wider than x (opposite of no rotation)
            chex.assert_scalar_positive(float(y_val) - float(x_val))

    @chex.variants(with_jit=True, without_jit=True)
    def test_elliptical_supergaussian(self) -> None:
        """Test elliptical super-Gaussian apodizer."""
        var_elliptical_super = self.variant(supergaussian_apodizer_elliptical)
        sigma_x = 30e-6
        sigma_y = 20e-6
        m = 4

        result = var_elliptical_super(self.test_wavefront, sigma_x, sigma_y, m)

        # Center should have maximum transmission
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        chex.assert_trees_all_close(center_val, 1.0, rtol=1e-10)

        # Should have flatter top due to high m
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

        # Results should be very close
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

    def test_jit_compilation(self) -> None:
        """Test JIT compilation of aperture functions."""

        @jax.jit
        def apply_circular(wf: OpticalWavefront) -> OpticalWavefront:
            return circular_aperture(wf, 50e-6)

        result_jit = apply_circular(self.test_wavefront)
        result_normal = circular_aperture(self.test_wavefront, 50e-6)

        chex.assert_trees_all_close(result_jit.field, result_normal.field)

    def test_gradient_computation(self) -> None:
        """Test gradient computation through apertures."""

        def loss_fn(diameter: Float[Array, " "]) -> Float[Array, " "]:
            apertured = circular_aperture(self.test_wavefront, diameter)
            return jnp.sum(jnp.abs(apertured.field) ** 2)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(jnp.array(50e-6))

        chex.assert_shape(grad, ())
        chex.assert_tree_all_finite(grad)

    def test_vmap_apertures(self) -> None:
        """Test vmapping over aperture parameters."""
        diameters = jnp.array([20e-6, 40e-6, 60e-6])

        def apply_aperture(d: Float[Array, " "]) -> Complex[Array, "64 64"]:
            result = circular_aperture(self.test_wavefront, d)
            return result.field

        vmapped_apply = jax.vmap(apply_aperture)
        batch_results = vmapped_apply(diameters)

        chex.assert_shape(batch_results, (3, self.ny, self.nx))

        # Larger apertures should transmit more light
        total_1 = jnp.sum(jnp.abs(batch_results[0]) ** 2)
        total_2 = jnp.sum(jnp.abs(batch_results[1]) ** 2)
        total_3 = jnp.sum(jnp.abs(batch_results[2]) ** 2)
        chex.assert_scalar_positive(float(total_2) - float(total_1))
        chex.assert_scalar_positive(float(total_3) - float(total_2))

    def test_composed_apertures(self) -> None:
        """Test composing multiple aperture functions."""

        @jax.jit
        def complex_aperture(wf: OpticalWavefront) -> OpticalWavefront:
            # Apply circular aperture
            wf = circular_aperture(wf, 80e-6)
            # Then apply Gaussian apodization
            wf = gaussian_apodizer(wf, 30e-6)
            # Finally apply slight attenuation
            wf = variable_transmission_aperture(wf, 0.9)
            return wf

        result = complex_aperture(self.test_wavefront)

        # Check that all transforms were applied
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        # Should be less than 0.9 due to composed effects
        chex.assert_scalar_positive(0.9 - float(center_val))

    def test_grad_through_composition(self) -> None:
        """Test gradient computation through composed apertures."""

        def loss_fn(
            params: Tuple[Float[Array, " "], Float[Array, " "]],
        ) -> Float[Array, " "]:
            diameter, sigma = params
            wf = circular_aperture(self.test_wavefront, diameter)
            wf = gaussian_apodizer(wf, sigma)
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

    def test_zero_diameter_aperture(self) -> None:
        """Test aperture with zero diameter."""
        result = circular_aperture(self.test_wavefront, 0.0)

        # Should block everything
        chex.assert_trees_all_close(
            jnp.abs(result.field), jnp.zeros_like(result.field), atol=1e-10
        )

    def test_very_large_aperture(self) -> None:
        """Test aperture much larger than field."""
        result = circular_aperture(self.test_wavefront, 1.0)  # 1 meter

        # Should transmit everything
        chex.assert_trees_all_close(
            jnp.abs(result.field), jnp.ones_like(result.field), rtol=1e-10
        )

    def test_annular_inverted_diameters(self) -> None:
        """Test annular aperture with inner > outer."""
        result = annular_aperture(self.test_wavefront, 60e-6, 20e-6)

        # Should block everything (invalid ring)
        chex.assert_trees_all_close(
            jnp.abs(result.field), jnp.zeros_like(result.field), atol=1e-10
        )

    def test_zero_sigma_gaussian(self) -> None:
        """Test Gaussian with zero or very small sigma."""
        # Very small sigma should create sharp peak
        result = gaussian_apodizer(self.test_wavefront, 1e-9)

        # Center should still be 1, but drops off very quickly
        center_val = jnp.abs(result.field[self.ny // 2, self.nx // 2])
        near_center_val = jnp.abs(result.field[self.ny // 2 + 1, self.nx // 2])

        chex.assert_trees_all_close(center_val, 1.0, rtol=1e-5)
        chex.assert_scalar_positive(float(center_val) - float(near_center_val))

    def test_complex_field_preservation(self) -> None:
        """Test that phase information is preserved."""
        # Create field with phase variation
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

        result = circular_aperture(wf, 30e-6)

        # Check that phase is preserved where transmitted
        center_idx = self.ny // 2, self.nx // 2
        original_phase = jnp.angle(complex_field[center_idx])
        result_phase = jnp.angle(result.field[center_idx])

        chex.assert_trees_all_close(result_phase, original_phase, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
