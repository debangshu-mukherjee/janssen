"""Tests for beam generation functions in janssen.models.beams module."""

# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=too-many-public-methods,no-self-use

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from janssen.models.beams import (
    bessel_beam,
    gaussian_beam,
    hermite_gaussian,
    laguerre_gaussian,
    plane_wave,
)
from janssen.utils import OpticalWavefront


class TestPlaneWave(chex.TestCase, parameterized.TestCase):
    """Test plane_wave function."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.wavelength = 500e-9
        self.dx = 1e-6

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ("square_128", (128, 128)),
        ("square_256", (256, 256)),
        ("non_square_128x256", (128, 256)),
        ("non_square_256x128", (256, 128)),
    )
    def test_plane_wave_shapes(self, grid_size: tuple):
        """Test plane_wave with various grid sizes.

        Note:
            Testing both square and non-square grids to verify
            that the beam generation handles all shapes correctly.
        """
        var_plane_wave = self.variant(plane_wave)
        result = var_plane_wave(
            wavelength=self.wavelength, dx=self.dx, grid_size=grid_size
        )
        chex.assert_equal(isinstance(result, OpticalWavefront), True)
        chex.assert_shape(result.field, grid_size)
        chex.assert_trees_all_close(result.wavelength, self.wavelength)
        chex.assert_trees_all_close(result.dx, self.dx)

    @chex.variants(without_jit=True)
    def test_plane_wave_uniform(self):
        """Test that plane wave has uniform amplitude.

        Note:
            Plane wave should have constant amplitude across the field.
            Default amplitude is 1.0.
        """
        var_plane_wave = self.variant(plane_wave)
        result = var_plane_wave(
            wavelength=self.wavelength, dx=self.dx, grid_size=(128, 128)
        )
        intensities = jnp.abs(result.field) ** 2
        chex.assert_trees_all_close(
            intensities, jnp.ones_like(intensities), rtol=1e-10
        )

    @chex.variants(without_jit=True)
    def test_plane_wave_with_amplitude(self):
        """Test plane wave with custom amplitude.

        Note:
            Verifying that the amplitude parameter scales
            the field correctly.
        """
        var_plane_wave = self.variant(plane_wave)
        amplitude = 2.5
        result = var_plane_wave(
            wavelength=self.wavelength,
            dx=self.dx,
            grid_size=(64, 64),
            amplitude=amplitude,
        )
        field_amplitude = jnp.abs(result.field)
        chex.assert_trees_all_close(
            field_amplitude, jnp.ones_like(field_amplitude) * amplitude, rtol=1e-10
        )


class TestGaussianBeam(chex.TestCase, parameterized.TestCase):
    """Test gaussian_beam function."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.wavelength = 500e-9
        self.dx = 1e-6
        self.w0 = 50e-6

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ("square_128", (128, 128)),
        ("square_256", (256, 256)),
        ("non_square_128x256", (128, 256)),
        ("non_square_256x128", (256, 128)),
        ("non_square_192x256", (192, 256)),
    )
    def test_gaussian_beam_shapes(self, grid_size: tuple):
        """Test gaussian_beam with various grid sizes.

        Note:
            Testing both square and non-square grids to verify
            proper shape handling throughout the computation.
        """
        var_gaussian_beam = self.variant(gaussian_beam)
        result = var_gaussian_beam(
            wavelength=self.wavelength,
            waist_0=self.w0,
            z_from_waist=0.0,
            dx=self.dx,
            grid_size=grid_size,
        )
        chex.assert_equal(isinstance(result, OpticalWavefront), True)
        chex.assert_shape(result.field, grid_size)

    @chex.variants(without_jit=True)
    def test_gaussian_beam_peak_at_center(self):
        """Test that Gaussian beam has peak at center.

        Note:
            The intensity should be maximum at the center
            and decay radially outward.
        """
        var_gaussian_beam = self.variant(gaussian_beam)
        grid_size = (256, 256)
        result = var_gaussian_beam(
            wavelength=self.wavelength,
            waist_0=self.w0,
            z_from_waist=0.0,
            dx=self.dx,
            grid_size=grid_size,
        )
        ny, nx = grid_size
        center_intensity = jnp.abs(result.field[ny // 2, nx // 2]) ** 2
        corner_intensity = jnp.abs(result.field[0, 0]) ** 2
        chex.assert_scalar_positive(
            float(center_intensity) - float(corner_intensity)
        )

    @chex.variants(without_jit=True)
    def test_gaussian_beam_waist(self):
        """Test Gaussian beam waist size.

        Note:
            At distance w0 from center, the intensity should drop
            to approximately 1/e² ≈ 0.135 of the peak value.
        """
        var_gaussian_beam = self.variant(gaussian_beam)
        grid_size = (256, 256)
        result = var_gaussian_beam(
            wavelength=self.wavelength,
            waist_0=self.w0,
            z_from_waist=0.0,
            dx=self.dx,
            grid_size=grid_size,
        )
        ny, nx = grid_size
        center_intensity = jnp.abs(result.field[ny // 2, nx // 2]) ** 2
        w0_idx = int(self.w0 / self.dx)
        if w0_idx < nx // 2:
            w0_intensity = jnp.abs(
                result.field[ny // 2, nx // 2 + w0_idx]
            ) ** 2
            expected_ratio = jnp.exp(-2.0)  # 1/e²
            actual_ratio = w0_intensity / center_intensity
            chex.assert_trees_all_close(
                actual_ratio, expected_ratio, rtol=0.1
            )

    @chex.variants(without_jit=True)
    def test_gaussian_beam_array_grid_size(self):
        """Test gaussian_beam with array grid_size.

        Note:
            grid_size can be provided as a JAX array instead of tuple.
        """
        var_gaussian_beam = self.variant(gaussian_beam)
        grid_size = jnp.array([128, 256], dtype=jnp.int32)
        result = var_gaussian_beam(
            wavelength=self.wavelength,
            waist_0=self.w0,
            z_from_waist=0.0,
            dx=self.dx,
            grid_size=grid_size,
        )
        chex.assert_shape(result.field, (128, 256))


class TestBesselBeam(chex.TestCase, parameterized.TestCase):
    """Test bessel_beam function."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.wavelength = 500e-9
        self.dx = 1e-6
        self.cone_angle = 0.1

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ("square_128", (128, 128)),
        ("square_256", (256, 256)),
        ("non_square_128x256", (128, 256)),
        ("non_square_256x128", (256, 128)),
        ("non_square_192x256", (192, 256)),
    )
    def test_bessel_beam_shapes(self, grid_size: tuple):
        """Test bessel_beam with various grid sizes.

        Note:
            Critical test for non-square grid support.
            Verifies that bessel_j0 and create_spatial_grid
            work correctly with all grid shapes.
        """
        var_bessel_beam = self.variant(bessel_beam)
        result = var_bessel_beam(
            wavelength=self.wavelength,
            cone_angle=self.cone_angle,
            dx=self.dx,
            grid_size=grid_size,
        )
        chex.assert_equal(isinstance(result, OpticalWavefront), True)
        chex.assert_shape(result.field, grid_size)
        chex.assert_trees_all_close(result.wavelength, self.wavelength)
        chex.assert_trees_all_close(result.dx, self.dx)

    @chex.variants(without_jit=True)
    def test_bessel_beam_finite_values(self):
        """Test that Bessel beam has finite field values.

        Note:
            The Bessel beam field should contain no NaN or Inf values.
        """
        var_bessel_beam = self.variant(bessel_beam)
        grid_size = (256, 256)
        result = var_bessel_beam(
            wavelength=self.wavelength,
            cone_angle=self.cone_angle,
            dx=self.dx,
            grid_size=grid_size,
        )
        chex.assert_tree_all_finite(result.field)

    @chex.variants(without_jit=True)
    def test_bessel_beam_array_grid_size(self):
        """Test bessel_beam with array grid_size.

        Note:
            grid_size can be provided as a JAX array instead of tuple.
            This is the critical test case that triggered the original bug fix.
        """
        var_bessel_beam = self.variant(bessel_beam)
        grid_size = jnp.array([128, 256], dtype=jnp.int32)
        result = var_bessel_beam(
            wavelength=self.wavelength,
            cone_angle=self.cone_angle,
            dx=self.dx,
            grid_size=grid_size,
        )
        chex.assert_shape(result.field, (128, 256))
        chex.assert_tree_all_finite(result.field)


class TestHermiteGaussian(chex.TestCase, parameterized.TestCase):
    """Test hermite_gaussian function."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.wavelength = 500e-9
        self.dx = 1e-6
        self.w0 = 50e-6

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ("square_256", (256, 256)),
        ("non_square_128x256", (128, 256)),
        ("non_square_256x192", (256, 192)),
    )
    def test_hermite_gaussian_shapes(self, grid_size: tuple):
        """Test hermite_gaussian with various grid sizes.

        Note:
            Testing non-square grid support for Hermite-Gaussian beams.
        """
        var_hermite_gaussian = self.variant(hermite_gaussian)
        result = var_hermite_gaussian(
            wavelength=self.wavelength,
            waist=self.w0,
            n=1,
            m=1,
            dx=self.dx,
            grid_size=grid_size,
        )
        chex.assert_equal(isinstance(result, OpticalWavefront), True)
        chex.assert_shape(result.field, grid_size)

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ("mode_00", 0, 0),
        ("mode_10", 1, 0),
        ("mode_01", 0, 1),
        ("mode_11", 1, 1),
    )
    def test_hermite_gaussian_modes(self, n: int, m: int):
        """Test Hermite-Gaussian with various mode numbers.

        Note:
            Mode (0, 0) should be equivalent to Gaussian beam.
            Higher modes have more complex intensity patterns.
        """
        var_hermite_gaussian = self.variant(hermite_gaussian)
        grid_size = (256, 256)
        result = var_hermite_gaussian(
            wavelength=self.wavelength,
            waist=self.w0,
            n=n,
            m=m,
            dx=self.dx,
            grid_size=grid_size,
        )
        chex.assert_tree_all_finite(result.field)


class TestLaguerreGaussian(chex.TestCase, parameterized.TestCase):
    """Test laguerre_gaussian function."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.wavelength = 500e-9
        self.dx = 1e-6
        self.w0 = 50e-6

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ("square_256", (256, 256)),
        ("non_square_128x256", (128, 256)),
        ("non_square_256x192", (256, 192)),
    )
    def test_laguerre_gaussian_shapes(self, grid_size: tuple):
        """Test laguerre_gaussian with various grid sizes.

        Note:
            Testing non-square grid support for Laguerre-Gaussian beams.
        """
        var_laguerre_gaussian = self.variant(laguerre_gaussian)
        result = var_laguerre_gaussian(
            wavelength=self.wavelength,
            waist=self.w0,
            p=0,
            l=1,
            dx=self.dx,
            grid_size=grid_size,
        )
        chex.assert_equal(isinstance(result, OpticalWavefront), True)
        chex.assert_shape(result.field, grid_size)

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ("mode_00", 0, 0),
        ("mode_01", 0, 1),
        ("mode_10", 1, 0),
        ("mode_11", 1, 1),
    )
    def test_laguerre_gaussian_modes(self, p: int, l_mode: int):
        """Test Laguerre-Gaussian with various mode numbers.

        Note:
            Mode (0, 0) should be equivalent to Gaussian beam.
            Non-zero l creates vortex beams with phase singularity.
        """
        var_laguerre_gaussian = self.variant(laguerre_gaussian)
        grid_size = (256, 256)
        result = var_laguerre_gaussian(
            wavelength=self.wavelength,
            waist=self.w0,
            p=p,
            l=l_mode,
            dx=self.dx,
            grid_size=grid_size,
        )
        chex.assert_tree_all_finite(result.field)
        if l_mode != 0:
            # Vortex beams should have zero intensity at center
            ny, nx = grid_size
            center_intensity = jnp.abs(result.field[ny // 2, nx // 2]) ** 2
            chex.assert_trees_all_close(center_intensity, 0.0, atol=0.1)


class TestJAXTransformations(chex.TestCase):
    """Test JAX transformations on beam generation functions."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.wavelength = 500e-9
        self.dx = 1e-6
        self.w0 = 50e-6

    def test_gradient_through_beam(self):
        """Test gradient computation through beam generation.

        Note:
            Beam functions should be differentiable with respect
            to parameters like waist_0 for optimization applications.
            Grid size must be static (Python tuple) for JIT compatibility.
        """
        def loss_fn(w0_val):
            beam = gaussian_beam(
                wavelength=self.wavelength,
                waist_0=w0_val,
                z_from_waist=0.0,
                dx=self.dx,
                grid_size=(64, 64),
            )
            return jnp.sum(jnp.abs(beam.field) ** 2)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(jnp.array(50e-6))
        chex.assert_tree_all_finite(grad)
        chex.assert_shape(grad, ())


if __name__ == "__main__":
    pytest.main([__file__])
