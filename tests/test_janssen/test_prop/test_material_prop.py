import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from janssen.prop import (
    multislice_propagation,
    optical_path_length,
    total_transmit,
)
from janssen.utils import (
    OpticalWavefront,
    SlicedMaterialFunction,
    make_optical_wavefront,
    make_sliced_material_function,
)


class TestMultislicePropagation(chex.TestCase, parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.nx = 64
        self.ny = 64
        self.nz = 10
        self.dx = 1e-6
        self.tz = 5e-6
        self.wavelength = 500e-9

        x = jnp.arange(-self.nx // 2, self.nx // 2) * self.dx
        y = jnp.arange(-self.ny // 2, self.ny // 2) * self.dx
        self.xx, self.yy = jnp.meshgrid(x, y, indexing="ij")
        self.r = jnp.sqrt(self.xx**2 + self.yy**2)

        sigma = 10 * self.dx
        gaussian_field = jnp.exp(-(self.r**2) / (2 * sigma**2))

        self.test_wavefront = make_optical_wavefront(
            field=gaussian_field.astype(complex),
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )

        self.plane_wave = make_optical_wavefront(
            field=jnp.ones((self.ny, self.nx), dtype=complex),
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_multislice_uniform_glass(self) -> None:
        """Test propagation through uniform glass material."""
        var_make_material = self.variant(make_sliced_material_function)
        var_multislice = self.variant(multislice_propagation)

        glass_n = 1.5 + 0.0j
        material_array = (
            jnp.ones((self.ny, self.nx, self.nz), dtype=jnp.complex128)
            * glass_n
        )

        material = var_make_material(
            material=material_array, dx=self.dx, tz=self.tz
        )

        output = var_multislice(self.test_wavefront, material)

        chex.assert_shape(output.field, self.test_wavefront.field.shape)
        chex.assert_trees_all_close(
            output.wavelength, self.test_wavefront.wavelength
        )
        chex.assert_trees_all_close(output.dx, self.test_wavefront.dx)

        expected_z = self.test_wavefront.z_position + self.nz * self.tz
        chex.assert_trees_all_close(output.z_position, expected_z, rtol=1e-10)

        chex.assert_trees_all_equal(
            jnp.allclose(output.field, self.test_wavefront.field), False
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_multislice_vacuum(self) -> None:
        """Test that vacuum (n=1) produces similar results to free-space."""
        var_make_material = self.variant(make_sliced_material_function)
        var_multislice = self.variant(multislice_propagation)

        vacuum_material = jnp.ones(
            (self.ny, self.nx, self.nz), dtype=jnp.complex128
        )
        material = var_make_material(
            material=vacuum_material, dx=self.dx, tz=self.tz
        )

        output = var_multislice(self.test_wavefront, material)

        chex.assert_shape(output.field, self.test_wavefront.field.shape)

        expected_z = self.test_wavefront.z_position + self.nz * self.tz
        chex.assert_trees_all_close(output.z_position, expected_z, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_multislice_with_absorption(self) -> None:
        """Test propagation through absorbing material."""
        var_make_material = self.variant(make_sliced_material_function)
        var_multislice = self.variant(multislice_propagation)

        absorbing_material = jnp.ones(
            (self.ny, self.nx, self.nz), dtype=jnp.complex128
        ) * (1.5 + 0.01j)

        material = var_make_material(
            material=absorbing_material, dx=self.dx, tz=self.tz
        )

        output = var_multislice(self.plane_wave, material)

        output_intensity = jnp.sum(jnp.abs(output.field) ** 2)
        input_intensity = jnp.sum(jnp.abs(self.plane_wave.field) ** 2)

        chex.assert_trees_all_equal(output_intensity < input_intensity, True)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("thin_slices", 5, 2e-6),
        ("medium_slices", 10, 5e-6),
        ("thick_slices", 20, 10e-6),
    )
    def test_multislice_different_slice_counts(
        self, nz: int, tz: float
    ) -> None:
        """Test propagation with different slice configurations."""
        var_make_material = self.variant(make_sliced_material_function)
        var_multislice = self.variant(multislice_propagation)

        material_array = (
            jnp.ones((self.ny, self.nx, nz), dtype=jnp.complex128) * 1.5
        )

        material = var_make_material(
            material=material_array, dx=self.dx, tz=tz
        )

        output = var_multislice(self.test_wavefront, material)

        expected_z = self.test_wavefront.z_position + nz * tz
        chex.assert_trees_all_close(output.z_position, expected_z, rtol=1e-9)
        chex.assert_shape(output.field, self.test_wavefront.field.shape)

    @chex.variants(with_jit=True, without_jit=True)
    def test_multislice_spatially_varying_material(self) -> None:
        """Test propagation through spatially varying material."""
        var_make_material = self.variant(make_sliced_material_function)
        var_multislice = self.variant(multislice_propagation)

        x = jnp.arange(self.nx) - self.nx / 2
        y = jnp.arange(self.ny) - self.ny / 2
        xx, yy = jnp.meshgrid(x, y, indexing="ij")
        r2d = jnp.sqrt(xx**2 + yy**2)

        varying_n = 1.5 + 0.1 * jnp.exp(-(r2d**2) / (10**2))
        material_array = jnp.stack([varying_n] * self.nz, axis=2).astype(
            jnp.complex128
        )

        material = var_make_material(
            material=material_array, dx=self.dx, tz=self.tz
        )

        output = var_multislice(self.test_wavefront, material)

        chex.assert_shape(output.field, self.test_wavefront.field.shape)
        chex.assert_trees_all_equal(
            jnp.allclose(output.field, self.test_wavefront.field), False
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_multislice_dx_mismatch_smaller_wavefront(self) -> None:
        """Test automatic resampling when wavefront dx is smaller."""
        var_make_material = self.variant(make_sliced_material_function)
        var_make_wavefront = self.variant(make_optical_wavefront)
        var_multislice = self.variant(multislice_propagation)

        smaller_dx = self.dx * 0.5
        wavefront_small_dx = var_make_wavefront(
            field=self.test_wavefront.field,
            wavelength=self.wavelength,
            dx=smaller_dx,
            z_position=0.0,
        )

        material_array = (
            jnp.ones((self.ny, self.nx, self.nz), dtype=jnp.complex128) * 1.5
        )
        material = var_make_material(
            material=material_array, dx=self.dx, tz=self.tz
        )

        output = var_multislice(wavefront_small_dx, material)

        chex.assert_trees_all_close(output.dx, smaller_dx, rtol=1e-10)
        chex.assert_shape(output.field, self.test_wavefront.field.shape)

    @chex.variants(with_jit=True, without_jit=True)
    def test_multislice_dx_mismatch_larger_wavefront(self) -> None:
        """Test automatic resampling when wavefront dx is larger."""
        var_make_material = self.variant(make_sliced_material_function)
        var_make_wavefront = self.variant(make_optical_wavefront)
        var_multislice = self.variant(multislice_propagation)

        larger_dx = self.dx * 2.0
        wavefront_large_dx = var_make_wavefront(
            field=self.test_wavefront.field,
            wavelength=self.wavelength,
            dx=larger_dx,
            z_position=0.0,
        )

        material_array = (
            jnp.ones((self.ny, self.nx, self.nz), dtype=jnp.complex128) * 1.5
        )
        material = var_make_material(
            material=material_array, dx=self.dx, tz=self.tz
        )

        output = var_multislice(wavefront_large_dx, material)

        chex.assert_trees_all_close(output.dx, self.dx, rtol=1e-10)
        chex.assert_shape(output.field, self.test_wavefront.field.shape)

    @chex.variants(with_jit=True, without_jit=True)
    def test_multislice_energy_conservation_vacuum(self) -> None:
        """Test energy conservation in vacuum (no absorption)."""
        var_make_material = self.variant(make_sliced_material_function)
        var_multislice = self.variant(multislice_propagation)

        vacuum_material = jnp.ones(
            (self.ny, self.nx, self.nz), dtype=jnp.complex128
        )
        material = var_make_material(
            material=vacuum_material, dx=self.dx, tz=self.tz
        )

        output = var_multislice(self.plane_wave, material)

        input_energy = jnp.sum(jnp.abs(self.plane_wave.field) ** 2)
        output_energy = jnp.sum(jnp.abs(output.field) ** 2)

        chex.assert_trees_all_close(output_energy, input_energy, rtol=1e-2)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("glass", 1.5 + 0.0j),
        ("water", 1.33 + 0.0j),
        ("high_index", 2.0 + 0.0j),
    )
    def test_multislice_different_refractive_indices(
        self, refractive_index: complex
    ) -> None:
        """Test propagation through different transparent materials."""
        var_make_material = self.variant(make_sliced_material_function)
        var_multislice = self.variant(multislice_propagation)

        material_array = (
            jnp.ones((self.ny, self.nx, self.nz), dtype=jnp.complex128)
            * refractive_index
        )

        material = var_make_material(
            material=material_array, dx=self.dx, tz=self.tz
        )

        output = var_multislice(self.test_wavefront, material)

        chex.assert_shape(output.field, self.test_wavefront.field.shape)
        chex.assert_trees_all_equal(
            jnp.allclose(output.field, self.test_wavefront.field), False
        )


class TestComputeOpticalPathLength(chex.TestCase, parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.nx = 32
        self.ny = 32
        self.nz = 10
        self.dx = 1e-6
        self.tz = 5e-6

        material_array = (
            jnp.ones((self.ny, self.nx, self.nz), dtype=jnp.complex128) * 1.5
        )
        self.uniform_material = make_sliced_material_function(
            material=material_array, dx=self.dx, tz=self.tz
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_opl_full_projection(self) -> None:
        """Test OPL computation for entire material."""
        var_opl = self.variant(optical_path_length)

        opl = var_opl(self.uniform_material)

        chex.assert_shape(opl, (self.ny, self.nx))

        expected_opl = 1.5 * self.nz * self.tz
        chex.assert_trees_all_close(
            opl[0, 0], expected_opl, rtol=1e-10, atol=1e-15
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_opl_single_ray(self) -> None:
        """Test OPL computation for a single ray."""
        var_opl = self.variant(optical_path_length)

        opl_2d = var_opl(self.uniform_material, x_idx=15, y_idx=15)
        opl = opl_2d[0, 0]

        chex.assert_shape(opl_2d, (self.ny, self.nx))

        expected_opl = 1.5 * self.nz * self.tz
        chex.assert_trees_all_close(opl, expected_opl, rtol=1e-10, atol=1e-15)

    @chex.variants(with_jit=True, without_jit=True)
    def test_opl_x_line(self) -> None:
        """Test OPL computation along a line at fixed x."""
        var_opl = self.variant(optical_path_length)

        opl_2d = var_opl(self.uniform_material, x_idx=15, y_idx=-1)
        opl = opl_2d[0, :]

        chex.assert_shape(opl_2d, (self.ny, self.nx))
        chex.assert_shape(opl, (self.nx,))

        expected_opl = 1.5 * self.nz * self.tz
        chex.assert_trees_all_close(opl[0], expected_opl, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_opl_y_line(self) -> None:
        """Test OPL computation along a line at fixed y."""
        var_opl = self.variant(optical_path_length)

        opl_2d = var_opl(self.uniform_material, x_idx=-1, y_idx=15)
        opl = opl_2d[:, 0]

        chex.assert_shape(opl_2d, (self.ny, self.nx))
        chex.assert_shape(opl, (self.ny,))

        expected_opl = 1.5 * self.nz * self.tz
        chex.assert_trees_all_close(opl[0], expected_opl, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_opl_varying_material(self) -> None:
        """Test OPL with spatially varying refractive index."""
        var_make_material = self.variant(make_sliced_material_function)
        var_opl = self.variant(optical_path_length)

        n_values = jnp.linspace(1.0, 2.0, self.nz)
        material_array = jnp.ones(
            (self.ny, self.nx, self.nz), dtype=jnp.complex128
        )
        for i in range(self.nz):
            material_array = material_array.at[:, :, i].set(n_values[i])

        varying_material = var_make_material(
            material=material_array, dx=self.dx, tz=self.tz
        )

        opl = var_opl(varying_material)

        expected_opl = jnp.sum(n_values) * self.tz
        chex.assert_trees_all_close(opl[0, 0], expected_opl, rtol=1e-9)


class TestComputeTotalTransmission(chex.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.nx = 32
        self.ny = 32
        self.nz = 10
        self.dx = 1e-6
        self.tz = 5e-6
        self.wavelength = 500e-9

    @chex.variants(with_jit=True, without_jit=True)
    def test_transmission_transparent_material(self) -> None:
        """Test transmission through transparent material (no absorption)."""
        var_make_material = self.variant(make_sliced_material_function)
        var_compute_transmission = self.variant(total_transmit)

        transparent_material = (
            jnp.ones((self.ny, self.nx, self.nz), dtype=jnp.complex128) * 1.5
        )

        material = var_make_material(
            material=transparent_material, dx=self.dx, tz=self.tz
        )

        transmission = var_compute_transmission(material, self.wavelength)

        chex.assert_shape(transmission, (self.ny, self.nx))

        chex.assert_trees_all_close(transmission, 1.0, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_transmission_absorbing_material(self) -> None:
        """Test transmission through absorbing material."""
        var_make_material = self.variant(make_sliced_material_function)
        var_compute_transmission = self.variant(total_transmit)

        absorbing_material = jnp.ones(
            (self.ny, self.nx, self.nz), dtype=jnp.complex128
        ) * (1.5 + 0.1j)

        material = var_make_material(
            material=absorbing_material, dx=self.dx, tz=self.tz
        )

        transmission = var_compute_transmission(material, self.wavelength)

        chex.assert_shape(transmission, (self.ny, self.nx))

        chex.assert_trees_all_equal((transmission < 1.0).all(), True)
        chex.assert_trees_all_equal((transmission > 0.0).all(), True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_transmission_varying_absorption(self) -> None:
        """Test transmission with spatially varying absorption."""
        var_make_material = self.variant(make_sliced_material_function)
        var_compute_transmission = self.variant(total_transmit)

        x = jnp.arange(self.nx) - self.nx / 2
        y = jnp.arange(self.ny) - self.ny / 2
        xx, yy = jnp.meshgrid(x, y, indexing="ij")
        r2d = jnp.sqrt(xx**2 + yy**2)

        kappa = 0.05 * jnp.exp(-(r2d**2) / (5**2))
        material_array = jnp.stack([1.5 + 1j * kappa] * self.nz, axis=2)

        material = var_make_material(
            material=material_array, dx=self.dx, tz=self.tz
        )

        transmission = var_compute_transmission(material, self.wavelength)

        center_idx = self.nx // 2
        edge_idx = 0

        chex.assert_trees_all_equal(
            transmission[center_idx, center_idx]
            < transmission[edge_idx, edge_idx],
            True,
        )


if __name__ == "__main__":
    pytest.main([__file__])
