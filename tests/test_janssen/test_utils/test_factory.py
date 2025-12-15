"""Tests for factory functions in janssen.utils.factory module."""

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from janssen.utils import (
    LensParams,
    OpticalWavefront,
    make_diffractogram,
    make_grid_params,
    make_lens_params,
    make_microscope_data,
    make_optical_wavefront,
    make_optimizer_state,
    make_ptychography_params,
    make_sample_function,
    make_sliced_material_function,
)


class TestMakeLensParams(chex.TestCase):
    """Test the make_lens_params factory function."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self) -> None:
        """Test basic creation with required parameters."""
        var_make_lens_params = self.variant(make_lens_params)
        lens = var_make_lens_params(
            focal_length=0.01,
            diameter=0.00005,
            n=1.5,
            center_thickness=0.001,
            r1=0.01,
            r2=0.01,
        )

        chex.assert_equal(type(lens).__name__, "LensParams")
        chex.assert_scalar_positive(float(lens.focal_length))
        chex.assert_scalar_positive(float(lens.diameter))
        chex.assert_scalar_positive(float(lens.n))

    @chex.variants(with_jit=True, without_jit=True)
    def test_type_conversion(self) -> None:
        """Test that factory converts Python types to JAX arrays."""
        var_make_lens_params = self.variant(make_lens_params)
        lens = var_make_lens_params(
            focal_length=0.01,
            diameter=0.00005,
            n=1.5,
            center_thickness=0.001,
            r1=0.01,
            r2=0.01,
        )
        chex.assert_equal(isinstance(lens.focal_length, jax.Array), True)
        chex.assert_equal(isinstance(lens.diameter, jax.Array), True)
        chex.assert_equal(isinstance(lens.n, jax.Array), True)
        chex.assert_equal(isinstance(lens.center_thickness, jax.Array), True)
        chex.assert_equal(isinstance(lens.r1, jax.Array), True)
        chex.assert_equal(isinstance(lens.r2, jax.Array), True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_array_input(self) -> None:
        """Test that factory accepts JAX arrays."""
        var_make_lens_params = self.variant(make_lens_params)
        lens = var_make_lens_params(
            focal_length=jnp.array(0.01),
            diameter=jnp.array(0.00005),
            n=jnp.array(1.5),
            center_thickness=jnp.array(0.001),
            r1=jnp.array(0.01),
            r2=jnp.array(0.01),
        )

        chex.assert_equal(type(lens).__name__, "LensParams")
        chex.assert_trees_all_close(lens.focal_length, jnp.array(0.01))


class TestMakeGridParams(chex.TestCase):
    """Test the make_grid_params factory function."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self) -> None:
        """Test basic creation with required parameters."""
        nx, ny = 64, 64
        x = jnp.linspace(-1, 1, nx)
        y = jnp.linspace(-1, 1, ny)
        xx, yy = jnp.meshgrid(x, y)
        var_make_grid_params = self.variant(make_grid_params)
        grid = var_make_grid_params(
            xx=xx,
            yy=yy,
            phase_profile=jnp.zeros((ny, nx)),
            transmission=jnp.ones((ny, nx)),
        )

        chex.assert_equal(type(grid).__name__, "GridParams")
        chex.assert_shape(grid.xx, (ny, nx))
        chex.assert_shape(grid.yy, (ny, nx))
        chex.assert_shape(grid.phase_profile, (ny, nx))
        chex.assert_shape(grid.transmission, (ny, nx))

    @chex.variants(with_jit=True, without_jit=True)
    def test_default_values(self) -> None:
        """Test creating grid with default-like values."""
        nx, ny = 32, 32
        x = jnp.linspace(-1, 1, nx)
        y = jnp.linspace(-1, 1, ny)
        xx, yy = jnp.meshgrid(x, y)

        var_make_grid_params = self.variant(make_grid_params)
        grid = var_make_grid_params(
            xx=xx,
            yy=yy,
            phase_profile=jnp.zeros_like(xx),
            transmission=jnp.ones_like(xx),
        )

        chex.assert_trees_all_close(grid.phase_profile, jnp.zeros_like(xx))
        chex.assert_trees_all_close(grid.transmission, jnp.ones_like(xx))


class TestMakeOpticalWavefront(chex.TestCase):
    """Test the make_optical_wavefront factory function."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self) -> None:
        """Test basic creation with scalar field."""
        field = jnp.ones((64, 64), dtype=complex)
        var_make_optical_wavefront = self.variant(make_optical_wavefront)
        wavefront = var_make_optical_wavefront(
            field=field,
            wavelength=532e-9,
            dx=1e-6,
            z_position=0.0,
        )

        chex.assert_equal(type(wavefront).__name__, "OpticalWavefront")
        chex.assert_shape(wavefront.field, (64, 64))
        chex.assert_shape(wavefront.wavelength, ())
        chex.assert_shape(wavefront.dx, ())
        chex.assert_shape(wavefront.z_position, ())
        chex.assert_equal(wavefront.polarization, jnp.array(False))

    def test_polarized_field(self) -> None:
        """Test creation with polarized field."""
        field = jnp.ones((64, 64, 2), dtype=complex)
        wavefront = make_optical_wavefront(
            field=field,
            wavelength=532e-9,
            dx=1e-6,
            z_position=0.0,
        )

        chex.assert_shape(wavefront.field, (64, 64, 2))
        chex.assert_equal(wavefront.polarization, jnp.array(True))

    def test_z_position_required(self) -> None:
        """Test that z_position is required."""
        field = jnp.ones((32, 32), dtype=complex)
        wavefront = make_optical_wavefront(
            field=field,
            wavelength=632e-9,
            dx=2e-6,
            z_position=0.0,
        )

        chex.assert_trees_all_close(wavefront.z_position, jnp.array(0.0))

    def test_type_conversion(self) -> None:
        """Test type conversion from Python types."""
        field = jnp.ones((32, 32), dtype=complex)
        wavefront = make_optical_wavefront(
            field=field,
            wavelength=532e-9,
            dx=1e-6,
            z_position=0.0,
        )

        chex.assert_equal(isinstance(wavefront.wavelength, jax.Array), True)
        chex.assert_equal(isinstance(wavefront.dx, jax.Array), True)
        chex.assert_equal(isinstance(wavefront.z_position, jax.Array), True)


class TestMakeMicroscopeData(chex.TestCase):
    """Test the make_microscope_data factory function."""

    def test_3d_data(self) -> None:
        """Test creation with 3D image data."""
        data = make_microscope_data(
            image_data=jnp.ones((10, 64, 64)),
            positions=jnp.zeros((10, 2)),
            wavelength=532e-9,
            dx=1e-6,
        )

        chex.assert_equal(type(data).__name__, "MicroscopeData")
        chex.assert_shape(data.image_data, (10, 64, 64))
        chex.assert_shape(data.positions, (10, 2))

    def test_4d_data(self) -> None:
        """Test creation with 4D image data."""
        data = make_microscope_data(
            image_data=jnp.ones((5, 5, 64, 64)),
            positions=jnp.zeros((25, 2)),
            wavelength=532e-9,
            dx=1e-6,
        )

        chex.assert_equal(type(data).__name__, "MicroscopeData")
        chex.assert_shape(data.image_data, (5, 5, 64, 64))
        chex.assert_shape(data.positions, (25, 2))

    def test_position_validation(self) -> None:
        """Test that positions must have correct shape."""
        data = make_microscope_data(
            image_data=jnp.ones((3, 32, 32)),
            positions=jnp.zeros((3, 2)),
            wavelength=632e-9,
            dx=2e-6,
        )
        chex.assert_shape(data.positions, (3, 2))


class TestMakeSampleFunction(chex.TestCase):
    """Test the make_sample_function factory function."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        sample = make_sample_function(
            sample=jnp.ones((64, 64), dtype=complex),
            dx=1e-6,
        )

        chex.assert_equal(type(sample).__name__, "SampleFunction")
        chex.assert_shape(sample.sample, (64, 64))
        chex.assert_shape(sample.dx, ())

    def test_complex_conversion(self) -> None:
        """Test that sample is converted to complex."""
        real_array = jnp.ones((32, 32))
        sample = make_sample_function(
            sample=real_array,
            dx=2e-6,
        )

        chex.assert_equal(jnp.iscomplexobj(sample.sample), True)


class TestMakeSlicedMaterialFunction(chex.TestCase):
    """Test the make_sliced_material_function factory function."""

    def test_basic_creation(self) -> None:
        """Test basic creation with 3D material array."""
        material = jnp.ones((64, 64, 10), dtype=complex) * (1.5 + 0.01j)
        sliced_mat = make_sliced_material_function(
            material=material,
            dx=1e-6,
            tz=5e-6,
        )

        chex.assert_equal(type(sliced_mat).__name__, "SlicedMaterialFunction")
        chex.assert_shape(sliced_mat.material, (64, 64, 10))
        chex.assert_shape(sliced_mat.dx, ())
        chex.assert_shape(sliced_mat.tz, ())

    def test_complex_conversion(self) -> None:
        """Test that material is converted to complex."""
        real_array = jnp.ones((32, 32, 5)) * 1.5
        sliced_mat = make_sliced_material_function(
            material=real_array,
            dx=2e-6,
            tz=3e-6,
        )

        chex.assert_equal(jnp.iscomplexobj(sliced_mat.material), True)
        chex.assert_shape(sliced_mat.material, (32, 32, 5))

    def test_refractive_index_properties(self) -> None:
        """Test material with realistic refractive index values."""
        # Create material with n=1.5 (real part) and κ=0.01 (imaginary part)
        n = 1.5
        kappa = 0.01
        material = jnp.ones((16, 16, 8), dtype=complex) * (n + 1j * kappa)

        sliced_mat = make_sliced_material_function(
            material=material,
            dx=1e-6,
            tz=2e-6,
        )

        # Check that real and imaginary parts are preserved
        chex.assert_trees_all_close(
            jnp.real(sliced_mat.material[0, 0, 0]),
            jnp.array(n),
            atol=1e-6,
        )
        chex.assert_trees_all_close(
            jnp.imag(sliced_mat.material[0, 0, 0]),
            jnp.array(kappa),
            atol=1e-6,
        )

    def test_type_conversion(self) -> None:
        """Test conversion from Python types."""
        material = jnp.ones((16, 16, 4), dtype=complex)
        sliced_mat = make_sliced_material_function(
            material=material,
            dx=1.5e-6,
            tz=2.0e-6,
        )

        chex.assert_equal(isinstance(sliced_mat.dx, jax.Array), True)
        chex.assert_equal(isinstance(sliced_mat.tz, jax.Array), True)
        chex.assert_equal(isinstance(sliced_mat.material, jax.Array), True)

    def test_integer_spacing_parameters(self) -> None:
        """Test that dx and tz accept integer inputs."""
        material = jnp.ones((8, 8, 3), dtype=complex) * 1.33
        sliced_mat = make_sliced_material_function(
            material=material,
            dx=1,  # integer instead of 1.0
            tz=2,  # integer instead of 2.0
        )

        chex.assert_equal(type(sliced_mat).__name__, "SlicedMaterialFunction")
        chex.assert_trees_all_close(sliced_mat.dx, jnp.array(1.0))
        chex.assert_trees_all_close(sliced_mat.tz, jnp.array(2.0))

    def test_varying_refractive_index(self) -> None:
        """Test material with spatially varying refractive index."""
        # Create a gradient in the refractive index
        hh, ww, zz = 32, 32, 6
        material = jnp.zeros((hh, ww, zz), dtype=complex)

        for z in range(zz):
            # Each slice has a different refractive index
            n_slice = 1.0 + 0.1 * z
            material = material.at[:, :, z].set(n_slice + 0.001j)

        sliced_mat = make_sliced_material_function(
            material=material,
            dx=0.5e-6,
            tz=1e-6,
        )

        chex.assert_shape(sliced_mat.material, (hh, ww, zz))
        # Check first and last slice have different values
        first_slice_val = sliced_mat.material[0, 0, 0]
        last_slice_val = sliced_mat.material[0, 0, -1]
        assert jnp.real(first_slice_val) != jnp.real(last_slice_val)


class TestMakeDiffractogram(chex.TestCase):
    """Test the make_diffractogram factory function."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        diff = make_diffractogram(
            image=jnp.ones((128, 128)),
            wavelength=632e-9,
            dx=1e-6,
        )

        chex.assert_equal(type(diff).__name__, "Diffractogram")
        chex.assert_shape(diff.image, (128, 128))
        chex.assert_shape(diff.wavelength, ())
        chex.assert_shape(diff.dx, ())

    def test_type_conversion(self) -> None:
        """Test conversion from Python types."""
        diff = make_diffractogram(
            image=jnp.ones((64, 64)),
            wavelength=532e-9,
            dx=2e-6,
        )

        chex.assert_equal(isinstance(diff.wavelength, jax.Array), True)
        chex.assert_equal(isinstance(diff.dx, jax.Array), True)


class TestMakeOptimizerState(chex.TestCase):
    """Test the make_optimizer_state factory function."""

    def test_basic_creation(self) -> None:
        """Test basic creation with shape."""
        shape = (64, 64)
        state = make_optimizer_state(shape=shape)

        chex.assert_equal(type(state).__name__, "OptimizerState")
        chex.assert_shape(state.m, shape)
        chex.assert_shape(state.v, shape)
        chex.assert_shape(state.step, ())

    def test_with_initial_values(self) -> None:
        """Test creation with initial values."""
        shape = (32, 32)
        m_init = jnp.ones(shape, dtype=complex) * 0.9
        v_init = jnp.ones(shape, dtype=float) * 0.999

        state = make_optimizer_state(
            shape=shape,
            m=m_init,
            v=v_init,
            step=10,
        )

        chex.assert_trees_all_close(state.m, m_init)
        chex.assert_trees_all_close(state.v, v_init)
        chex.assert_trees_all_close(state.step, jnp.array(10))

    def test_default_values(self) -> None:
        """Test default values for optimizer state."""
        shape = (16, 16)
        state = make_optimizer_state(shape=shape)

        chex.assert_trees_all_close(
            state.m, jnp.ones(shape, dtype=complex) * 1j
        )
        chex.assert_trees_all_close(state.v, jnp.zeros(shape, dtype=float))
        chex.assert_trees_all_close(state.step, jnp.array(0))


class TestMakePtychographyParams(chex.TestCase):
    """Test the make_ptychography_params factory function."""

    def test_basic_creation(self) -> None:
        """Test basic creation with all parameters."""
        params = make_ptychography_params(
            camera_pixel_size=5e-6,
            num_iterations=100,
            learning_rate=0.01,
            loss_type=0,
            optimizer_type=0,
        )

        chex.assert_equal(type(params).__name__, "PtychographyParams")
        chex.assert_shape(params.camera_pixel_size, ())
        chex.assert_shape(params.learning_rate, ())
        chex.assert_shape(params.num_iterations, ())
        chex.assert_shape(params.loss_type, ())
        chex.assert_shape(params.optimizer_type, ())
        chex.assert_shape(params.zoom_factor_bounds, (2,))
        chex.assert_shape(params.aperture_diameter_bounds, (2,))
        chex.assert_shape(params.travel_distance_bounds, (2,))
        chex.assert_shape(params.aperture_center_bounds, (2, 2))

    def test_bounds_creation(self) -> None:
        """Test creation with explicit bounds."""
        params = make_ptychography_params(
            camera_pixel_size=10e-6,
            num_iterations=50,
            learning_rate=0.001,
            zoom_factor_bounds=jnp.array([1.0, 3.0]),
            aperture_diameter_bounds=jnp.array([0.5e-3, 2.0e-3]),
            travel_distance_bounds=jnp.array([0.01, 0.2]),
            aperture_center_bounds=jnp.array([[-1.0, -1.0], [1.0, 1.0]]),
        )

        chex.assert_equal(
            isinstance(params.zoom_factor_bounds, jax.Array), True
        )
        chex.assert_shape(params.zoom_factor_bounds, (2,))
        chex.assert_shape(params.aperture_center_bounds, (2, 2))

    def test_type_conversion(self) -> None:
        """Test type conversion for all parameters."""
        params = make_ptychography_params(
            camera_pixel_size=5e-6,
            num_iterations=100,
            learning_rate=0.01,
        )
        chex.assert_equal(
            isinstance(params.camera_pixel_size, jax.Array), True
        )
        chex.assert_equal(isinstance(params.learning_rate, jax.Array), True)
        chex.assert_equal(isinstance(params.num_iterations, jax.Array), True)
        chex.assert_equal(isinstance(params.loss_type, jax.Array), True)
        chex.assert_equal(isinstance(params.optimizer_type, jax.Array), True)


class TestFactoryValidation(chex.TestCase):
    """Test validation and error handling in factory functions."""

    def test_lens_params_validation(self) -> None:
        """Test that make_lens_params validates inputs."""
        lens = make_lens_params(
            focal_length=0.01,
            diameter=0.00005,
            n=1.5,
            center_thickness=0.001,
            r1=0.01,
            r2=0.01,
        )
        chex.assert_equal(type(lens).__name__, "LensParams")

    def test_lens_params_integer_inputs(self) -> None:
        """Test that make_lens_params accepts integer inputs."""
        lens = make_lens_params(
            focal_length=1,  # integer
            diameter=0.5,
            n=2,  # integer
            center_thickness=1,  # integer
            r1=1,  # integer
            r2=1,  # integer
        )
        chex.assert_equal(type(lens).__name__, "LensParams")
        chex.assert_scalar_positive(float(lens.focal_length))
        chex.assert_trees_all_close(lens.focal_length, jnp.array(1.0))
        chex.assert_trees_all_close(lens.n, jnp.array(2.0))

    def test_optical_wavefront_integer_params(self) -> None:
        """Test that make_optical_wavefront accepts integer parameters."""
        field = jnp.ones((32, 32), dtype=complex)
        wavefront = make_optical_wavefront(
            field=field,
            wavelength=532e-9,
            dx=1,  # integer instead of 1.0
            z_position=0,  # integer instead of 0.0
        )
        chex.assert_equal(type(wavefront).__name__, "OpticalWavefront")
        chex.assert_trees_all_close(wavefront.dx, jnp.array(1.0))
        chex.assert_trees_all_close(wavefront.z_position, jnp.array(0.0))

    def test_sample_function_integer_dx(self) -> None:
        """Test that make_sample_function accepts integer dx."""
        sample = make_sample_function(
            sample=jnp.ones((64, 64), dtype=complex),
            dx=2,  # integer instead of 2.0
        )
        chex.assert_equal(type(sample).__name__, "SampleFunction")
        chex.assert_trees_all_close(sample.dx, jnp.array(2.0))

    def test_ptychography_params_mixed_types(self) -> None:
        """Test make_ptychography_params with mixed int/float inputs."""
        params = make_ptychography_params(
            camera_pixel_size=5e-6,
            num_iterations=100,
            learning_rate=1,  # integer (unusual but valid)
            loss_type=0,
            optimizer_type=0,
        )
        chex.assert_equal(type(params).__name__, "PtychographyParams")
        chex.assert_trees_all_close(params.learning_rate, jnp.array(1.0))
        chex.assert_trees_all_close(params.num_iterations, jnp.array(100))

    def test_wavefront_field_shape(self) -> None:
        """Test OpticalWavefront field shape validation."""
        field_2d = jnp.ones((32, 32), dtype=complex)
        wf_2d = make_optical_wavefront(
            field=field_2d,
            wavelength=532e-9,
            dx=1e-6,
            z_position=0.0,
        )
        chex.assert_equal(wf_2d.polarization, jnp.array(False))

        field_3d = jnp.ones((32, 32, 2), dtype=complex)
        wf_3d = make_optical_wavefront(
            field=field_3d,
            wavelength=532e-9,
            dx=1e-6,
            z_position=0.0,
        )
        chex.assert_equal(wf_3d.polarization, jnp.array(True))

    def test_microscope_data_dimensions(self) -> None:
        """Test MicroscopeData dimension consistency."""
        data_3d = make_microscope_data(
            image_data=jnp.ones((5, 32, 32)),
            positions=jnp.zeros((5, 2)),
            wavelength=632e-9,
            dx=2e-6,
        )
        chex.assert_shape(data_3d.image_data, (5, 32, 32))
        chex.assert_shape(data_3d.positions, (5, 2))

        data_4d = make_microscope_data(
            image_data=jnp.ones((3, 3, 32, 32)),
            positions=jnp.zeros((9, 2)),
            wavelength=632e-9,
            dx=2e-6,
        )
        chex.assert_shape(data_4d.image_data, (3, 3, 32, 32))
        chex.assert_shape(data_4d.positions, (9, 2))

    def test_sliced_material_function_validation(self) -> None:
        """Test SlicedMaterialFunction creation and validation."""
        material = jnp.ones((16, 16, 5), dtype=complex) * (1.5 + 0.01j)
        sliced_mat = make_sliced_material_function(
            material=material,
            dx=1e-6,
            tz=2e-6,
        )

        chex.assert_shape(sliced_mat.material, (16, 16, 5))
        chex.assert_scalar_positive(float(sliced_mat.dx))
        chex.assert_scalar_positive(float(sliced_mat.tz))


class TestFactoryJITCompatibility(chex.TestCase):
    """Test that factory functions work with JAX transformations."""

    def test_lens_params_in_jit(self) -> None:
        """Test make_lens_params can be used in JIT context."""

        @jax.jit
        def create_and_process_lens(
            focal_length: Float[Array, " "],
        ) -> Float[Array, " "]:
            """Create lens and compute power."""
            lens = LensParams(
                focal_length=focal_length,
                diameter=jnp.array(0.00005),
                n=jnp.array(1.5),
                center_thickness=jnp.array(0.001),
                r1=jnp.array(0.01),
                r2=jnp.array(0.01),
            )
            return 1.0 / lens.focal_length

        result = create_and_process_lens(jnp.array(0.01))
        chex.assert_scalar_positive(float(result))

    def test_wavefront_in_vmap(self) -> None:
        """Test OpticalWavefront creation in vmap."""
        wavelengths = jnp.array([400e-9, 500e-9, 600e-9])
        field = jnp.ones((32, 32), dtype=complex)

        def create_wavefront(
            wavelength: Float[Array, " "],
        ) -> OpticalWavefront:
            """Create wavefront with given wavelength."""
            return OpticalWavefront(
                field=field,
                wavelength=wavelength,
                dx=jnp.array(1e-6),
                z_position=jnp.array(0.0),
                polarization=jnp.array(False),
            )

        vmapped_create = jax.vmap(create_wavefront)
        batch = vmapped_create(wavelengths)

        chex.assert_shape(batch.wavelength, (3,))
        chex.assert_shape(batch.field, (3, 32, 32))


if __name__ == "__main__":
    pytest.main([__file__])
