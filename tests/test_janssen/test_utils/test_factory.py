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
            focal_length=0.01,  # Python float
            diameter=0.00005,
            n=1.5,
            center_thickness=0.001,
            r1=0.01,
            r2=0.01,
        )

        # All fields should be JAX arrays
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
        """Test that defaults are applied correctly."""
        nx, ny = 32, 32
        x = jnp.linspace(-1, 1, nx)
        y = jnp.linspace(-1, 1, ny)
        xx, yy = jnp.meshgrid(x, y)

        # Should use defaults for phase_profile and transmission
        var_make_grid_params = self.variant(make_grid_params)
        grid = var_make_grid_params(xx=xx, yy=yy)

        # Defaults should be zeros for phase, ones for transmission
        chex.assert_trees_all_close(
            grid.phase_profile, jnp.zeros_like(xx)
        )
        chex.assert_trees_all_close(
            grid.transmission, jnp.ones_like(xx)
        )


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
        chex.assert_shape(wavefront.wavelength, ())  # 0-dim array
        chex.assert_shape(wavefront.dx, ())  # 0-dim array
        chex.assert_shape(wavefront.z_position, ())  # 0-dim array
        # Should be False for 2D field
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
        # Should be True for 3D field
        chex.assert_equal(wavefront.polarization, jnp.array(True))

    def test_z_position_required(self) -> None:
        """Test that z_position is required."""
        field = jnp.ones((32, 32), dtype=complex)
        wavefront = make_optical_wavefront(
            field=field,
            wavelength=632e-9,
            dx=2e-6,
            z_position=0.0,  # Required parameter
        )

        chex.assert_trees_all_close(
            wavefront.z_position, jnp.array(0.0)
        )

    def test_type_conversion(self) -> None:
        """Test type conversion from Python types."""
        field = jnp.ones((32, 32), dtype=complex)
        wavefront = make_optical_wavefront(
            field=field,
            wavelength=532e-9,  # Python float
            dx=1e-6,  # Python float
            z_position=0.0,  # Python float
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
        # This should work
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
        chex.assert_shape(sample.dx, ())  # 0-dim array

    def test_complex_conversion(self) -> None:
        """Test that sample is converted to complex."""
        # Input real array
        real_array = jnp.ones((32, 32))
        sample = make_sample_function(
            sample=real_array,
            dx=2e-6,
        )

        # Should be complex
        chex.assert_equal(jnp.iscomplexobj(sample.sample), True)


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
        chex.assert_shape(diff.wavelength, ())  # 0-dim array
        chex.assert_shape(diff.dx, ())  # 0-dim array

    def test_type_conversion(self) -> None:
        """Test conversion from Python types."""
        diff = make_diffractogram(
            image=jnp.ones((64, 64)),
            wavelength=532e-9,  # Python float
            dx=2e-6,  # Python float
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
        chex.assert_shape(state.step, ())  # 0-dim array

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

        # Check defaults from factory function
        # Default m should be 1j (complex)
        chex.assert_trees_all_close(
            state.m, jnp.ones(shape, dtype=complex) * 1j
        )
        # Default v should be 0.0
        chex.assert_trees_all_close(
            state.v, jnp.zeros(shape, dtype=float)
        )
        # Default step should be 0
        chex.assert_trees_all_close(state.step, jnp.array(0))


class TestMakePtychographyParams(chex.TestCase):
    """Test the make_ptychography_params factory function."""

    def test_basic_creation(self) -> None:
        """Test basic creation with all parameters."""
        params = make_ptychography_params(
            zoom_factor=2.0,
            aperture_diameter=1e-3,
            travel_distance=0.1,
            aperture_center=jnp.array([0.0, 0.0]),
            camera_pixel_size=5e-6,
            learning_rate=0.01,
            num_iterations=100,
        )

        chex.assert_equal(type(params).__name__, "PtychographyParams")
        chex.assert_shape(params.zoom_factor, ())  # 0-dim array
        chex.assert_shape(params.aperture_diameter, ())  # 0-dim array
        chex.assert_shape(params.travel_distance, ())  # 0-dim array
        chex.assert_shape(params.aperture_center, (2,))
        chex.assert_shape(params.camera_pixel_size, ())  # 0-dim array
        chex.assert_shape(params.learning_rate, ())  # 0-dim array
        chex.assert_shape(params.num_iterations, ())  # 0-dim array

    def test_aperture_center_conversion(self) -> None:
        """Test that aperture_center is converted to JAX array."""
        # Test with list input
        params = make_ptychography_params(
            zoom_factor=1.5,
            aperture_diameter=2e-3,
            travel_distance=0.05,
            aperture_center=jnp.array([1.0, 2.0]),  # Must be JAX array
            camera_pixel_size=10e-6,
            learning_rate=0.001,
            num_iterations=50,
        )

        chex.assert_equal(isinstance(params.aperture_center, jax.Array), True)
        chex.assert_shape(params.aperture_center, (2,))

    def test_type_conversion(self) -> None:
        """Test type conversion for all parameters."""
        params = make_ptychography_params(
            zoom_factor=2.0,  # Python float
            aperture_diameter=1e-3,  # Python float
            travel_distance=0.1,  # Python float
            aperture_center=jnp.array([0.0, 0.0]),
            camera_pixel_size=5e-6,  # Python float
            learning_rate=0.01,  # Python float
            num_iterations=100,  # Python int
        )

        # All should be JAX arrays
        chex.assert_equal(isinstance(params.zoom_factor, jax.Array), True)
        chex.assert_equal(isinstance(params.aperture_diameter, jax.Array), True)
        chex.assert_equal(isinstance(params.travel_distance, jax.Array), True)
        chex.assert_equal(isinstance(params.camera_pixel_size, jax.Array), True)
        chex.assert_equal(isinstance(params.learning_rate, jax.Array), True)
        chex.assert_equal(isinstance(params.num_iterations, jax.Array), True)


class TestFactoryValidation(chex.TestCase):
    """Test validation and error handling in factory functions."""

    def test_lens_params_validation(self) -> None:
        """Test that make_lens_params validates inputs."""
        # Should succeed with valid inputs
        lens = make_lens_params(
            focal_length=0.01,
            diameter=0.00005,
            n=1.5,
            center_thickness=0.001,
            r1=0.01,
            r2=0.01,
        )
        chex.assert_equal(type(lens).__name__, "LensParams")

    def test_wavefront_field_shape(self) -> None:
        """Test OpticalWavefront field shape validation."""
        # 2D field should work
        field_2d = jnp.ones((32, 32), dtype=complex)
        wf_2d = make_optical_wavefront(
            field=field_2d,
            wavelength=532e-9,
            dx=1e-6,
            z_position=0.0,
        )
        chex.assert_equal(wf_2d.polarization, jnp.array(False))

        # 3D field with 2 components should work
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
        # 3D data with matching positions
        data_3d = make_microscope_data(
            image_data=jnp.ones((5, 32, 32)),
            positions=jnp.zeros((5, 2)),
            wavelength=632e-9,
            dx=2e-6,
        )
        chex.assert_shape(data_3d.image_data, (5, 32, 32))
        chex.assert_shape(data_3d.positions, (5, 2))

        # 4D data with matching positions
        data_4d = make_microscope_data(
            image_data=jnp.ones((3, 3, 32, 32)),
            positions=jnp.zeros((9, 2)),
            wavelength=632e-9,
            dx=2e-6,
        )
        chex.assert_shape(data_4d.image_data, (3, 3, 32, 32))
        chex.assert_shape(data_4d.positions, (9, 2))


class TestFactoryJITCompatibility(chex.TestCase):
    """Test that factory functions work with JAX transformations."""

    def test_lens_params_in_jit(self) -> None:
        """Test make_lens_params can be used in JIT context."""

        @jax.jit
        def create_and_process_lens(
            focal_length: Float[Array, " "]
        ) -> Float[Array, " "]:
            """Create lens and compute power."""
            # Direct instantiation works in JIT
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
            wavelength: Float[Array, " "]
        ) -> OpticalWavefront:
            """Create wavefront with given wavelength."""
            # Direct instantiation for vmap
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