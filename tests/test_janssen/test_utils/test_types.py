"""Tests for PyTree structures in janssen.utils.types module."""

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import pytest
from jaxtyping import Array, Complex, Float

from janssen.utils import (
    Diffractogram,
    GridParams,
    LensParams,
    MicroscopeData,
    OpticalWavefront,
    OptimizerState,
    PtychographyParams,
    SampleFunction,
    make_diffractogram,
    make_grid_params,
    make_lens_params,
    make_microscope_data,
    make_optical_wavefront,
    make_optimizer_state,
    make_ptychography_params,
    make_sample_function,
)


class TestOpticalWavefrontPyTree(chex.TestCase):
    """Test that OpticalWavefront behaves correctly as a PyTree."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        # Create test data
        self.field = jnp.ones((64, 64), dtype=complex) * (1 + 0.5j)
        self.wavelength = jnp.array(532e-9)
        self.dx = jnp.array(1e-6)
        self.z_position = jnp.array(0.0)
        self.polarization = jnp.array(False)

        # Create test wavefront
        self.wavefront = make_optical_wavefront(
            field=self.field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=self.z_position,
        )

    def test_is_pytree(self) -> None:
        """Test that OpticalWavefront is recognized as a PyTree."""
        # Check that it can be flattened and unflattened
        leaves, treedef = tree.tree_flatten(self.wavefront)
        reconstructed = tree.tree_unflatten(treedef, leaves)

        # Verify reconstruction
        chex.assert_trees_all_close(reconstructed.field, self.wavefront.field)
        chex.assert_trees_all_close(
            reconstructed.wavelength, self.wavefront.wavelength
        )
        chex.assert_trees_all_close(reconstructed.dx, self.wavefront.dx)
        chex.assert_trees_all_close(
            reconstructed.z_position, self.wavefront.z_position
        )
        chex.assert_trees_all_close(
            reconstructed.polarization, self.wavefront.polarization
        )

    def test_tree_map(self) -> None:
        """Test that tree_map works correctly on OpticalWavefront."""

        def scale_by_two(x: Array) -> Array:
            """Scale numeric values by 2."""
            if isinstance(x, jax.Array | jnp.ndarray):
                if x.dtype in [jnp.bool_, bool]:
                    return x  # Don't scale boolean
                return x * 2
            return x

        # Apply tree_map
        scaled = tree.tree_map(scale_by_two, self.wavefront)

        # Check that numeric fields are scaled
        chex.assert_trees_all_close(scaled.field, self.wavefront.field * 2)
        chex.assert_trees_all_close(
            scaled.wavelength, self.wavefront.wavelength * 2
        )
        chex.assert_trees_all_close(scaled.dx, self.wavefront.dx * 2)
        chex.assert_trees_all_close(
            scaled.z_position, self.wavefront.z_position * 2
        )
        # Polarization (bool) should not be scaled
        chex.assert_trees_all_close(
            scaled.polarization, self.wavefront.polarization
        )

    def test_tree_reduce(self) -> None:
        """Test that tree_reduce works correctly on OpticalWavefront."""

        def sum_leaves(acc: float, x: Array) -> float:
            """Sum all numeric leaves."""
            if isinstance(x, jax.Array | jnp.ndarray):
                if x.dtype in [jnp.bool_, bool]:
                    return acc  # Skip boolean
                return acc + jnp.sum(x)
            return acc

        # Calculate sum of all numeric leaves
        total = tree.tree_reduce(sum_leaves, self.wavefront, 0.0)

        # Verify the sum is reasonable (non-zero for our test data)
        chex.assert_scalar_positive(float(jnp.abs(total)))

    def test_tree_leaves(self) -> None:
        """Test that tree_leaves extracts all leaves correctly."""
        leaves = tree.tree_leaves(self.wavefront)

        # Should have 5 leaves (field, wavelength, dx, z_position,
        # polarization)
        chex.assert_equal(len(leaves), 5)

        # Check that all leaves are JAX arrays
        for leaf in leaves:
            # Check leaf is either jax.Array or numpy ndarray
            is_array = isinstance(leaf, jax.Array | jnp.ndarray)
            chex.assert_equal(is_array, True)

    def test_tree_structure(self) -> None:
        """Test that tree structure is preserved correctly."""
        # Get tree structure
        structure = tree.tree_structure(self.wavefront)

        # Create a new wavefront with different values
        new_field = jnp.zeros((64, 64), dtype=complex)
        new_wavelength = jnp.array(632e-9)
        new_wavefront = make_optical_wavefront(
            field=new_field,
            wavelength=new_wavelength,
            dx=self.dx,
            z_position=self.z_position,
        )

        # Check that structures match
        new_structure = tree.tree_structure(new_wavefront)
        chex.assert_equal(structure, new_structure)

    def test_jit_compatibility(self) -> None:
        """Test that OpticalWavefront works with JIT compilation."""

        @jax.jit
        def process_wavefront(wf: OpticalWavefront) -> Complex[Array, " "]:
            """Simple function that processes wavefront."""
            return jnp.sum(wf.field) * wf.wavelength / wf.dx

        # Should work without errors
        result = process_wavefront(self.wavefront)
        chex.assert_shape(result, ())
        chex.assert_tree_all_finite(result)

    def test_grad_compatibility(self) -> None:
        """Test that OpticalWavefront works with gradient computation."""

        def loss_function(wf: OpticalWavefront) -> Float[Array, " "]:
            """Simple loss function for testing gradients."""
            return jnp.sum(jnp.abs(wf.field) ** 2) * wf.wavelength

        # Create gradient function with allow_int=True to handle boolean field
        grad_fn = jax.grad(loss_function, allow_int=True)

        # Compute gradient (should work without errors)
        grad = grad_fn(self.wavefront)

        # Check that gradient has same structure
        chex.assert_equal(type(grad).__name__, "OpticalWavefront")
        chex.assert_shape(grad.field, self.wavefront.field.shape)

    def test_vmap_compatibility(self) -> None:
        """Test that OpticalWavefront works with vmap."""
        # Create batch of wavefronts with different wavelengths
        wavelengths = jnp.array([400e-9, 500e-9, 600e-9])

        def create_wavefront(
            wavelength: Float[Array, " "],
        ) -> OpticalWavefront:
            """Create wavefront with given wavelength."""
            return make_optical_wavefront(
                field=self.field,
                wavelength=wavelength,
                dx=self.dx,
                z_position=self.z_position,
            )

        # Use vmap to create batch
        vmapped_create = jax.vmap(create_wavefront)
        batch = vmapped_create(wavelengths)

        # Check batch dimensions
        chex.assert_shape(batch.field, (3, 64, 64))
        chex.assert_shape(batch.wavelength, (3,))
        chex.assert_shape(batch.dx, (3,))
        chex.assert_shape(batch.z_position, (3,))
        chex.assert_shape(batch.polarization, (3,))

    def test_polarized_field(self) -> None:
        """Test OpticalWavefront with polarized field."""
        # Create polarized field with shape (H, W, 2)
        polarized_field = jnp.ones((64, 64, 2), dtype=complex)
        polarized_field = polarized_field.at[..., 0].set(1 + 0.5j)
        polarized_field = polarized_field.at[..., 1].set(0.5 + 1j)

        # Create polarized wavefront
        polarized_wf = OpticalWavefront(
            field=polarized_field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=self.z_position,
            polarization=jnp.array(True),
        )

        # Test that it's still a valid PyTree
        leaves, treedef = tree.tree_flatten(polarized_wf)
        reconstructed = tree.tree_unflatten(treedef, leaves)

        chex.assert_trees_all_close(reconstructed.field, polarized_wf.field)
        chex.assert_trees_all_close(
            reconstructed.polarization, jnp.array(True)
        )

    def test_tree_all_and_any(self) -> None:
        """Test tree_all and tree_any operations."""
        # Create two wavefronts
        wf1 = self.wavefront
        wf2 = make_optical_wavefront(
            field=self.field * 2,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=self.z_position,
        )

        # Test tree equality check
        equal = tree.tree_all(
            tree.tree_map(
                lambda x, y: (
                    jnp.allclose(x, y)
                    if isinstance(x, jnp.ndarray)
                    else x == y
                ),
                wf1,
                wf1,
            )
        )
        chex.assert_equal(equal, True)

        # Test inequality
        not_equal = tree.tree_all(
            tree.tree_map(
                lambda x, y: (
                    jnp.allclose(x, y)
                    if isinstance(x, jnp.ndarray)
                    else x == y
                ),
                wf1,
                wf2,
            )
        )
        chex.assert_equal(not_equal, False)

    def test_custom_tree_methods(self) -> None:
        """Test the custom tree_flatten and tree_unflatten methods."""
        # Use the custom methods directly
        children, aux_data = self.wavefront.tree_flatten()

        # aux_data should be None
        chex.assert_equal(aux_data, None)

        # children should be a tuple of 5 elements
        chex.assert_equal(len(children), 5)

        # Reconstruct using custom method
        reconstructed = OpticalWavefront.tree_unflatten(aux_data, children)

        # Verify reconstruction
        chex.assert_trees_all_close(reconstructed.field, self.wavefront.field)
        chex.assert_trees_all_close(
            reconstructed.wavelength, self.wavefront.wavelength
        )
        chex.assert_trees_all_close(reconstructed.dx, self.wavefront.dx)
        chex.assert_trees_all_close(
            reconstructed.z_position, self.wavefront.z_position
        )
        chex.assert_trees_all_close(
            reconstructed.polarization, self.wavefront.polarization
        )


class TestLensParamsPyTree(chex.TestCase):
    """Test that LensParams behaves correctly as a PyTree."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.lens_params = make_lens_params(
            focal_length=0.01,
            diameter=0.00005,
            n=1.5,
            center_thickness=0.001,
            r1=0.01,
            r2=0.01,
        )

    def test_is_pytree(self) -> None:
        """Test that LensParams is recognized as a PyTree."""
        leaves, treedef = tree.tree_flatten(self.lens_params)
        reconstructed = tree.tree_unflatten(treedef, leaves)

        chex.assert_trees_all_close(
            reconstructed.focal_length, self.lens_params.focal_length
        )
        chex.assert_trees_all_close(
            reconstructed.diameter, self.lens_params.diameter
        )
        chex.assert_trees_all_close(reconstructed.n, self.lens_params.n)
        chex.assert_trees_all_close(
            reconstructed.center_thickness,
            self.lens_params.center_thickness,
        )
        chex.assert_trees_all_close(reconstructed.r1, self.lens_params.r1)
        chex.assert_trees_all_close(reconstructed.r2, self.lens_params.r2)

    def test_tree_map(self) -> None:
        """Test that tree_map works correctly on LensParams."""
        scaled = tree.tree_map(lambda x: x * 2, self.lens_params)

        chex.assert_trees_all_close(
            scaled.focal_length, self.lens_params.focal_length * 2
        )
        chex.assert_trees_all_close(
            scaled.diameter, self.lens_params.diameter * 2
        )

    def test_jit_compatibility(self) -> None:
        """Test that LensParams works with JIT compilation."""

        @jax.jit
        def compute_power(lens: LensParams) -> Float[Array, " "]:
            return 1.0 / lens.focal_length

        result = compute_power(self.lens_params)
        chex.assert_shape(result, ())
        chex.assert_tree_all_finite(result)

    def test_vmap_compatibility(self) -> None:
        """Test that LensParams works with vmap."""
        focal_lengths = jnp.array([0.01, 0.02, 0.03])

        def create_lens(f: Float[Array, " "]) -> LensParams:
            # For vmap, we can't use factory functions that require concrete values
            # So we use direct instantiation here
            return LensParams(
                focal_length=f,
                diameter=self.lens_params.diameter,
                n=self.lens_params.n,
                center_thickness=self.lens_params.center_thickness,
                r1=self.lens_params.r1,
                r2=self.lens_params.r2,
            )

        vmapped_create = jax.vmap(create_lens)
        batch = vmapped_create(focal_lengths)

        chex.assert_shape(batch.focal_length, (3,))
        chex.assert_shape(batch.diameter, (3,))


class TestGridParamsPyTree(chex.TestCase):
    """Test that GridParams behaves correctly as a PyTree."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        nx, ny = 64, 64
        x = jnp.linspace(-1, 1, nx)
        y = jnp.linspace(-1, 1, ny)
        xx, yy = jnp.meshgrid(x, y)

        self.grid_params = make_grid_params(
            xx=xx,
            yy=yy,
            phase_profile=jnp.zeros((ny, nx)),
            transmission=jnp.ones((ny, nx)),
        )

    def test_is_pytree(self) -> None:
        """Test that GridParams is recognized as a PyTree."""
        leaves, treedef = tree.tree_flatten(self.grid_params)
        reconstructed = tree.tree_unflatten(treedef, leaves)

        chex.assert_trees_all_close(reconstructed.xx, self.grid_params.xx)
        chex.assert_trees_all_close(reconstructed.yy, self.grid_params.yy)
        chex.assert_trees_all_close(
            reconstructed.phase_profile, self.grid_params.phase_profile
        )
        chex.assert_trees_all_close(
            reconstructed.transmission, self.grid_params.transmission
        )

    def test_tree_map(self) -> None:
        """Test that tree_map works correctly on GridParams."""
        scaled = tree.tree_map(lambda x: x * 2, self.grid_params)

        chex.assert_trees_all_close(scaled.xx, self.grid_params.xx * 2)
        chex.assert_trees_all_close(scaled.yy, self.grid_params.yy * 2)
        chex.assert_trees_all_close(
            scaled.phase_profile, self.grid_params.phase_profile * 2
        )

    def test_jit_compatibility(self) -> None:
        """Test that GridParams works with JIT compilation."""

        @jax.jit
        def compute_radial(grid: GridParams) -> Float[Array, "64 64"]:
            return jnp.sqrt(grid.xx**2 + grid.yy**2)

        result = compute_radial(self.grid_params)
        chex.assert_shape(result, (64, 64))
        chex.assert_tree_all_finite(result)


class TestMicroscopeDataPyTree(chex.TestCase):
    """Test that MicroscopeData behaves correctly as a PyTree."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        # Test with 3D data
        self.data_3d = make_microscope_data(
            image_data=jnp.ones((10, 64, 64)),
            positions=jnp.zeros((10, 2)),
            wavelength=532e-9,
            dx=1e-6,
        )

        # Test with 4D data
        self.data_4d = make_microscope_data(
            image_data=jnp.ones((5, 5, 64, 64)),
            positions=jnp.zeros((25, 2)),
            wavelength=532e-9,
            dx=1e-6,
        )

    def test_is_pytree_3d(self) -> None:
        """Test that MicroscopeData with 3D data is a PyTree."""
        leaves, treedef = tree.tree_flatten(self.data_3d)
        reconstructed = tree.tree_unflatten(treedef, leaves)

        chex.assert_trees_all_close(
            reconstructed.image_data, self.data_3d.image_data
        )
        chex.assert_trees_all_close(
            reconstructed.positions, self.data_3d.positions
        )
        chex.assert_trees_all_close(
            reconstructed.wavelength, self.data_3d.wavelength
        )
        chex.assert_trees_all_close(reconstructed.dx, self.data_3d.dx)

    def test_is_pytree_4d(self) -> None:
        """Test that MicroscopeData with 4D data is a PyTree."""
        leaves, treedef = tree.tree_flatten(self.data_4d)
        reconstructed = tree.tree_unflatten(treedef, leaves)

        chex.assert_trees_all_close(
            reconstructed.image_data, self.data_4d.image_data
        )

    def test_tree_map(self) -> None:
        """Test that tree_map works correctly on MicroscopeData."""
        scaled = tree.tree_map(lambda x: x * 2, self.data_3d)

        chex.assert_trees_all_close(
            scaled.image_data, self.data_3d.image_data * 2
        )
        chex.assert_trees_all_close(
            scaled.wavelength, self.data_3d.wavelength * 2
        )

    def test_jit_compatibility(self) -> None:
        """Test that MicroscopeData works with JIT compilation."""

        @jax.jit
        def compute_mean(data: MicroscopeData) -> Float[Array, " "]:
            return jnp.mean(data.image_data)

        result = compute_mean(self.data_3d)
        chex.assert_shape(result, ())
        chex.assert_tree_all_finite(result)


class TestSampleFunctionPyTree(chex.TestCase):
    """Test that SampleFunction behaves correctly as a PyTree."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.sample_func = make_sample_function(
            sample=jnp.ones((64, 64), dtype=complex) * (1 + 0.5j),
            dx=1e-6,
        )

    def test_is_pytree(self) -> None:
        """Test that SampleFunction is recognized as a PyTree."""
        leaves, treedef = tree.tree_flatten(self.sample_func)
        reconstructed = tree.tree_unflatten(treedef, leaves)

        chex.assert_trees_all_close(
            reconstructed.sample, self.sample_func.sample
        )
        chex.assert_trees_all_close(reconstructed.dx, self.sample_func.dx)

    def test_tree_map(self) -> None:
        """Test that tree_map works correctly on SampleFunction."""
        scaled = tree.tree_map(lambda x: x * 2, self.sample_func)

        chex.assert_trees_all_close(scaled.sample, self.sample_func.sample * 2)
        chex.assert_trees_all_close(scaled.dx, self.sample_func.dx * 2)

    def test_jit_compatibility(self) -> None:
        """Test that SampleFunction works with JIT compilation."""

        @jax.jit
        def compute_amplitude(
            sample: SampleFunction,
        ) -> Float[Array, "64 64"]:
            return jnp.abs(sample.sample)

        result = compute_amplitude(self.sample_func)
        chex.assert_shape(result, (64, 64))
        chex.assert_tree_all_finite(result)

    def test_grad_compatibility(self) -> None:
        """Test gradient computation with SampleFunction."""

        def loss_function(sample: SampleFunction) -> Float[Array, " "]:
            return jnp.sum(jnp.abs(sample.sample) ** 2) * sample.dx

        grad_fn = jax.grad(loss_function)
        grad = grad_fn(self.sample_func)

        chex.assert_equal(type(grad).__name__, "SampleFunction")
        chex.assert_shape(grad.sample, self.sample_func.sample.shape)


class TestDiffractogramPyTree(chex.TestCase):
    """Test that Diffractogram behaves correctly as a PyTree."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.diffractogram = make_diffractogram(
            image=jnp.ones((128, 128)),
            wavelength=632e-9,
            dx=1e-6,
        )

    def test_is_pytree(self) -> None:
        """Test that Diffractogram is recognized as a PyTree."""
        leaves, treedef = tree.tree_flatten(self.diffractogram)
        reconstructed = tree.tree_unflatten(treedef, leaves)

        chex.assert_trees_all_close(
            reconstructed.image, self.diffractogram.image
        )
        chex.assert_trees_all_close(
            reconstructed.wavelength, self.diffractogram.wavelength
        )
        chex.assert_trees_all_close(reconstructed.dx, self.diffractogram.dx)

    def test_tree_map(self) -> None:
        """Test that tree_map works correctly on Diffractogram."""
        scaled = tree.tree_map(lambda x: x * 2, self.diffractogram)

        chex.assert_trees_all_close(scaled.image, self.diffractogram.image * 2)
        chex.assert_trees_all_close(
            scaled.wavelength, self.diffractogram.wavelength * 2
        )

    def test_jit_compatibility(self) -> None:
        """Test that Diffractogram works with JIT compilation."""

        @jax.jit
        def compute_total_intensity(
            diff: Diffractogram,
        ) -> Float[Array, " "]:
            return jnp.sum(diff.image)

        result = compute_total_intensity(self.diffractogram)
        chex.assert_shape(result, ())
        chex.assert_tree_all_finite(result)


class TestOptimizerStatePyTree(chex.TestCase):
    """Test that OptimizerState behaves correctly as a PyTree."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        shape = (64, 64)
        self.optimizer_state = make_optimizer_state(
            shape=shape,
            m=jnp.zeros(shape, dtype=complex),
            v=jnp.zeros(shape, dtype=float),
            step=0,
        )

    def test_is_pytree(self) -> None:
        """Test that OptimizerState is recognized as a PyTree."""
        leaves, treedef = tree.tree_flatten(self.optimizer_state)
        reconstructed = tree.tree_unflatten(treedef, leaves)

        chex.assert_trees_all_close(reconstructed.m, self.optimizer_state.m)
        chex.assert_trees_all_close(reconstructed.v, self.optimizer_state.v)
        chex.assert_trees_all_close(
            reconstructed.step, self.optimizer_state.step
        )

    def test_tree_map(self) -> None:
        """Test that tree_map works correctly on OptimizerState."""

        def scale_non_int(x: Array) -> Array:
            """Scale non-integer arrays by 2, increment integers."""
            if hasattr(x, "dtype") and x.dtype in [jnp.int32, jnp.int64]:
                return x + 1  # Increment integers
            return x * 2  # Scale others

        scaled = tree.tree_map(scale_non_int, self.optimizer_state)

        chex.assert_trees_all_close(scaled.m, self.optimizer_state.m * 2)
        chex.assert_trees_all_close(scaled.v, self.optimizer_state.v * 2)
        chex.assert_trees_all_close(scaled.step, self.optimizer_state.step + 1)

    def test_jit_compatibility(self) -> None:
        """Test that OptimizerState works with JIT compilation."""

        @jax.jit
        def update_state(state: OptimizerState) -> OptimizerState:
            # Use direct instantiation for JIT compatibility
            return OptimizerState(
                m=state.m * 0.9,
                v=state.v * 0.999,
                step=state.step + 1,
            )

        result = update_state(self.optimizer_state)
        chex.assert_equal(type(result).__name__, "OptimizerState")
        chex.assert_shape(result.step, ())


class TestPtychographyParamsPyTree(chex.TestCase):
    """Test that PtychographyParams behaves correctly as a PyTree."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.ptycho_params = make_ptychography_params(
            zoom_factor=2.0,
            aperture_diameter=1e-3,
            travel_distance=0.1,
            aperture_center=jnp.array([0.0, 0.0]),
            camera_pixel_size=5e-6,
            learning_rate=0.01,
            num_iterations=100,
        )

    def test_is_pytree(self) -> None:
        """Test that PtychographyParams is recognized as a PyTree."""
        leaves, treedef = tree.tree_flatten(self.ptycho_params)
        reconstructed = tree.tree_unflatten(treedef, leaves)

        chex.assert_trees_all_close(
            reconstructed.zoom_factor, self.ptycho_params.zoom_factor
        )
        chex.assert_trees_all_close(
            reconstructed.aperture_diameter,
            self.ptycho_params.aperture_diameter,
        )
        chex.assert_trees_all_close(
            reconstructed.travel_distance,
            self.ptycho_params.travel_distance,
        )
        chex.assert_trees_all_close(
            reconstructed.aperture_center,
            self.ptycho_params.aperture_center,
        )
        chex.assert_trees_all_close(
            reconstructed.camera_pixel_size,
            self.ptycho_params.camera_pixel_size,
        )
        chex.assert_trees_all_close(
            reconstructed.learning_rate, self.ptycho_params.learning_rate
        )
        chex.assert_trees_all_close(
            reconstructed.num_iterations,
            self.ptycho_params.num_iterations,
        )

    def test_tree_map(self) -> None:
        """Test that tree_map works correctly on PtychographyParams."""

        def scale_floats(x: Array) -> Array:
            """Scale float arrays by 2, keep integers unchanged."""
            if hasattr(x, "dtype") and x.dtype in [jnp.int32, jnp.int64]:
                return x  # Don't scale integers
            return x * 2

        scaled = tree.tree_map(scale_floats, self.ptycho_params)

        chex.assert_trees_all_close(
            scaled.zoom_factor, self.ptycho_params.zoom_factor * 2
        )
        chex.assert_trees_all_close(
            scaled.learning_rate, self.ptycho_params.learning_rate * 2
        )
        # num_iterations should not be scaled
        chex.assert_trees_all_close(
            scaled.num_iterations, self.ptycho_params.num_iterations
        )

    def test_jit_compatibility(self) -> None:
        """Test that PtychographyParams works with JIT compilation."""

        @jax.jit
        def compute_magnification(
            params: PtychographyParams,
        ) -> Float[Array, " "]:
            return params.zoom_factor * params.travel_distance

        result = compute_magnification(self.ptycho_params)
        chex.assert_shape(result, ())
        chex.assert_tree_all_finite(result)

    def test_vmap_compatibility(self) -> None:
        """Test that PtychographyParams works with vmap."""
        zoom_factors = jnp.array([1.0, 2.0, 3.0])

        def create_params(zoom: Float[Array, " "]) -> PtychographyParams:
            # For vmap, we can't use factory functions that require concrete values
            # So we use direct instantiation here
            return PtychographyParams(
                zoom_factor=zoom,
                aperture_diameter=self.ptycho_params.aperture_diameter,
                travel_distance=self.ptycho_params.travel_distance,
                aperture_center=self.ptycho_params.aperture_center,
                camera_pixel_size=self.ptycho_params.camera_pixel_size,
                learning_rate=self.ptycho_params.learning_rate,
                num_iterations=self.ptycho_params.num_iterations,
            )

        vmapped_create = jax.vmap(create_params)
        batch = vmapped_create(zoom_factors)

        chex.assert_shape(batch.zoom_factor, (3,))
        chex.assert_shape(batch.aperture_center, (3, 2))


if __name__ == "__main__":
    pytest.main([__file__])
